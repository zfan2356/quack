# Copyright (c) 2025, Tri Dao.
from typing import Optional

import torch
from torch import Tensor

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute.runtime import from_dlpack, make_ptr

from quack.dense_gemm_sm90 import TileSchedulerOptions
from quack.gemm_act_sm90 import GemmActSm90
from quack.cute_dsl_utils import torch2cute_dtype_map, get_max_active_clusters
import quack.activation


class GemmDActSm90(GemmActSm90):
    # Different from GemmActSm90, here act_bwd_fn must take in 2 arguments (x, dout)
    # and return 2 arguments (dx, out)
    EpilogueArguments = GemmActSm90.EpilogueArguments
    EpilogueParams = GemmActSm90.EpilogueParams

    @cute.jit
    def epi_visit_acc_subtile(
        self,
        params: EpilogueParams,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        assert tRS_rC is not None
        tRS_rC_acc = cute.make_fragment_like(tRS_rC, self.acc_dtype)
        tRS_rC_acc.store(tRS_rC.load().to(self.acc_dtype))
        # If we don't have .shape here, the compiler generates local stores and loads
        if const_expr(params.act_fn is not None):
            tRS_rPostAct = cute.make_fragment(tRS_rD.layout.shape, self.acc_dtype)
            for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                tRS_rD[i], tRS_rPostAct[i] = params.act_fn(tRS_rC_acc[i], tRS_rD[i])
        else:
            tRS_rPostAct = tRS_rC_acc
        # Type conversion
        tRS_rPostAct_out = cute.make_fragment_like(tRS_rPostAct, self.postact_dtype)
        tRS_rPostAct_out.store(tRS_rPostAct.load().to(self.postact_dtype))
        return tRS_rPostAct_out


dact_fn_map = {
    None: None,
    "relu": quack.activation.drelu,
    "relu_sq": quack.activation.drelu_sq,
    "gelu_tanh_approx": quack.activation.dgelu_tanh_approx,
}


def gemm_dact_sm90(
    A: Tensor,  # (l, m, k)
    B: Tensor,  # (l, n, k)
    Out: Tensor,  # (l, m, n)
    PreAct: Tensor,  # (l, m, n)
    PostAct: Tensor,  # (l, m, n)
    tile_count_semaphore: Optional[Tensor],  # (1,)
    activation: Optional[str],
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    pingpong: bool = True,
    persistent: bool = True,
) -> None:
    assert A.dim() == 3 and A.is_cuda, "A must be A 3D CUDA tensor"
    L, M, K = A.shape
    assert A.dtype in torch2cute_dtype_map, "Unsupported dtype for A"
    assert B.dim() == 3 and B.is_cuda, "B must be A 3D CUDA tensor"
    _, N, _ = B.shape
    assert B.dtype == A.dtype
    assert B.shape == (L, N, K), f"B must have shape {(L, N, K)}, got {B.shape}"
    assert Out.dim() == 3 and Out.is_cuda, "Out must be A 3D CUDA tensor"
    assert Out.dtype in torch2cute_dtype_map, "Unsupported dtype for Out"
    assert Out.shape == (L, M, N), f"Out must have shape {(L, M, N)}, got {Out.shape}"
    assert PostAct.dim() == 3 and PostAct.is_cuda, "PostAct must be A 3D CUDA tensor"
    assert PostAct.dtype in torch2cute_dtype_map, "Unsupported dtype for PostAct"
    assert PostAct.shape == (L, M, N), f"PostAct must have shape {(L, M, N)}, got {PostAct.shape}"
    assert PreAct.dim() == 3 and PreAct.is_cuda, "PreAct must be A 3D CUDA tensor"
    assert PreAct.shape == (L, M, N), f"PreAct must have shape {(L, M, N)}, got {PreAct.shape}"
    assert PreAct.dtype in torch2cute_dtype_map, "Unsupported dtype for PreAct"
    assert activation in dact_fn_map, f"Unsupported activation {activation}"
    A, B, Out, PreAct, PostAct = [
        x.permute(1, 2, 0) if x is not None else None for x in (A, B, Out, PreAct, PostAct)
    ]  # (m, k, l), (n, k, l), (m, n, l)

    a_dtype = torch2cute_dtype_map[A.dtype]
    b_dtype = a_dtype
    d_dtype = torch2cute_dtype_map[Out.dtype] if Out is not None else None
    postact_dtype = torch2cute_dtype_map[PostAct.dtype]
    c_dtype = torch2cute_dtype_map[PreAct.dtype] if PreAct is not None else None
    acc_dtype = cutlass.Float32
    tile_shape_mnk = (tile_M, tile_N, 64)  # TODO: adjust for fp8
    cluster_shape_mnk = (cluster_M, cluster_N, 1)
    a_major = "k" if A.stride(1) == 1 else "m"
    b_major = "k" if B.stride(1) == 1 else "n"
    d_major = "n" if Out is not None and Out.stride(1) == 1 else "m"
    postact_major = "n" if PostAct.stride(1) == 1 else "m"
    c_major = "n" if PreAct is not None and PreAct.stride(1) == 1 else "m"
    if not GemmDActSm90.is_valid_dtypes(a_dtype, b_dtype, acc_dtype, d_dtype, a_major, b_major):
        raise TypeError(
            f"Skipping due to unsupported combination of types and majors: {a_dtype}, {b_dtype}, {acc_dtype}, {d_dtype}, {a_major=}, {b_major=}"
        )
    if persistent:
        max_active_clusters = get_max_active_clusters(cluster_M * cluster_N)
    else:
        max_active_clusters = 0
    mA = from_dlpack(A.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=1 if a_major == "k" else 0
    )
    mB = from_dlpack(B.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=1 if b_major == "k" else 0
    )
    mOut = from_dlpack(Out.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=1 if d_major == "n" else 0
    )
    mPostAct = from_dlpack(PostAct.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=1 if postact_major == "n" else 0
    )
    mPreAct = from_dlpack(PreAct.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=1 if c_major == "n" else 0
    )

    act_fn = dact_fn_map[activation]
    epi_args = GemmDActSm90.EpilogueArguments(mPostAct, act_fn)
    scheduler_args = TileSchedulerOptions(
        Int32(max_active_clusters),
        tile_count_semaphore=make_ptr(
            Int32, tile_count_semaphore.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        )
        if tile_count_semaphore is not None
        else None,
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (
        a_dtype,
        d_dtype,
        postact_dtype,
        c_dtype,
        activation,
        tile_shape_mnk,
        cluster_shape_mnk,
        a_major,
        b_major,
        d_major,
        c_major,
        pingpong,
        persistent,
        tile_count_semaphore is not None,
    )

    cache = gemm_dact_sm90.compile_cache
    if compile_key not in cache:
        gemm = GemmDActSm90(
            acc_dtype,
            a_dtype,
            tile_shape_mnk,
            cluster_shape_mnk,
            pingpong=pingpong,
            is_persistent=persistent,
        )
        cache[compile_key] = cute.compile(
            gemm,
            mA,
            mB,
            mOut,
            mPreAct,
            epi_args,
            scheduler_args,
            None,  # varlen_args
            None,  # mAIdx
            current_stream,
        )
    cache[compile_key](mA, mB, mOut, mPreAct, epi_args, scheduler_args, None, None, current_stream)


gemm_dact_sm90.compile_cache = {}
