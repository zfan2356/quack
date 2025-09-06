# Copyright (c) 2025, Tri Dao.
from typing import Tuple, Optional, Callable
from dataclasses import dataclass

import torch
from torch import Tensor

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warpgroup
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Int32, Boolean, const_expr
from cutlass.cute.runtime import from_dlpack, make_ptr

from quack.cute_dsl_utils import ArgumentsBase, ParamsBase
from quack.dense_gemm_sm90 import GemmSm90, TileSchedulerOptions
from quack.cute_dsl_utils import torch2cute_dtype_map, get_max_active_clusters
import quack.activation


class GemmActSm90(GemmSm90):
    @dataclass
    class EpilogueArguments(ArgumentsBase):
        mPostAct: cute.Tensor
        act_fn: cutlass.Constexpr[Optional[Callable]] = None

    @dataclass
    class EpilogueParams(ParamsBase):
        tma_atom_postact: cute.CopyAtom
        mPostAct_mnl: cute.Tensor
        epi_postact_smem_layout_staged: cute.ComposedLayout
        act_fn: cutlass.Constexpr[Optional[Callable]] = None

    def epi_to_underlying_arguments(
        self, args: EpilogueArguments, *, loc=None, ip=None
    ) -> EpilogueParams:
        self.postact_dtype = args.mPostAct.element_type
        self.postact_layout = cutlass.utils.LayoutEnum.from_tensor(args.mPostAct)

        self.tile_shape_postact_mn = self.tile_shape_mnk[:2]
        self.epi_tile_postact = self.epi_tile
        postact_major_mode_size = (
            self.epi_tile_postact[1]
            if self.postact_layout.is_n_major_c()
            else self.epi_tile_postact[0]
        )
        postact_smem_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                self.postact_layout, self.postact_dtype, postact_major_mode_size
            ),
            self.postact_dtype,
        )
        epi_postact_smem_layout_staged = cute.tile_to_shape(
            postact_smem_layout_atom,
            cute.append(self.epi_tile_postact, self.epi_stage),
            order=(0, 1, 2),
        )
        tma_atom_postact, tma_tensor_postact = self._make_tma_epi_atoms_and_tensors(
            args.mPostAct,
            epi_postact_smem_layout_staged,
            self.epi_tile_postact,
            store_or_load="store",
        )
        return GemmActSm90.EpilogueParams(
            tma_atom_postact, tma_tensor_postact, epi_postact_smem_layout_staged, args.act_fn
        )

    @staticmethod
    def epi_smem_bytes_per_stage(
        args: EpilogueArguments,
        tile_shape_mnk: Tuple[int, int, int],
        epi_tile: Tuple[int, int],
    ) -> int:
        postact_dtype = args.mPostAct.element_type
        postact_bytes_per_stage = cute.size(epi_tile) * (postact_dtype.width // 8)
        return postact_bytes_per_stage

    def epi_get_smem_struct(self, params: EpilogueParams):
        @cute.struct
        class EpiSharedStorage:
            sPostAct: cute.struct.Align[
                cute.struct.MemRange[
                    self.postact_dtype, cute.cosize(params.epi_postact_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        return EpiSharedStorage

    def epi_get_smem_tensors(self, params: EpilogueParams, storage) -> Tuple[cute.Tensor, ...]:
        sPostAct = storage.epi.sPostAct.get_tensor(
            params.epi_postact_smem_layout_staged.outer,
            swizzle=params.epi_postact_smem_layout_staged.inner,
        )
        return (sPostAct,)

    @cute.jit
    def epilogue(
        self,
        params: EpilogueParams,
        epi_smem_tensors: Tuple[cute.Tensor, ...],
        epi_pipeline: cutlass.pipeline.PipelineAsync,
        epi_read_state: cutlass.pipeline.PipelineState,
        epi_producer_state: cutlass.pipeline.PipelineState,
        tiled_mma: cute.TiledMma,
        tRS_rAcc: cute.Tensor,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor],
        tiled_copy_r2s: cute.core.ThrCopy,
        tRS_sD: cute.Tensor,
        tiled_copy_s2r: Optional[cute.core.ThrCopy],
        tSR_rC: Optional[cute.Tensor],
        tSR_sC: Optional[cute.Tensor],
        copy_D: Optional[Callable],
        bSG_sD: cute.Tensor,
        bSG_gD: cute.Tensor,
        epi_load_g2s: Optional[Callable],
        tile_coord_mnkl: cute.Coord,
        cu_seqlens_m: Optional[cute.Tensor],
        epilogue_barrier: cutlass.pipeline.NamedBarrier,
        tile_scheduler,
        tidx: Int32,
        is_tma_warp: Boolean,
    ) -> Tuple[cutlass.pipeline.PipelineState, cutlass.pipeline.PipelineState]:
        has_C = const_expr(tRS_rC is not None)
        has_D = const_expr(copy_D is not None)
        assert cu_seqlens_m is None, "GemmActSm90 doesn't support varlen_m for now"

        tma_atom_postact = params.tma_atom_postact
        mPostAct_mnl = params.mPostAct_mnl
        (sPostAct,) = epi_smem_tensors
        tiled_copy_C_atom = self.epilog_smem_copy_atom(tiled_mma)
        copy_atom_postact_r2s = sm90_utils.sm90_get_smem_store_op(
            self.postact_layout, elem_ty_d=self.postact_dtype, elem_ty_acc=self.acc_dtype
        )
        tiled_copy_postact_r2s = cute.make_tiled_copy_S(copy_atom_postact_r2s, tiled_copy_C_atom)
        thr_copy_postact_r2s = tiled_copy_postact_r2s.get_slice(tidx)
        tRS_sPostAct = thr_copy_postact_r2s.partition_D(sPostAct)
        bSG_sPostAct, bSG_gPostAct = self.epilog_gmem_copy_and_partition(
            tma_atom_postact,
            mPostAct_mnl,
            self.tile_shape_postact_mn,
            self.epi_tile_postact,
            sPostAct,
            tile_coord_mnkl,
            cu_seqlens_m,
        )

        # We iterate over epi tiles in the N dimension first before the M dimension
        epi_tile_layout = cute.make_layout(
            bSG_gPostAct.shape[1], stride=(bSG_gPostAct.shape[1][1], 1)
        )
        epi_tile_num = cute.size(bSG_gPostAct.shape[1])
        num_prev_subtiles = tile_scheduler.num_tiles_executed * epi_tile_num

        if const_expr(epi_load_g2s is not None):
            for epi_idx in cutlass.range(min(epi_tile_num, self.epi_c_stage), unroll=1):
                epi_producer_state = epi_load_g2s(epi_producer_state, epi_idx, is_tma_warp)

        for epi_idx in cutlass.range_constexpr(epi_tile_num):
            # Copy from acc to D registers
            for epi_v in cutlass.range_constexpr(cute.size(tRS_rD)):
                tRS_rD[epi_v] = tRS_rAcc[epi_idx * cute.size(tRS_rD) + epi_v]
            if const_expr(has_C):
                epi_pipeline.consumer_wait(epi_read_state)
                cute.copy(tiled_copy_s2r, tSR_sC[None, None, None, epi_read_state.index], tSR_rC)
                # Fence to make sure shared memory read is visible to TMA load
                cute.arch.fence_proxy(
                    cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
                )
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    epi_pipeline.consumer_release(epi_read_state)
                epi_read_state.advance()
            if const_expr(epi_load_g2s is not None and epi_idx + self.epi_c_stage < epi_tile_num):
                epi_producer_state = epi_load_g2s(
                    epi_producer_state, epi_idx + self.epi_c_stage, is_tma_warp
                )
            tRS_rPostAct = self.epi_visit_acc_subtile(params, tRS_rD, tRS_rC)
            epi_buffer = (num_prev_subtiles + epi_idx) % cute.size(tRS_sPostAct, mode=[3])
            # Copy from D registers to shared memory
            if const_expr(has_D):
                # Type conversion
                tRS_rD_out = cute.make_fragment_like(tRS_rD, self.d_dtype)
                tRS_rD_out.store(tRS_rD.load().to(self.d_dtype))
                cute.copy(tiled_copy_r2s, tRS_rD_out, tRS_sD[None, None, None, epi_buffer])
            cute.copy(
                tiled_copy_postact_r2s,
                tiled_copy_postact_r2s.retile(tRS_rPostAct),
                tRS_sPostAct[None, None, None, epi_buffer],
            )
            # Fence and barrier to make sure shared memory store is visible to TMA store
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
            )
            epilogue_barrier.arrive_and_wait()
            # Get the global memory coordinate for the current epi tile
            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            # Copy from shared memory to global memory
            if is_tma_warp:
                if const_expr(has_D):
                    copy_D(bSG_sD[None, epi_buffer], bSG_gD[None, gmem_coord])
                cute.copy(
                    tma_atom_postact,
                    bSG_sPostAct[None, epi_buffer],
                    bSG_gPostAct[None, gmem_coord],
                )
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(self.epi_stage - 1, read=True)
            epilogue_barrier.arrive_and_wait()

        return epi_read_state, epi_producer_state

    @cute.jit
    def epi_visit_acc_subtile(
        self,
        params: EpilogueParams,
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        if const_expr(tRS_rC is not None):
            tRS_rD.store(tRS_rD.load() + tRS_rC.load().to(tRS_rD.element_type))
        # If we don't have .shape here, the compiler generates local stores and loads
        if const_expr(params.act_fn is not None):
            tRS_rPostAct = cute.make_fragment(tRS_rD.layout.shape, self.acc_dtype)
            for i in cutlass.range(cute.size(tRS_rPostAct), unroll_full=True):
                tRS_rPostAct[i] = params.act_fn(tRS_rD[i])
        else:
            tRS_rPostAct = tRS_rD
        # Type conversion
        tRS_rPostAct_out = cute.make_fragment_like(tRS_rPostAct, self.postact_dtype)
        tRS_rPostAct_out.store(tRS_rPostAct.load().to(self.postact_dtype))
        return tRS_rPostAct_out


act_fn_map = {
    None: None,
    "relu": quack.activation.relu,
    "relu_sq": quack.activation.relu_sq,
    "gelu_tanh_approx": quack.activation.gelu_tanh_approx,
}


def gemm_act_sm90(
    A: Tensor,  # (l, m, k)
    B: Tensor,  # (l, n, k)
    D: Optional[Tensor],  # (l, m, n)
    C: Optional[Tensor],  # (l, m, n)
    PostAct: Tensor,  # (l, m, n)
    activation: Optional[str],
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    pingpong: bool = False,
    persistent: bool = True,
) -> None:
    tile_count_semaphore = None
    assert A.dim() == 3 and A.is_cuda, "A must be A 3D CUDA tensor"
    L, M, K = A.shape
    assert A.dtype in torch2cute_dtype_map, "Unsupported dtype for A"
    assert B.dim() == 3 and B.is_cuda, "B must be A 3D CUDA tensor"
    _, N, _ = B.shape
    assert B.dtype == A.dtype
    assert B.shape == (L, N, K), f"B must have shape {(L, N, K)}, got {B.shape}"
    if D is not None:
        assert D.dim() == 3 and D.is_cuda, "D must be A 3D CUDA tensor"
        assert D.dtype in torch2cute_dtype_map, "Unsupported dtype for D"
        assert D.shape == (L, M, N), f"D must have shape {(L, M, N)}, got {D.shape}"
    assert PostAct.dim() == 3 and PostAct.is_cuda, "PostAct must be A 3D CUDA tensor"
    assert PostAct.dtype in torch2cute_dtype_map, "Unsupported dtype for PostAct"
    assert PostAct.shape == (L, M, N), f"PostAct must have shape {(L, M, N)}, got {PostAct.shape}"
    if C is not None:
        assert C.dim() == 3 and C.is_cuda, "C must be A 3D CUDA tensor"
        assert C.shape == (L, M, N), f"C must have shape {(L, M, N)}, got {C.shape}"
        assert C.dtype in torch2cute_dtype_map, "Unsupported dtype for C"
    assert activation in act_fn_map, f"Unsupported activation {activation}"
    A, B, D, C, PostAct = [
        x.permute(1, 2, 0) if x is not None else None for x in (A, B, D, C, PostAct)
    ]  # (m, k, l), (n, k, l), (m, n, l)

    a_dtype = torch2cute_dtype_map[A.dtype]
    b_dtype = a_dtype
    d_dtype = torch2cute_dtype_map[D.dtype] if D is not None else None
    postact_dtype = torch2cute_dtype_map[PostAct.dtype]
    c_dtype = torch2cute_dtype_map[C.dtype] if C is not None else None
    acc_dtype = cutlass.Float32
    tile_shape_mnk = (tile_M, tile_N, 64)  # TODO: adjust for fp8
    cluster_shape_mnk = (cluster_M, cluster_N, 1)
    a_major = "k" if A.stride(1) == 1 else "m"
    b_major = "k" if B.stride(1) == 1 else "n"
    d_major = "n" if D is not None and D.stride(1) == 1 else "m"
    postact_major = "n" if PostAct.stride(1) == 1 else "m"
    c_major = "n" if C is not None and C.stride(1) == 1 else "m"
    if not GemmActSm90.is_valid_dtypes(a_dtype, b_dtype, acc_dtype, d_dtype, a_major, b_major):
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
    mD = (
        from_dlpack(D.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=1 if d_major == "n" else 0
        )
        if D is not None
        else None
    )
    mPostAct = from_dlpack(PostAct.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=1 if postact_major == "n" else 0
    )
    mC = (
        from_dlpack(C.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=1 if c_major == "n" else 0
        )
        if C is not None
        else None
    )
    act_fn = act_fn_map[activation]
    epi_args = GemmActSm90.EpilogueArguments(mPostAct, act_fn)
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

    cache = gemm_act_sm90.compile_cache
    if compile_key not in cache:
        gemm = GemmActSm90(
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
            mD,
            mC,
            epi_args,
            scheduler_args,
            None,  # varlen_args
            None,  # mAIdx
            current_stream,
        )
    cache[compile_key](mA, mB, mD, mC, epi_args, scheduler_args, None, None, current_stream)


gemm_act_sm90.compile_cache = {}
