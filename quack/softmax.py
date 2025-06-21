import math
import torch
from typing import Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import quack.utils as utils
from quack.reduction_base import ReductionBase, torch2cute_dtype_map


class Softmax(ReductionBase):
    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        # 2 stages: 1 for max, 1 for sum
        super().__init__(dtype, N, stage=2)

    def _calculate_threads_per_row(self):
        N = self.N
        return (
            8
            if N <= 64
            else (
                16
                if N <= 128
                else (32 if N <= 3072 else (64 if N <= 6144 else (128 if N <= 16384 else 256)))
            )
        )

    def _set_cluster_n(self):
        N = self.N
        if cutlass.const_expr(self.dtype.width == 16):
            cluster_n = (
                1
                if N <= 16 * 1024
                else (
                    2
                    if N <= 32 * 1024
                    else (4 if N <= 64 * 1024 else (8 if N <= 128 * 1024 else 16))
                )
            )
        else:  # fp32
            cluster_n = (
                1
                if N <= 32 * 1024
                else (
                    2
                    if N <= 64 * 1024
                    else (4 if N <= 128 * 1024 else (8 if N <= 256 * 1024 else 16))
                )
            )
        self.cluster_n = cluster_n

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        assert mO.element_type == self.dtype
        self._set_cluster_n()
        tiler_mn, tv_layout = self._get_tv_layout()
        num_threads = cute.size(tv_layout, mode=[0])
        num_warps = num_threads // cute.arch.WARP_SIZE
        self.kernel(mX, mO, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if self.cluster_n > 1 else None,
            smem=self._smem_size_in_bytes(tiler_mn, num_warps),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, cluster_y, _ = cute.arch.block_idx()

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        # slice for CTAs
        gX, gO, cX = [
            cute.local_tile(mT, tiler_mn, (bidx, 0 if self.cluster_n == 1 else cluster_y))
            for mT in (mX, mO, idX)
        ]

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        # declare the atoms which will be used later for memory copy
        copy_atom_load_X = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), mX.element_type, num_bits_per_copy=128
        )
        copy_atom_store_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), gO.element_type, num_bits_per_copy=128
        )

        thr_copy_X = cute.make_tiled_copy(copy_atom_load_X, tv_layout, tiler_mn).get_slice(tidx)
        thr_copy_O = cute.make_tiled_copy(copy_atom_store_O, tv_layout, tiler_mn).get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgO = thr_copy_O.partition_D(gO)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        # allocate fragments for gmem->rmem
        tXrX, tXrO = [cute.make_fragment_like(thr) for thr in (tXgX, tXgO)]

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        is_even_N = cutlass.const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1]) if not is_even_N else None
        )
        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_load_X, tXgX, tXsX, pred=tXpX)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)
        # Fill OOB values with -inf
        if cutlass.const_expr(not is_even_N):
            tXrX_fp32 = cute.make_fragment_like(tXrX, cutlass.Float32)
            tXrX_fp32.store(x)
            for rest_v in range(tXpX.shape[0]):
                for rest_k in range(tXpX.shape[2]):
                    if not tXpX[rest_v, 0, rest_k]:
                        tXrX_fp32[(None, rest_v), None, rest_k].fill(-cutlass.Float32.inf)
            x = tXrX_fp32.load()
        threads_per_row = tv_layout.shape[0][0]
        max_x = utils.row_reduce(
            x,
            cute.ReductionOp.MAX,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr + 0 if self.cluster_n > 1 else None,
            init_val=-cutlass.Float32.inf,
            hook_fn=cute.arch.cluster_wait if cutlass.const_expr(self.cluster_n > 1) else None,
        )
        log2_e = math.log2(math.e)
        exp_x = cute.math.exp2((x - max_x) * log2_e, fastmath=True)
        denom = utils.row_reduce(
            exp_x,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 1],
            mbar_ptr + 1 if self.cluster_n > 1 else None,
            init_val=0.0,
        )
        y = exp_x * (1.0 / denom)
        tXrO.store(y.to(tXrO.element_type))
        tOpO = (
            utils.predicate_k(thr_copy_O.partition_S(cX), limit=shape[1]) if not is_even_N else None
        )
        if tXcX[0][0] < shape[0]:
            cute.copy(copy_atom_store_O, tXrO, tXgO, pred=tOpO)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Softmax forward pass.
    Args:
        x: Input tensor of shape (M, N)
    Returns:
        Softmax output tensor of same shape as x
    """
    assert x.dim() == 2, "Input must be 2D"
    assert x.is_cuda, "Tensor must be on CUDA device"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported dtype"
    M, N = x.shape
    out = torch.empty_like(x)
    dtype = torch2cute_dtype_map[x.dtype]
    convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
    )
    x_tensor, out_tensor = [convert_from_dlpack(tensor) for tensor in (x, out)]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compile_key = (dtype, N)
    if compile_key not in softmax.compile_cache:
        softmax_op = Softmax(dtype, N)
        softmax.compile_cache[compile_key] = cute.compile(
            softmax_op, x_tensor, out_tensor, current_stream
        )
    softmax.compile_cache[compile_key](x_tensor, out_tensor, current_stream)
    return out


softmax.compile_cache = {}
