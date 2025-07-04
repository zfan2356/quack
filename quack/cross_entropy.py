import math
import torch
from typing import Optional, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import quack.utils as utils
from quack.reduction_base import ReductionBase, torch2cute_dtype_map


class CrossEntropy(ReductionBase):
    def __init__(self, dtype: Type[cutlass.Numeric], N: int, online_softmax: bool = True):
        # 2 stages: 1 for max, 1 for sum
        super().__init__(
            dtype,
            N,
            stage=2 if not online_softmax else 1,
            reduction_dtype=cutlass.Float32 if not online_softmax else cutlass.Int64,
        )
        self.online_softmax = online_softmax
        self.reload_from = None if N <= 16384 or online_softmax else "smem"

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
                if N <= 16 * 1024
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
        mTarget: cute.Tensor,
        mLoss: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        self._set_cluster_n()
        tiler_mn, tv_layout = self._get_tv_layout()
        num_threads = cute.size(tv_layout, mode=[0])
        num_warps = num_threads // cute.arch.WARP_SIZE
        self.kernel(mX, mTarget, mLoss, mLSE, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if cutlass.const_expr(self.cluster_n > 1) else None,
            smem=self._smem_size_in_bytes(tiler_mn, num_warps),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,  # (M, N)
        mTarget: cute.Tensor,  # (M,)
        mLoss: cute.Tensor,  # (M,)
        mLSE: Optional[cute.Tensor],  # (M,)
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        shape: cute.Shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        # slice for CTAs
        gX, cX = [cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) for mT in (mX, idX)]

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        # declare the atoms which will be used later for memory copy
        copy_atom_load_X = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), gX.element_type, num_bits_per_copy=128
        )
        thr_copy_X = cute.make_tiled_copy(copy_atom_load_X, tv_layout, tiler_mn).get_slice(tidx)

        #### Thread View
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]
        tXrX = cute.make_fragment_like(tXgX)

        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        row = tXcX[0][0]
        target = cute.Int32.zero
        if row < shape[0] and tXcX[0][1] == 0:
            target = cute.Int32(mTarget[row])

        is_even_N = cutlass.const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if cutlass.const_expr(not is_even_N)
            else None
        )
        if row < shape[0]:
            cute.copy(copy_atom_load_X, tXgX, tXsX, pred=tXpX)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        # Fill OOB values with -inf
        if cutlass.const_expr(not is_even_N):
            utils.fill_oob(tXsX, tXpX, -tXsX.element_type.inf)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)

        target_logit = cute.Float32.zero
        if row < shape[0] and tXcX[0][1] == 0:
            target_logit = cute.Float32(mX[row, target])

        threads_per_row = tv_layout.shape[0][0]
        if cutlass.const_expr(not self.online_softmax):
            max_x = utils.row_reduce(
                x,
                cute.ReductionOp.MAX,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr + 0 if cutlass.const_expr(self.cluster_n > 1) else None,
                init_val=-cutlass.Float32.inf,
                hook_fn=cute.arch.cluster_wait if cutlass.const_expr(self.cluster_n > 1) else None,
            )
            if cutlass.const_expr(self.reload_from == "smem"):
                cute.autovec_copy(tXsX, tXrX)
                x = tXrX.load().to(cute.Float32)
            log2_e = math.log2(math.e)
            # exp_x = cute.math.exp2((x - max_x) * log2_e, fastmath=True)
            # a bit faster, probably because it's calling ex2.approx.ftz instead of ex2.approx?
            # exp_x = utils.exp2f((x - max_x) * log2_e)
            # This would use ffma instead of fadd then fmul
            exp_x = utils.exp2f(x * log2_e - (max_x * log2_e))
            denom = utils.row_reduce(
                exp_x,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 1],
                mbar_ptr + 1 if cutlass.const_expr(self.cluster_n > 1) else None,
                init_val=0.0,
            )
        else:
            max_x, denom, _ = utils.online_softmax_reduce(
                x,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr,
                hook_fn=cute.arch.cluster_wait if cutlass.const_expr(self.cluster_n > 1) else None,
            )

        if (
            tXcX[0][1] == 0
            and row < shape[0]
            and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
        ):
            ln_2 = math.log(2.0)
            lse = max_x + utils.log2f(denom) * ln_2
            loss_val = lse - target_logit
            mLoss[row] = loss_val.to(mLoss.element_type)
            if cutlass.const_expr(mLSE is not None):
                mLSE[row] = lse


def cross_entropy(
    x: torch.Tensor,
    target: torch.Tensor,
    return_lse: bool = False,
) -> torch.Tensor:
    """Cross entropy forward pass.

    Args:
        x: Input logits tensor of shape (M, N)
        target: Target class indices tensor of shape (M,)

    Returns:
        Cross entropy loss tensor of shape (M,)
    """
    assert x.dim() == 2, "Input must be 2D"
    assert target.dim() == 1, "Target must be 1D"
    assert x.shape[0] == target.shape[0], "Batch dimensions must match"
    assert x.is_cuda and target.is_cuda, "Tensors must be on CUDA device"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported input dtype"
    assert target.dtype in [torch.int32, torch.int64], "Target must be int32 or int64"
    M, N = x.shape
    device = x.device
    loss = torch.empty(M, device=device, dtype=torch.float32)
    lse = torch.empty(M, device=device, dtype=torch.float32) if return_lse else None
    dtype = torch2cute_dtype_map[x.dtype]
    convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
    )
    x_tensor = convert_from_dlpack(x)
    loss_tensor = from_dlpack(loss.detach(), assumed_align=4).mark_compact_shape_dynamic(mode=0)
    lse_tensor = (
        from_dlpack(lse.detach(), assumed_align=4).mark_compact_shape_dynamic(mode=0)
        if lse is not None
        else None
    )
    target_tensor = from_dlpack(target.detach(), assumed_align=8).mark_compact_shape_dynamic(mode=0)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (dtype, N, lse is not None)
    if compile_key not in cross_entropy.compile_cache:
        cross_entropy_op = CrossEntropy(dtype, N)
        cross_entropy.compile_cache[compile_key] = cute.compile(
            cross_entropy_op, x_tensor, target_tensor, loss_tensor, lse_tensor, stream
        )
    cross_entropy.compile_cache[compile_key](
        x_tensor, target_tensor, loss_tensor, lse_tensor, stream
    )
    return loss if not return_lse else (loss, lse)


cross_entropy.compile_cache = {}
