import math
import torch
import operator
from typing import Callable, Union, Optional

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import quack.utils as utils


@cute.kernel
def cross_entropy_kernel(
    mX: cute.Tensor,  # (M, N)
    mTarget: cute.Tensor, # (M,)
    mLoss: cute.Tensor, # (M,)
    mLSE: Optional[cute.Tensor], # (M,)
    tv_layout: cute.Layout,
    tiler_mn: cute.Shape,
    cluster_n: cutlass.Constexpr = 1,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, cluster_y, _ = cute.arch.block_idx()

    shape: cute.Shape = mX.shape
    idX = cute.make_identity_tensor(shape)
    # slice for CTAs
    gX, cX = [
        cute.local_tile(mT, tiler_mn, (bidx, 0 if cluster_n == 1 else cluster_y))
        for mT in (mX, idX)
    ]

    smem = cutlass.utils.SmemAllocator()
    sX = smem.allocate_tensor(mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16)
    num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
    warps_per_row = utils.max_constexpr(tv_layout.shape[0][0] // cute.arch.WARP_SIZE, 1)
    reduction_buffer_layout = cute.make_ordered_layout(
        # 2 stages: 1 for max, 1 for sum
        (num_warps // warps_per_row, (warps_per_row, cluster_n), 2),
        order=(1, 0, 2)
    )
    reduction_buffer = smem.allocate_tensor(cutlass.Float32, reduction_buffer_layout, byte_alignment=4)
    if cutlass.const_expr(cluster_n > 1):
        # 1 mbar for max reduction, 1 mbar for sum reduction
        mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=2)
    else:
        mbar_ptr = None

    # declare the atoms which will be used later for memory copy
    copy_atom_load_X = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type, num_bits_per_copy=128)
    copy_atom_load_X_async = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), gX.element_type, num_bits_per_copy=128)

    thr_copy_X = cute.make_tiled_copy(copy_atom_load_X, tv_layout, tiler_mn).get_slice(tidx)
    thr_copy_X_async = cute.make_tiled_copy(copy_atom_load_X_async, tv_layout, tiler_mn).get_slice(tidx)

    #### Thread View
    tXgX = thr_copy_X_async.partition_S(gX)
    tXsX = thr_copy_X_async.partition_S(sX)
    tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

    # allocate fragments for gmem->rmem
    tXrX = cute.make_fragment_like(tXgX)  # only logits fragment needed

    if cluster_n > 1:
        if tidx < 2:
            cute.arch.mbarrier_init_arrive_cnt(mbar_ptr + tidx, 1)
        cute.arch.mbarrier_init_fence()
        if tidx < 2:
            cute.arch.mbarrier_init_tx_bytes(mbar_ptr + tidx, num_warps * cluster_n * cutlass.Float32.width // 8)
        # Cluster arrive after barrier init
        cute.arch.cluster_arrive_relaxed()

    row = tXcX[0][0]
    target = cute.Int32.zero
    if row < shape[0] and tXcX[0][1] == 0:
        target = cute.Int32(mTarget[row])

    is_even_N = cutlass.const_expr(shape[1] == tiler_mn[1] * cluster_n)
    tXpX = utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1]) if not is_even_N else None
    if row < shape[0]:
        cute.copy(copy_atom_load_X_async, tXgX, tXsX, pred=tXpX)
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

    target_logit = cute.Float32.zero
    if row < shape[0] and tXcX[0][1] == 0:
        target_logit = cute.Float32(mX[row, target])

    threads_per_row = tv_layout.shape[0][0]
    max_x = utils.row_reduce(
        x,
        cute.ReductionOp.MAX,
        threads_per_row,
        reduction_buffer[None, None, 0],
        mbar_ptr + 0 if cluster_n > 1 else None,
        init_val=-cutlass.Float32.inf,
        hook_fn=cute.arch.cluster_wait if cutlass.const_expr(cluster_n > 1) else None
    )
    log2_e = math.log2(math.e)
    # exp_x = cute.math.exp2((x - max_x) * log2_e, fastmath=True)
    exp_x = utils.exp2f((x - max_x) * log2_e)  # a bit faster, idk why
    denom = utils.row_reduce(
        exp_x,
        cute.ReductionOp.ADD,
        threads_per_row,
        reduction_buffer[None, None, 1],
        mbar_ptr + 1 if cluster_n > 1 else None,
        init_val=0.0,
    )

    if tXcX[0][1] == 0 and row < shape[0] and (cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0):
        ln_2 = math.log(2.0)
        lse = max_x + utils.log2f(denom) * ln_2
        loss_val = lse - target_logit
        mLoss[row] = loss_val.to(mLoss.element_type)
        if cutlass.const_expr(mLSE is not None):
            mLSE[row] = lse


@cute.jit
def cross_entropy_interface(
    mX: cute.Tensor,
    mTarget: cute.Tensor,
    mLoss: cute.Tensor,
    mLSE: Optional[cute.Tensor],
    stream: cuda.CUstream,
    N: cutlass.Constexpr,
    copy_bits: cutlass.Constexpr = 128
):
    vecsize = copy_bits // mX.element_type.width
    assert N % vecsize == 0, f"Input N {N} is not divisible by vector size {vecsize}"
    num_threads = 128 if N <= 16384 else 256

    num_warps = num_threads // cute.arch.WARP_SIZE
    assert num_threads % cute.arch.WARP_SIZE == 0
    threads_per_row = 8 if N <= 64 else (16 if N <= 128 else (32 if N <= 3072 else (64 if N <= 6144 else (128 if N <= 16384 else 256))))

    if cutlass.const_expr(mX.element_type.width == 16):
        cluster_n = 1 if N <= 16 * 1024 else (2 if N <= 32 * 1024 else (4 if N <= 64 * 1024 else (8 if N <= 128 * 1024 else 16)))
    else:  # fp32
        cluster_n = 1 if N <= 16 * 1024 else (2 if N <= 64 * 1024 else (4 if N <= 128 * 1024 else 8))

    num_blocks_N = cute.ceil_div(N // vecsize, threads_per_row * cluster_n)
    cols_per_block = num_threads // threads_per_row
    tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_row)  # This rounds up N
    tv_layout = cute.make_layout(
        ((threads_per_row, cols_per_block), (vecsize, num_blocks_N)),
        stride=((vecsize * cols_per_block, 1), (cols_per_block, cols_per_block * vecsize * threads_per_row))
    )

    smem_allocated = cute.size_in_bytes(mX.element_type, cute.make_layout(tiler_mn)) + 2 * num_warps * cluster_n * (cutlass.Float32.width // 8) + 2 * (cutlass.Int64.width // 8)
    cross_entropy_kernel(mX, mTarget, mLoss, mLSE, tv_layout, tiler_mn, cluster_n).launch(
        grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), cluster_n, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
        # Launching with cluster=[1, 1, 1] instead of None slows down the kernel by ~8us
        cluster=[1, cluster_n, 1] if cluster_n > 1 else None,
        smem=smem_allocated,
        stream=stream,
    )


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


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
    assert target.dtype == torch.int64, "Target must be int64"
    M, N = x.shape
    device = x.device
    loss = torch.empty(M, device=device, dtype=torch.float32)
    lse = torch.empty(M, device=device, dtype=torch.float32) if return_lse else None
    dtype = torch2cute_dtype_map[x.dtype]
    convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16)
        .mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
    )
    x_tensor, = [convert_from_dlpack(tensor) for tensor in (x,)]
    loss_tensor = from_dlpack(loss.detach(), assumed_align=4).mark_compact_shape_dynamic(mode=0)
    lse_tensor = from_dlpack(loss.detach(), assumed_align=4).mark_compact_shape_dynamic(mode=0) if lse is not None else None
    target_tensor = from_dlpack(target.detach(), assumed_align=8).mark_compact_shape_dynamic(mode=0)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compile_key = (dtype, N, lse_tensor is not None)
    if compile_key not in cross_entropy.compile_cache:
        cross_entropy.compile_cache[compile_key] = cute.compile(
            cross_entropy_interface, x_tensor, target_tensor, loss_tensor, lse_tensor, stream, N
        )
    cross_entropy.compile_cache[compile_key](x_tensor, target_tensor, loss_tensor, lse_tensor, stream)
    return loss if not return_lse else (loss, lse)


cross_entropy.compile_cache = {}
