import math
import torch
import operator
from typing import Callable

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.torch as cutlass_torch

import quack.utils as utils


@cute.kernel
def softmax_kernel(
    gX: cute.Tensor,
    gO: cute.Tensor,
    cX: cute.Tensor,  # coordinate tensor
    shape: cute.Shape,
    tv_layout: cute.Layout,
    tiler_mn: cute.Shape,
    cluster_n: cutlass.Constexpr = 1,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, cluster_y, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()

    # slice for CTAs
    # logical id -> address
    blkX, blkOut, blkCrd = [gT[(None, None), bidx if cluster_n == 1 else (bidx, cluster_y)] for gT in (gX, gO, cX)]

    # declare the atoms which will be used later for memory copy
    copy_atom_load_X = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type, num_bits_per_copy=128)
    copy_atom_load_X_async = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), gX.element_type, num_bits_per_copy=128)
    copy_atom_store_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gO.element_type, num_bits_per_copy=128)

    thr_copy_X = cute.make_tiled_copy(copy_atom_load_X, tv_layout, tiler_mn).get_slice(tidx)
    thr_copy_X_async = cute.make_tiled_copy(copy_atom_load_X_async, tv_layout, tiler_mn).get_slice(tidx)
    thr_copy_O = cute.make_tiled_copy(copy_atom_store_O, tv_layout, tiler_mn).get_slice(tidx)

    smem = cutlass.utils.SmemAllocator()
    # Don't use blkX.layout here, because the stride is N, not N_rounded
    sX = smem.allocate_tensor(gX.element_type, cute.make_ordered_layout(blkX.shape, order=(1, 0)), byte_alignment=16)
    num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
    warps_per_row = utils.max_constexpr(tv_layout.shape[0][0] // cute.arch.WARP_SIZE, 1)

    reduction_buffer_layout = cute.make_ordered_layout(
        # 2 stages: 1 for max, 1 for sum
        (num_warps // warps_per_row, warps_per_row if cluster_n == 1 else (warps_per_row, cluster_n), 2),
        order=(1, 0, 2)
    )
    reduction_buffer = smem.allocate_tensor(cutlass.Float32, reduction_buffer_layout, byte_alignment=4)
    if cutlass.const_expr(cluster_n > 1):
        # 1 mbar for max reduction, 1 mbar for sum reduction
        mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=2)
    else:
        mbar_ptr = None

    tXgX = thr_copy_X_async.partition_S(blkX)
    tXsX = thr_copy_X_async.partition_S(sX)
    tXgO = thr_copy_O.partition_D(blkOut)
    tXcX = thr_copy_X.partition_S(blkCrd)[(0, None), None, None]

    # allocate fragments for gmem->rmem
    tXrX, tXrO = [cute.make_fragment_like(thr) for thr in (tXgX, tXgO)]

    if cluster_n > 1:
        if tidx < 2:
            cute.arch.mbarrier_init_arrive_cnt(mbar_ptr + tidx, 1)
        cute.arch.mbarrier_init_fence()
        if tidx < 2:
            cute.arch.mbarrier_init_tx_bytes(mbar_ptr + tidx, num_warps * cluster_n * cutlass.Float32.width // 8)
        # Cluster arrive after barrier init
        cute.arch.cluster_arrive_relaxed()

    tXpX = cute.make_fragment_like(tXgX[(0, None), None, None], cutlass.Boolean)
    for i in range(cute.size(tXpX)):
        tXpX[i] = cute.elem_less(tXcX[i][1], shape[1])

    if tXcX[0][0] < shape[0]:
        cute.copy(copy_atom_load_X_async, tXgX, tXsX, pred=tXpX)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)

    cute.autovec_copy(tXsX, tXrX)
    x = tXrX.load().to(cute.Float32)
    max_x = utils.warp_reduce(
        x.reduce(cute.ReductionOp.MAX, init_val=float('-inf'), reduction_profile=0),
        cute.arch.fmax,
        width=utils.min_constexpr(tv_layout.shape[0][0], cute.arch.WARP_SIZE),
    )
    if cutlass.const_expr(cluster_n > 1):
        cute.arch.cluster_wait()
    if cutlass.const_expr(warps_per_row > 1 or cluster_n > 1):
        max_mbar_ptr = mbar_ptr + 0 if cluster_n > 1 else None
        max_x = utils.block_or_cluster_reduce(
            max_x, cute.arch.fmax, reduction_buffer[None, None, 0], max_mbar_ptr, init_val=-cutlass.Float32.inf
        )
    log2_e = math.log2(math.e)
    exp_x = cute.math.exp2((x - max_x) * log2_e, fastmath=True)
    denom = utils.warp_reduce(
        exp_x.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0),
        operator.add,
        width=utils.min_constexpr(tv_layout.shape[0][0], cute.arch.WARP_SIZE),
    )
    if cutlass.const_expr(warps_per_row > 1 or cluster_n > 1):
        sum_mbar_ptr = mbar_ptr + 1 if cluster_n > 1 else None
        denom = utils.block_or_cluster_reduce(
            denom, operator.add, reduction_buffer[None, None, 1], sum_mbar_ptr, init_val=0.0
        )
    inv = 1.0 / denom
    y = exp_x * inv

    tXrO.store(y.to(tXrO.element_type))
    tOcX = thr_copy_O.partition_S(blkCrd)[(0, None), None, None]
    tOpO = cute.make_fragment_like(tXgO[(0, None), None, None], cutlass.Boolean)
    for i in range(cute.size(tOpO)):
        tOpO[i] = cute.elem_less(tOcX[i][1], shape[1])
    if tXcX[0][0] < shape[0]:
        cute.copy(copy_atom_store_O, tXrO, tXgO, pred=tOpO)


@cute.jit
def softmax_interface(
    mX: cute.Tensor,
    mOut: cute.Tensor,
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
        cluster_n = 1 if N <= 32 * 1024 else (2 if N <= 64 * 1024 else (4 if N <= 128 * 1024 else (8 if N <= 256 * 1024 else 16)))

    num_blocks_N = cute.ceil_div(N // vecsize, threads_per_row * cluster_n)
    cols_per_block = num_threads // threads_per_row
    tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_row)  # This rounds up N
    tv_layout = cute.make_layout(
        ((threads_per_row, cols_per_block), (vecsize, num_blocks_N)),
        stride=((vecsize * cols_per_block, 1), (cols_per_block, cols_per_block * vecsize * threads_per_row))
    )

    idX = cute.make_identity_tensor(mX.shape)
    gX, gO, cX = [cute.zipped_divide(mT, tiler_mn) for mT in (mX, mOut, idX)]  # ((TileM,TileN),(RestM,RestN))

    smem_allocated = cute.size_in_bytes(mX.element_type, cute.make_layout(gX.shape[0])) + 2 * num_warps * cluster_n * (cutlass.Float32.width // 8) + 2 * (cutlass.Int64.width // 8)
    softmax_kernel(gX, gO, cX, mX.shape, tv_layout, tiler_mn, cluster_n).launch(
        grid=[cute.size(gX, mode=[1, 0]), cluster_n, 1],
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
    device = x.device
    out = torch.empty_like(x)
    dtype = torch2cute_dtype_map[x.dtype]
    convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16)
        .mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
    )
    x_tensor, out_tensor = [convert_from_dlpack(tensor) for tensor in (x, out)]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compile_key = (dtype, N)
    if compile_key not in softmax.compile_cache:
        softmax.compile_cache[compile_key] = cute.compile(
            softmax_interface, x_tensor, out_tensor, current_stream, N
        )
    softmax.compile_cache[compile_key](x_tensor, out_tensor, current_stream)
    return out


softmax.compile_cache = {}
