import math
import argparse
import torch
import time
import operator
from typing import Type, Callable, Union

from triton.testing import do_bench

from einops import rearrange

try:
    import cudnn
except ImportError:
    cudnn = None

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.torch as cutlass_torch

from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import nvvm, llvm
from cutlass._mlir.dialects import cute as _cute_ir


def warp_reduce(val: cute.Numeric, op: Callable, width: cutlass.Constexpr = cute.arch.WARP_SIZE) -> cute.Numeric:
    for i in range(int(math.log2(width))):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@cute.jit
def block_reduce(val: cute.Numeric, op: Callable, reduction_buffer: cute.Tensor, init_val: cute.Numeric = 0.0) -> cute.Numeric:
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    warps_per_row = reduction_buffer.shape[1]
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier()
    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]
    return warp_reduce(block_reduce_val, op)


@dsl_user_op
def st_async(val: Union[float, cute.Float32], smem_ptr: cute.Pointer, mbar_ptr: cute.Pointer, peer_cta_rank_in_cluster: cute.typing.Int, *, loc=None, ip=None) -> None:
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    mbar_ptr_i32 = mbar_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [smem_ptr_i32, mbar_ptr_i32, peer_cta_rank_in_cluster.ir_value(), cute.Float32(val).ir_value(loc=loc, ip=ip)],
        ".reg .b32 smem_ptr, mbar_ptr;\n"
        "mapa.shared::cluster.u32 smem_ptr, $0, $2;\n"
        "mapa.shared::cluster.u32 mbar_ptr, $1, $2;\n"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [smem_ptr], $3, [mbar_ptr];",
        "r,r,r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def mapa_i32(smem_ptr: cute.Pointer, mbar_ptr: cute.Pointer, peer_cta_rank_in_cluster: cute.Int32, *, loc=None, ip=None) -> (cutlass.Int32, cutlass.Int32):
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    mbar_ptr_i32 = mbar_ptr.toint(loc=loc, ip=ip).ir_value()
    res0 = cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    res1 = cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [mbar_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    return res0, res1


@cute.jit
def cluster_reduce(val: cute.Numeric, op: Callable, reduction_buffer: cute.Tensor, mbar_ptr: cute.Pointer, init_val: cute.Numeric = 0.0) -> cute.Numeric:
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    warps_per_row, cluster_n = reduction_buffer.shape[1]
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    # smem_ptr = reduction_buffer[row_idx, None].iterator + col_idx * warps_per_row + cta_rank_in_cluster
    # peer_cta_rank_in_cluster = lane_idx
    # smem_llvm_ptr = _cute_ir.inttoptr(cutlass.Int64(mapa_shared_cluster(smem_ptr, peer_cta_rank_in_cluster)))
    # smem_mapa, mbar_mapa = mapa_i32(smem_ptr, mbar_ptr, peer_cta_rank_in_cluster)
    # if lane_idx < cluster_n and warp_idx == 0:
    #     # cute.printf("st_async: lane_idx = {}, cta_rank = {}, smem_ptr = {}, mbar_ptr = {}", lane_idx, cta_rank_in_cluster, smem_ptr, mbar_ptr)
    #     cute.printf("st_async: lane_idx = {}, cta_rank = {}, smem_ptr = {}, smem_mapa = {}, mbar_ptr = {}, mbar_mapa = {}", lane_idx, cta_rank_in_cluster, smem_ptr, smem_mapa, mbar_ptr, mbar_mapa)
    if lane_idx < cluster_n:
        # What's the right way to get the address an element in reduction_buffer?
        st_async(val, reduction_buffer[row_idx, (col_idx, None)].iterator + cta_rank_in_cluster * reduction_buffer.stride[1][1], mbar_ptr, lane_idx)
    cute.arch.mbarrier_wait(mbar_ptr, phase=0)
    block_reduce_val = init_val
    num_iter = cute.ceil_div(warps_per_row * cluster_n, cute.arch.WARP_SIZE)
    for i in cutlass.range_constexpr(num_iter):
        idx = lane_idx + i * cute.arch.WARP_SIZE
        if idx < cute.size(reduction_buffer, mode=[1]):
            block_reduce_val = op(block_reduce_val, reduction_buffer[row_idx, idx])
    return warp_reduce(block_reduce_val, op)


@cute.jit
def min_constexpr(a: cutlass.Constexpr, b: cutlass.Constexpr) -> cutlass.Constexpr:
    return a if a < b else b

@cute.jit
def max_constexpr(a: cutlass.Constexpr, b: cutlass.Constexpr) -> cutlass.Constexpr:
    return a if a > b else b


@dsl_user_op
def rsqrt(a: Union[float, cute.Float32], *, loc=None, ip=None) -> cute.Float32:
    return cute.Float32(
        llvm.inline_asm(
            T.f32(),
            [cute.Float32(a).ir_value(loc=loc, ip=ip)],
            "rsqrt.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.kernel
def rmsnorm_kernel(
    gX: cute.Tensor,
    gW: cute.Tensor,
    gO: cute.Tensor,
    gRstd: cute.Tensor,
    cX: cute.Tensor,  # coordinate tensor
    eps: cute.Float32,
    shape: cute.Shape,
    tv_layout: cute.Layout,
    tiler_mn: cute.Shape,
    cluster_n: cutlass.Constexpr = 1,
    reload_from: cutlass.Constexpr = None,
    delay_w_load: cutlass.Constexpr = False,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, cluster_y, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()

    # slice for CTAs
    # logical id -> address
    blkX, blkOut, blkRstd, blkCrd = [gT[(None, None), bidx if cluster_n == 1 else (bidx, cluster_y)] for gT in (gX, gO, gRstd, cX)]
    blkW = gW[(None, None), 0 if cluster_n == 1 else (0, cluster_y)]


    print(f"[DSL INFO] Sliced Tensors per thread block:")
    print(f"[DSL INFO]   blkX = {blkX.type}")
    print(f"[DSL INFO]   blkW = {blkW.type}")
    print(f"[DSL INFO]   blkOut = {blkOut.type}")
    print(f"[DSL INFO]   blkRstd = {blkRstd.type}")
    print(f"[DSL INFO]   blkCrd = {blkCrd.type}")

    # declare the atoms which will be used later for memory copy
    copy_atom_load_X = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type, num_bits_per_copy=128)
    copy_atom_load_X_async = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), gX.element_type, num_bits_per_copy=128)
    copy_atom_load_W = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gW.element_type, num_bits_per_copy=128)
    copy_atom_store_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gO.element_type, num_bits_per_copy=128)

    thr_copy_X = cute.make_tiled_copy(copy_atom_load_X, tv_layout, tiler_mn).get_slice(tidx)
    thr_copy_X_async = cute.make_tiled_copy(copy_atom_load_X_async, tv_layout, tiler_mn).get_slice(tidx)
    thr_copy_W = cute.make_tiled_copy(copy_atom_load_W, tv_layout, tiler_mn).get_slice(tidx)
    thr_copy_O = cute.make_tiled_copy(copy_atom_store_O, tv_layout, tiler_mn).get_slice(tidx)

    smem = cutlass.utils.SmemAllocator()
    # Don't use blkX.layout here, because the stride is N, not N_rounded
    sX = smem.allocate_tensor(gX.element_type, cute.make_ordered_layout(blkX.shape, order=(1, 0)), byte_alignment=16)
    num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
    warps_per_row = max_constexpr(tv_layout.shape[0][0] // cute.arch.WARP_SIZE, 1)
    # reduction_buffer_layout = cute.make_ordered_layout((num_warps // warps_per_row, warps_per_row), order=(1, 0))
    reduction_buffer_layout = cute.make_ordered_layout((num_warps // warps_per_row, warps_per_row if cluster_n == 1 else (warps_per_row, cluster_n)), order=(1, 0))
    reduction_buffer = smem.allocate_tensor(cutlass.Float32, reduction_buffer_layout, byte_alignment=4)
    print(f"[DSL INFO] reduction_buffer = {reduction_buffer.type}")
    mbar_ptr = cute.Pointer()
    if cluster_n > 1:
        mbar_ptr = smem.allocate(cutlass.Int64.width // 8, byte_alignment=8)

    tWgW = thr_copy_W.partition_S(blkW)
    # tXgX = thr_copy_X.partition_S(blkX)
    tXgX = thr_copy_X_async.partition_S(blkX)
    tXsX = thr_copy_X_async.partition_S(sX)
    tXgO, tXrRstd = [thr_copy_O.partition_D(blk) for blk in (blkOut, blkRstd)]
    tXcX = thr_copy_X.partition_S(blkCrd)[(0, None), None, None]

    # allocate fragments for gmem->rmem
    tWrW = cute.make_fragment_like(tWgW)
    tXrW = thr_copy_X.retile(tWrW)
    tXrX, tXrO = [cute.make_fragment_like(thr) for thr in (tXgX, tXgO)]

    print(f"[DSL INFO] Sliced Tensors per thread:")
    print(f"[DSL INFO]   tXgX = {tXgX.type}")
    print(f"[DSL INFO]   tXsX = {tXsX.type}")
    print(f"[DSL INFO]   tWgW = {tWgW.type}")
    print(f"[DSL INFO]   tWrW = {tWrW.type}")
    print(f"[DSL INFO]   tXgO = {tXgO.type}")
    print(f"[DSL INFO]   tXrRstd = {tXrRstd.type}")
    # print(f"[DSL INFO]   tXrRstd filter = {cute.filter_zeros(tXrRstd).type}")
    print(f"[DSL INFO]   tXcX = {tXcX.type}")
    # print(f"[DSL INFO]   thr_copy_X = {thr_copy_X.type}")

    # # Print per thread predicate mask
    # if tidx == 0 and bidx == 0:
    #     cute.printf("block_dim = {}", cute.arch.grid_dim())
    #     cute.printf("shape = {}", shape)
    #     cute.print_tensor(tXgX)
    #     cute.print_tensor(tWgW)
    #     cute.print_tensor(tXpX)


    if cluster_n > 1:
        if tidx == 0:
            cute.arch.mbarrier_init_arrive_cnt(mbar_ptr, 1)
        cute.arch.mbarrier_init_fence()
        if tidx == 0:
            cute.arch.mbarrier_init_tx_bytes(mbar_ptr, num_warps * cluster_n * cutlass.Float32.width // 8)
        # Cluster arrive after barrier init
        cute.arch.cluster_arrive_relaxed()


    tXpX = cute.make_fragment_like(tXgX[(0, None), None, None], cutlass.Boolean)
    for i in range(cute.size(tXpX)):
        tXpX[i] = cute.elem_less(tXcX[i][1], shape[1])
    # tXrX.fill(0.0)
    if tXcX[0][0] < shape[0]:
        # cute.copy(copy_atom_load_X, tXgX, tXrX, pred=tXpX)
        cute.copy(copy_atom_load_X_async, tXgX, tXsX, pred=tXpX)
    cute.arch.cp_async_commit_group()

    tWpW = cute.make_fragment_like(tWgW[(0, None), None, None], cutlass.Boolean)
    tWcX = thr_copy_W.partition_S(blkCrd)[(0, None), None, None]
    for i in range(cute.size(tWpW)):
        tWpW[i] = cute.elem_less(tWcX[i][1], shape[1])
    if not delay_w_load:
        cute.copy(copy_atom_load_W, tWgW, tWrW, pred=tWpW)

    print(f"[DSL INFO]   tXpX = {tXpX.type}")
    print(f"[DSL INFO]   tWpP = {tWpW.type}")

    cute.arch.cp_async_wait_group(0)
    cute.autovec_copy(tXsX, tXrX)
    x = tXrX.load().to(cute.Float32)
    # if tidx == 1 and bidx == 0:
    #     cute.print_tensor(tXpX)
    #     for i in range(cute.size(tXcX)):
    #         cute.printf("tXcX[{}] = {}, {}", i, tXcX[i][0], tXcX[i][1])
    #     cute.printf(shape)
    #     for i in range(cute.size(tXrX)):
    #         xi = x[i].to(cute.Float32)
    #         cute.printf("x[{}] = {}", i, xi)

    # thr_sum_sq_A = cutlass.Float32.zero
    # for i in range(cute.size(tXrX)):
    #     xi = x[i].to(cute.Float32)
    #     thr_sum_sq_A = thr_sum_sq_A + xi * xi
    #     # thr_sum_sq_A = thr_sum_sq_A + (x[i] * x[i] if pred[i] else 0.0)
    # warp_sum_sq_x = warp_reduce(thr_sum_sq_A, operator.add)
    warp_sum_sq_x = warp_reduce(
        (x * x).reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0),
        operator.add,
        width=min_constexpr(tv_layout.shape[0][0], cute.arch.WARP_SIZE),
    )
    if cluster_n > 1:
        cute.arch.cluster_wait()
    sum_sq_x = warp_sum_sq_x
    if cutlass.const_expr(warps_per_row * cluster_n) > 1:
        if cutlass.const_expr(cluster_n) == 1:
            sum_sq_x = block_reduce(sum_sq_x, operator.add, reduction_buffer, init_val=0.0)
        else:
            sum_sq_x = cluster_reduce(sum_sq_x, operator.add, reduction_buffer, mbar_ptr, init_val=0.0)
            # if tidx == 0:
            #     cute.printf("sum_sq_x = {}", sum_sq_x)
    # rstd = cute.Float32(1.0) / cutlass._mlir.dialects.math.sqrt(var + eps)
    rstd = rsqrt(sum_sq_x / shape[1] + eps)
    # Only the thread corresponding to column 0 writes out the rstd to gmem
    if tXcX[0][1] == 0 and tXcX[0][0] < shape[0]:
        if cluster_n == 1:
            tXrRstd[0] = rstd
        else:
            if cute.arch.block_idx_in_cluster() == 0:
                tXrRstd[0] = rstd
    if delay_w_load:
        cute.copy(copy_atom_load_W, tWgW, tWrW, pred=tWpW)
    if reload_from == "smem":
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)
    elif reload_from == "gmem":
        cute.copy(copy_atom_load_X, tXgX, tXrX, pred=tXpX)
        x = tXrX.load().to(cute.Float32)
    x_hat = x * rstd
    w = tXrW.load().to(cute.Float32)
    y = x_hat * w
    tXrO.store(y.to(tXrO.element_type))
    tOcX = thr_copy_O.partition_S(blkCrd)[(0, None), None, None]
    tOpO = cute.make_fragment_like(tXgO[(0, None), None, None], cutlass.Boolean)
    for i in range(cute.size(tOpO)):
        tOpO[i] = cute.elem_less(tOcX[i][1], shape[1])
    if tXcX[0][0] < shape[0]:
        cute.copy(copy_atom_store_O, tXrO, tXgO, pred=tOpO)

    # # Wait for all thread blocks in the Cluster
    # if cluster_n > 1:
    #     # cute.arch.cluster_arrive()
    #     cute.arch.cluster_wait()

    # row_idx = tXcX[0][0]
    # jump = gdim
    # for row_offset in cutlass.range_dynamic(bidx, cute.size(gX, mode=[1]), jump):
    #     cute.arch.cp_async_wait_group(0)
    #     cute.autovec_copy(tXsX, tXrX)
    #     x = tXrX.load().to(cute.Float32)

    #     # thr_sum_sq_A = cutlass.Float32.zero
    #     # for i in range(cute.size(tXrX)):
    #     #     xi = x[i].to(cute.Float32)
    #     #     thr_sum_sq_A = thr_sum_sq_A + xi * xi
    #     #     # thr_sum_sq_A = thr_sum_sq_A + (x[i] * x[i] if pred[i] else 0.0)
    #     # warp_sum_sq_x = warp_reduce(thr_sum_sq_A, operator.add)
    #     warp_sum_sq_x = warp_reduce(
    #         (x * x).reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0),
    #         operator.add,
    #         width=min_constexpr(tv_layout.shape[0][0], cute.arch.WARP_SIZE),
    #     )
    #     sum_sq_x = warp_sum_sq_x
    #     if cutlass.const_expr(warps_per_row) > 1:
    #         cute.arch.barrier()
    #         sum_sq_x = block_reduce(sum_sq_x, operator.add, reduction_buffer, init_val=0.0)
    #     var = sum_sq_x / shape[1]
    #     # rstd = cute.Float32(1.0) / cutlass._mlir.dialects.math.sqrt(var + eps)
    #     rstd = rsqrt(var + eps)
    #     tXgO, tXrRstd = [thr_copy_O.partition_D(gT[(None, None), row_offset]) for gT in (gO, gRstd)]
    #     # Only the thread corresponding to column 0 writes out the rstd to gmem
    #     if cute.get(tXcX[0], mode=[1]) == 0 and row_idx < shape[0]:
    #         tXrRstd[0] = rstd
    #     cute.autovec_copy(tXsX, tXrX)
    #     x = tXrX.load().to(cute.Float32)
    #     tXgX = thr_copy_X_async.partition_S(gX[(None, None), row_offset + jump])
    #     if row_idx + jump < shape[0]:
    #         cute.copy(copy_atom_load_X_async, tXgX, tXsX, pred=tXpX)
    #     cute.arch.cp_async_commit_group()
    #     x_hat = x * rstd
    #     w = tXrW.load().to(cute.Float32)
    #     y = x_hat * w
    #     tXrO.store(y.to(tXrO.element_type))
    #     tOcX = thr_copy_O.partition_S(blkCrd)
    #     tOpO = cute.make_fragment_like(tXgO[(0, None), None, None], cutlass.Boolean)
    #     for i in range(cute.size(tOpO)):
    #         tOpO[i] = cute.elem_less(tOcX[i][1], shape[1])
    #     if row_idx < shape[0]:
    #         cute.copy(copy_atom_store_O, tXrO, tXgO, pred=tOpO)
    #     row_idx += jump
    #     # cute.copy(copy_atom_store_O, tXrO, tXgO)

    #     # if tidx == 0 and bidx == 0:
    #     #     # cute.printf("tXrX[0] = {}", tXrX[0])
    #     #     # cute.printf(f"sum_A type = {type(warp_sum_sq_x)}")
    #     #     cute.printf("sum_A = {}", warp_sum_sq_x)
    #     #     cute.printf("var_A = {}", var)
    #     #     cute.printf("rstd_A = {}", rstd)


@cute.jit
def rmsnorm(
    # mX_: cute.Tensor,
    mX: cute.Tensor,
    mW: cute.Tensor,
    mOut: cute.Tensor,
    mRstd: cute.Tensor,
    stream: cuda.CUstream,
    N: cutlass.Constexpr,
    eps: cutlass.Float32 = 1e-6,
    copy_bits: cutlass.Constexpr = 128
):
    N = mX.shape[1]
    # new_shape = (mX_.shape[0], cute.assume(mX_.shape[1], 128))
    # breakpoint()
    # mX = cute.make_tensor(mX_.iterator, cute.make_layout(new_shape, stride=mX_.stride))
    vecsize = copy_bits // mX.element_type.width
    assert N % vecsize == 0, f"Input N {N} is not divisible by vector size {vecsize}"
    num_threads = 128 if N <= 16384 else 256
    num_warps = num_threads // cute.arch.WARP_SIZE
    assert num_threads % cute.arch.WARP_SIZE == 0
    threads_per_row = 8 if N <= 64 else (16 if N <= 128 else (32 if N <= 3072 else (64 if N <= 6144 else (128 if N <= 16384 else 256))))
    # cluster_n = 4 is faster and cluster_n = 2 for N=64k for some reason
    # Similarly cluster_n = 8 is faster for N=128k
    cluster_n = 1 if N <= 32 * 1024 else (4 if N <= 64 * 1024 else (8 if N <= 128 * 1024 else 16))
    num_blocks_N = cute.ceil_div(N // vecsize, threads_per_row * cluster_n)

    cols_per_block = num_threads // threads_per_row
    tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_row)  # This rounds up N
    tv_layout = cute.make_layout(
        ((threads_per_row, cols_per_block), (vecsize, num_blocks_N)),
        stride=((vecsize * cols_per_block, 1), (cols_per_block, cols_per_block * vecsize * threads_per_row))
    )

    print(f"[DSL INFO] Input Tensors:")
    print(f"[DSL INFO]   mX = {mX.type}")
    print(f"[DSL INFO]   mW = {mW.type}")
    print(f"[DSL INFO]   mOut = {mOut.type}")
    print(f"[DSL INFO]   mRstd = {mRstd.type}")

    print(f"[DSL INFO] Tiling Parameters:")
    print(f"[DSL INFO]   tiler_mn = {tiler_mn} per thread block")
    print(f"[DSL INFO]   tv_layout = {tv_layout}")

    mW_expanded_layout = cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
    mW_expanded = cute.make_tensor(mW.iterator, mW_expanded_layout)
    mRstd_expanded_layout = cute.append(mRstd.layout, cute.make_layout((N,), stride=(0,)))
    mRstd_expanded = cute.make_tensor(mRstd.iterator, mRstd_expanded_layout)
    idX = cute.make_identity_tensor(mX.shape)
    gX, gW, gO, gRstd, cX = [cute.zipped_divide(mT, tiler_mn) for mT in (mX, mW_expanded, mOut, mRstd_expanded, idX)]  # ((TileM,TileN),(RestM,RestN))
    print(f"[DSL INFO] Tiled Tensors:")
    print(f"[DSL INFO]   gX = {gX.type}")
    print(f"[DSL INFO]   gW = {gW.type}")
    print(f"[DSL INFO]   gO = {gO.type}")
    print(f"[DSL INFO]   gRstd = {gRstd.type}")
    print(f"[DSL INFO]   coord tensor = {cX.type}")

    # reload_from = None if N <= 16384 else ("smem" if N <= 32768 else "gmem")
    reload_from = None if N <= 16384 else "smem"
    # delay_w_load = N > 64 * 1024
    delay_w_load = False
    rmsnorm_kernel(gX, gW, gO, gRstd, cX, eps, mX.shape, tv_layout, tiler_mn, cluster_n, reload_from).launch(
        grid=[cute.size(gX, mode=[1, 0]), cluster_n, 1],
        # grid=[132 * 8, 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
        # Launching with cluster=[1, 1, 1] instead of None slows down the kernel by ~8us
        cluster=[1, cluster_n, 1] if cluster_n > 1 else None,
        smem=cute.size_in_bytes(mX.element_type, gX.layout[0]) + num_warps * cluster_n * (cutlass.Float32.width // 8) + (cutlass.Int64.width // 8),
        stream=stream,
    )


def rmsnorm_ref(x, w, eps=1e-6):
    x_f32 = x.float()
    return (x_f32 / (torch.sqrt(torch.mean(x_f32 * x_f32, dim=-1, keepdim=True) + eps)) * w).to(x.dtype)


def rstd_ref(x, eps=1e-6):
    x_f32 = x.float()
    return 1.0 / torch.sqrt(torch.mean(x_f32 * x_f32, dim=-1) + eps)


def rmsnorm_cudnn_setup(M, N, dtype):
    x_gpu = torch.empty(M, N, dtype=dtype, device="cuda")
    scale_gpu = torch.empty(1, N, dtype=dtype, device="cuda")
    epsilon_cpu = torch.ones((1, 1), dtype=torch.float32, device="cpu")
    out_gpu = torch.empty_like(x_gpu)
    inv_var_gpu = torch.empty(M, 1, dtype=torch.float32, device="cuda")
    handle = cudnn.create_handle()
    graph = cudnn.pygraph(
        handle=handle,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    # create tensor handles with the graph API
    x = graph.tensor_like(x_gpu.detach()).set_name("X")
    scale = graph.tensor_like(scale_gpu.detach()).set_name("scale")
    epsilon = graph.tensor_like(epsilon_cpu).set_name("epsilon")
    (out, inv_var) = graph.rmsnorm(
        name="rmsnorm",
        input=x,
        norm_forward_phase=cudnn.norm_forward_phase.TRAINING,
        scale=scale,
        epsilon=epsilon,
    )
    # enable all outputs
    out.set_name("output").set_output(True).set_data_type(out_gpu.dtype)
    inv_var.set_name("inv_var").set_output(True).set_data_type(inv_var_gpu.dtype)
    graph.build([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    # Mapping of (handles -> memory)
    variant_pack = {
        x: x_gpu.detach(),
        scale: scale_gpu.detach(),
        epsilon: epsilon_cpu,
        out: out_gpu,
        inv_var: inv_var_gpu,
    }
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def run(*args, **kwargs):
        graph.execute(variant_pack, workspace)
        return out_gpu, inv_var_gpu

    return run


def rmsnorm_bwd_ref(x, w, dout, rstd, eps=1e-6):
    x_f32 = x.float()
    x_hat = x_f32 * rstd.unsqueeze(1)
    wdy = dout * w
    c1 = (x_hat * wdy).mean(dim=-1, keepdim=True)
    dx = (wdy - x_hat * c1) * rstd.unsqueeze(1)
    return dx.to(x.dtype)


def run_rmsnorm(
    M,
    N,
    dtype: Type[cutlass.Numeric],
    skip_ref_check=False,
    benchmark=True,
    warmup_iterations=2,
    iterations=200,
):
    if not torch.cuda.is_available():
        raise RuntimeError(f"Ampere GPU is required to run this example!")

    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"Input and Output Data type: {dtype}")

    torch_dtype = cutlass_torch.dtype(dtype)

    device = "cuda"
    x = torch.randn(M, N, device=device, dtype=torch_dtype)
    w = torch.randn(N, device=device, dtype=torch.float32)
    out = torch.empty_like(x)
    rstd = torch.empty(M, device=device, dtype=torch.float32)

    print(f"Input tensor shapes:")
    print(f"x: {x.shape}, dtype: {x.dtype}")
    print(f"w: {w.shape}, dtype: {w.dtype}")
    print(f"out: {out.shape}, dtype: {out.dtype}")
    print(f"rstd: {rstd.shape}, dtype: {rstd.dtype}\n")

    convert_from_dlpack = lambda x: (
        from_dlpack(x, assumed_align=16)
        # .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
    )
    x_tensor, out_tensor = [convert_from_dlpack(tensor) for tensor in (x, out)]
    # x_tensor_dynamic = x_tensor.mark_layout_dynamic(leading_dim=1)
    # print(x_tensor)
    # print(x_tensor_dynamic)
    # breakpoint()
    # x_tensor = cute.make_tensor(x_tensor, cute.make_layout((x.shape[0], x.shape[1]), stride=(0, 1)))
    w_tensor = from_dlpack(w, assumed_align=16)
    rstd_tensor = from_dlpack(rstd, assumed_align=4).mark_compact_shape_dynamic(mode=0)

    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    print("Compiling kernel with cute.compile ...")
    compiled_func = cute.compile(rmsnorm, x_tensor, w_tensor, out_tensor, rstd_tensor, stream, x.shape[1])
    print("Executing kernel...")
    eps = 1e-6
    compiled_func(x_tensor, w_tensor, out_tensor, rstd_tensor, stream, eps)

    compiled_func_ref = torch.compile(rmsnorm_ref)
    if not skip_ref_check:
        # compiled_func(x_tensor, w_tensor, out_tensor, rstd_tensor, stream, eps)
        print("Verifying results...")
        out_ref = compiled_func_ref(x, w, eps=eps)
        torch.testing.assert_close(out_ref, out)
        torch.testing.assert_close(rstd_ref(x, eps=eps), rstd)
        print("Results verified successfully!")

    if benchmark:
        fn = lambda: compiled_func(x_tensor, w_tensor, out_tensor, rstd_tensor, stream, eps)
        time.sleep(0.5)
        avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
        mem_bw = (2 * x.numel() * dtype.width // 8) / (avg_time / 1000) / 1e9
        print(f"Kernel execution time: {avg_time:.4f} ms")
        print(f"Mem throughput: {mem_bw:.2f} GB/s")

        fn = lambda: compiled_func_ref(x, w, eps=eps)
        for _ in range(5): fn()  # warm up
        time.sleep(0.5)
        avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
        mem_bw_ref = (2 * x.numel() * dtype.width // 8) / (avg_time / 1000) / 1e9
        print(f"Ref kernel execution time: {avg_time:.4f} ms")
        print(f"Ref mem throughput: {mem_bw_ref:.2f} GB/s")

        if cudnn is not None:
            # x_expanded = rearrange(x, "m n -> m n 1 1")
            # w_expanded = rearrange(w, "n -> 1 n 1 1")
            run_cudnn = rmsnorm_cudnn_setup(M, N, torch_dtype)
            # out_cudnn, inv_var_cudnn = run_cudnn(x_expanded, w_expanded, eps=eps)
            # torch.testing.assert_close(out_cudnn, out)
            # torch.testing.assert_close(inv_var_cudnn, rstd)
            # print("cuDNN kernel executed successfully!")
            time.sleep(0.5)
            avg_time = do_bench(run_cudnn, warmup=warmup_iterations, rep=iterations)
            mem_bw_cudnn = (2 * x.numel() * dtype.width // 8) / (avg_time / 1000) / 1e9
            print(f"Cudnn kernel execution time: {avg_time:.4f} ms")
            print(f"Cudnn mem throughput: {mem_bw_cudnn:.2f} GB/s")

        # from flash_attn.ops.triton.layer_norm import rms_norm_fn
        # from flash_attn.utils.benchmark import pytorch_profiler
        # fn = lambda: rms_norm_fn(x, w, bias=None)
        # avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
        # print(f"Triton kernel execution time: {avg_time:.4f} ms")
        # print(f"Triton mem throughput: {(2 * x.numel() * dtype.width // 8) / (avg_time / 1000) / 1e9:.2f} GB/s")
        # pytorch_profiler(rms_norm_fn, x, w, bias=None)
        return mem_bw, mem_bw_ref


@cute.kernel
def rmsnorm_bwd_kernel(
    gX: cute.Tensor,
    gW: cute.Tensor,
    gDout: cute.Tensor,
    gRstd: cute.Tensor,
    gDx: cute.Tensor,
    cX: cute.Tensor,  # coordinate tensor
    eps: cute.Float32,
    shape: cute.Shape,
    tv_layout: cute.Layout,
    tiler_mn: cute.Shape,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # slice for CTAs
    # logical id -> address
    blkX, blkDout, blkRstd, blkDx, blkCrd = [gT[(None, None), bidx] for gT in (gX, gDout, gRstd, gDx, cX)]
    blkW = gW[(None, None), 0]

    # print(f"[DSL INFO] Sliced Tensors per thread block:")
    # print(f"[DSL INFO]   blkX = {blkX.type}")
    # print(f"[DSL INFO]   blkW = {blkW.type}")
    # print(f"[DSL INFO]   blkDout = {blkDout.type}")
    # print(f"[DSL INFO]   blkRstd = {blkRstd.type}")
    # print(f"[DSL INFO]   blkDx = {blkDx.type}")
    # print(f"[DSL INFO]   blkCrd = {blkCrd.type}")

    # declare the atoms which will be used later for memory copy
    copy_atom_load_X = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type)
    copy_atom_load_W = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gW.element_type)
    copy_atom_load_Dout = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gDout.element_type)
    copy_atom_store_Dx = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gDx.element_type)

    thr_copy_X = cute.make_tiled_copy(copy_atom_load_X, tv_layout, tiler_mn).get_slice(tidx)
    thr_copy_W = cute.make_tiled_copy(copy_atom_load_W, tv_layout, tiler_mn).get_slice(tidx)
    thr_copy_Dout = cute.make_tiled_copy(copy_atom_load_Dout, tv_layout, tiler_mn).get_slice(tidx)
    thr_copy_Dx = cute.make_tiled_copy(copy_atom_store_Dx, tv_layout, tiler_mn).get_slice(tidx)

    tWgW = thr_copy_W.partition_S(blkW)
    tXgX = thr_copy_X.partition_S(blkX)
    thrDout, tXrRstd = [thr_copy_Dout.partition_S(blk) for blk in (blkDout, blkRstd)]
    thrDx = thr_copy_Dx.partition_D(blkDx)
    tXcX = thr_copy_W.partition_S(blkCrd)

    smem = cutlass.utils.SmemAllocator()
    num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
    warps_per_row = max_constexpr(tv_layout.shape[0][0] // cute.arch.WARP_SIZE, 1)
    reduction_buffer_layout = cute.make_ordered_layout((num_warps // warps_per_row, warps_per_row), order=(1, 0))
    reduction_buffer = smem.allocate_tensor(cutlass.Float32, reduction_buffer_layout, byte_alignment=4)
    # print(f"[DSL INFO] reduction_buffer = {reduction_buffer.type}")


    # allocate fragments for gmem->rmem
    tXrW = cute.make_fragment_like(tWgW)
    tXrX, frgDout, frgDx = [cute.make_fragment_like(thr) for thr in (tXgX, thrDout, thrDx)]
    tXpX = cute.make_fragment(tXrW.shape, cutlass.Boolean)

    print(f"[DSL INFO] Sliced Tensors per thread:")
    print(f"[DSL INFO]   tXgX = {tXgX.type}")
    print(f"[DSL INFO]   tWgW = {tWgW.type}")
    print(f"[DSL INFO]   thrDout = {thrDout.type}")
    print(f"[DSL INFO]   tXrRstd = {tXrRstd.type}")
    print(f"[DSL INFO]   tXcX = {tXcX.type}")

    # Print per thread predicate mask
    # if tidx == 0 and bidx == 0:
    #     cute.printf("block_dim = {}", cute.arch.grid_dim())
    #     cute.printf("shape = {}", shape)
    #     cute.print_tensor(tXgX)
    #     cute.print_tensor(tWgW)
    #     cute.print_tensor(tXpX)

    for i in range(cute.size(tXpX)):
        tXpX[i] = cute.elem_less(tXcX[i], shape)
    cute.copy(copy_atom_load_W, tWgW, tXrW, pred=tXpX)
    tXrX.fill(0.0)
    cute.copy(copy_atom_load_X, tXgX, tXrX, pred=tXpX)
    frgDout.fill(0.0)
    cute.copy(copy_atom_load_Dout, thrDout, frgDout, pred=tXpX)

    x = tXrX.load().to(cute.Float32)
    w = tXrW.load().to(cute.Float32)
    dout = frgDout.load().to(cute.Float32)
    rstd = tXrRstd[0]
    x_hat = x * rstd
    wdy = dout * w
    warp_sum_xhat_wdy = warp_reduce(
        (x_hat * wdy).reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0),
        operator.add,
        width=min_constexpr(tv_layout.shape[0][0], cute.arch.WARP_SIZE),
    )
    sum_xhat_wdy = warp_sum_xhat_wdy
    if cutlass.const_expr(warps_per_row) > 1:
        sum_xhat_wdy = block_reduce(sum_xhat_wdy, operator.add, reduction_buffer, init_val=0.0)
    mean_xhat_wdy = sum_xhat_wdy / shape[1]
    dx = (wdy - x_hat * mean_xhat_wdy) * rstd
    frgDx.store(dx.to(frgDout.element_type))
    cute.copy(copy_atom_store_Dx, frgDx, thrDx, pred=tXpX)


@cute.jit
def rmsnorm_bwd(
    mX: cute.Tensor,
    mW: cute.Tensor,
    mDout: cute.Tensor,
    mRstd: cute.Tensor,
    mDx: cute.Tensor,
    eps: cutlass.Float32 = 1e-6,
    copy_bits: cutlass.Constexpr = 128
):
    N = mX.shape[1]
    vecsize = copy_bits // mX.element_type.width
    assert N % vecsize == 0, f"Input N {N} is not divisible by vector size {vecsize}"
    num_threads = 128 if N <= 16384 else 256
    assert num_threads % cute.arch.WARP_SIZE == 0
    threads_per_row = 8 if N <= 64 else (16 if N <= 128 else (32 if N <= 3072 else (64 if N <= 6144 else (128 if N <= 16384 else 256))))
    num_blocks_N = cute.ceil_div(N // vecsize, threads_per_row)

    cols_per_block = num_threads // threads_per_row
    tiler_mn = (cols_per_block, N)
    tv_layout = cute.make_layout(
        ((threads_per_row, cols_per_block), (vecsize, num_blocks_N)),
        stride=((vecsize * cols_per_block, 1), (cols_per_block, cols_per_block * vecsize * threads_per_row))
    )

    print(f"[DSL INFO] Input Tensors:")
    print(f"[DSL INFO]   mX = {mX.type}")
    print(f"[DSL INFO]   mW = {mW.type}")
    print(f"[DSL INFO]   mDout = {mDout.type}")
    print(f"[DSL INFO]   mRstd = {mRstd.type}")
    print(f"[DSL INFO]   mDx = {mDx.type}")

    print(f"[DSL INFO] Tiling Parameters:")
    print(f"[DSL INFO]   tiler_mn = {tiler_mn} per thread block")
    print(f"[DSL INFO]   tv_layout = {tv_layout}")

    mW_expanded_layout = cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
    mW_expanded = cute.make_tensor(mW.iterator, mW_expanded_layout)
    mRstd_expanded_layout = cute.append(mRstd.layout, cute.make_layout((N,), stride=(0,)))
    mRstd_expanded = cute.make_tensor(mRstd.iterator, mRstd_expanded_layout)
    idX = cute.make_identity_tensor(mX.shape)
    gX, gW, gDout, gRstd, gDx, cX = [cute.zipped_divide(mT, tiler_mn) for mT in (mX, mW_expanded, mDout, mRstd_expanded, mDx, idX)]  # ((TileM,TileN),(RestM,RestN))
    print(f"[DSL INFO] Tiled Tensors:")
    print(f"[DSL INFO]   gX = {gX.type}")
    print(f"[DSL INFO]   gW = {gW.type}")
    print(f"[DSL INFO]   gDout = {gDout.type}")
    print(f"[DSL INFO]   gRstd = {gRstd.type}")
    print(f"[DSL INFO]   gDx = {gDx.type}")
    print(f"[DSL INFO]   coord tensor = {cX.type}")

    rmsnorm_bwd_kernel(gX, gW, gDout, gRstd, gDx, cX, eps, mX.shape, tv_layout, tiler_mn).launch(
        grid=[cute.size(gX, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
        smem=32 * 4  # TODO
    )


def run_rmsnorm_bwd(
    M,
    N,
    dtype: Type[cutlass.Numeric],
    skip_ref_check=False,
    benchmark=True,
    warmup_iterations=2,
    iterations=200,
):
    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"Input and Output Data type: {dtype}")

    torch_dtype = cutlass_torch.dtype(dtype)
    device = "cuda"
    x = torch.randn(M, N, device=device, dtype=torch_dtype)
    w = torch.randn(N, device=device, dtype=torch.float32)
    dout = torch.randn(M, N, device=device, dtype=torch_dtype)
    rstd = torch.randn(M, device=device, dtype=torch.float32)
    dx = torch.empty_like(x)

    print(f"Input tensor shapes:")
    print(f"x: {x.shape}, dtype: {x.dtype}")
    print(f"w: {w.shape}, dtype: {w.dtype}")
    print(f"out: {dout.shape}, dtype: {dout.dtype}")
    print(f"rstd: {rstd.shape}, dtype: {rstd.dtype}\n")
    print(f"dx: {dout.shape}, dtype: {dx.dtype}")

    convert_from_dlpack = lambda x: (
        from_dlpack(x, assumed_align=16)
        .mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
    )
    x_tensor, dout_tensor, dx_tensor = [convert_from_dlpack(tensor) for tensor in (x, dout, dx)]
    w_tensor = from_dlpack(w, assumed_align=16)
    rstd_tensor = from_dlpack(rstd, assumed_align=4).mark_compact_shape_dynamic(mode=0)

    print("Compiling kernel with cute.compile ...")
    compiled_func = cute.compile(rmsnorm_bwd, x_tensor, w_tensor, dout_tensor, rstd_tensor, dx_tensor)
    print("Executing kernel...")

    eps = 1e-6
    compiled_func_ref = torch.compile(rmsnorm_bwd_ref)
    if not skip_ref_check:
        compiled_func(x_tensor, w_tensor, dout_tensor, rstd_tensor, dx_tensor, eps)
        print("Verifying results...")
        torch.testing.assert_close(compiled_func_ref(x, w, dout, rstd, eps=eps), dx)
        print("Results verified successfully!")

    if benchmark:
        fn = lambda: compiled_func(x_tensor, w_tensor, dout_tensor, rstd_tensor, dx_tensor, eps)
        avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
        print(f"Kernel execution time: {avg_time:.4f} ms")
        print(f"Mem throughput: {(3 * x.numel() * dtype.width // 8) / (avg_time / 1000) / 1e9:.2f} GB/s")
        fn = lambda: compiled_func_ref(x, w, dout, rstd)
        fn()
        avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
        print(f"Ref kernel execution time: {avg_time:.4f} ms")
        print(f"Ref mem throughput: {(3 * x.numel() * dtype.width // 8) / (avg_time / 1000) / 1e9:.2f} GB/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="example of elementwise add to demonstrate the numpy/pytorch as input for kernels"
    )
    parser.add_argument("--M", default=32768, type=int)
    parser.add_argument("--N", default=1024, type=int)
    parser.add_argument("--warmup_iterations", default=10, type=int)
    parser.add_argument("--iterations", default=100, type=int)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument("--benchmark", action="store_true")

    args = parser.parse_args()
    run_rmsnorm(
        args.M,
        args.N,
        dtype=cutlass.BFloat16,
        skip_ref_check=args.skip_ref_check,
        benchmark=args.benchmark,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
    )
    # N_vals = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    # results = []
    # for N in N_vals:
    #     res = run_rmsnorm(
    #         args.M,
    #         N,
    #         dtype=cutlass.BFloat16,
    #         skip_ref_check=False,
    #         benchmark=True,
    #         warmup_iterations=args.warmup_iterations,
    #         iterations=args.iterations,
    #     )
    #     results.append(res)
    # print(results)
    # run_rmsnorm_bwd(
    #     args.M,
    #     args.N,
    #     dtype=cutlass.BFloat16,
    #     skip_ref_check=args.skip_ref_check,
    #     benchmark=args.benchmark,
    #     warmup_iterations=args.warmup_iterations,
    #     iterations=args.iterations,
    # )
    print("\nPASS")
