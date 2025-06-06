import math
import argparse
import torch
import torch.nn.functional as F
import time
import operator
from typing import Type, Callable, Union, Tuple

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


log2_e = 1.4426950408889634

def exp_arch(val):
    return cute.arch.exp2(val * log2_e)

def exp_math(val):
    return cute.math.exp2(val * log2_e)

@cute.jit
def minimum(a: cutlass.Constexpr, b: cutlass.Constexpr) -> cutlass.Constexpr:
    return a if a < b else b

@cute.jit
def maximum(a: cutlass.Constexpr, b: cutlass.Constexpr) -> cutlass.Constexpr:
    return a if a > b else b

@cute.jit
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
def set_block_rank(smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: cute.Int32, *, loc=None, ip=None) -> cutlass.Int32:
    """Map the given smem pointer to the address at another CTA rank in the cluster.
    """
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return cutlass.Int32(
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


@dsl_user_op
def store_shared_remote(
    val: float | cute.Float32, smem_ptr: cute.Pointer, mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: cute.typing.Int, *, loc=None, ip=None
) -> None:
    remote_smem_ptr_i32 = set_block_rank(smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [remote_smem_ptr_i32, cute.Float32(val).ir_value(loc=loc, ip=ip), remote_mbar_ptr_i32],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [$0], $1, [$2];",
        "r,f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@cute.jit
def cluster_reduce(val: cute.Numeric, op: Callable, reduction_buffer: cute.Tensor, mbar_ptr: cute.Pointer, init_val: cute.Numeric = 0.0) -> cute.Numeric:
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    warps_per_row, cluster_n = reduction_buffer.shape[1]
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if lane_idx < cluster_n:
        store_shared_remote(
            val, elem_pointer(reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster))),
            mbar_ptr, peer_cta_rank_in_cluster=lane_idx
        )
    
    cute.arch.mbarrier_wait(mbar_ptr, phase=0)
    block_reduce_val = init_val
    num_iter = cute.ceil_div(warps_per_row * cluster_n, cute.arch.WARP_SIZE)
    for i in cutlass.range_constexpr(num_iter):
        idx = lane_idx + i * cute.arch.WARP_SIZE
        if idx < cute.size(reduction_buffer, mode=[1]):
            block_reduce_val = op(block_reduce_val, reduction_buffer[row_idx, idx])
    ret = warp_reduce(block_reduce_val, op)
    return ret

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
    warps_per_row = maximum(tv_layout.shape[0][0] // cute.arch.WARP_SIZE, 1)
    # reduction_buffer_layout = cute.make_ordered_layout((num_warps // warps_per_row, warps_per_row), order=(1, 0))

    max_val_reduction_buffer_layout = cute.make_ordered_layout((num_warps // warps_per_row, warps_per_row if cluster_n == 1 else (warps_per_row, cluster_n)), order=(1, 0))
    denom_reduction_buffer_layout   = cute.make_ordered_layout((num_warps // warps_per_row, warps_per_row if cluster_n == 1 else (warps_per_row, cluster_n)), order=(1, 0))
    
    max_val_reduction_buffer = smem.allocate_tensor(cutlass.Float32, max_val_reduction_buffer_layout, byte_alignment=4)
    denom_reduction_buffer   = smem.allocate_tensor(cutlass.Float32, denom_reduction_buffer_layout, byte_alignment=4)


    max_val_mbar_ptr = cute.Pointer()
    denom_mbar_ptr = cute.Pointer()
    
    if cluster_n > 1:
        max_val_mbar_ptr = smem.allocate(cutlass.Int64.width // 8, byte_alignment=8)
        denom_mbar_ptr   = smem.allocate(cutlass.Int64.width // 8, byte_alignment=8)

    tXgX = thr_copy_X_async.partition_S(blkX)
    tXsX = thr_copy_X_async.partition_S(sX)
    tXgO = thr_copy_O.partition_D(blkOut) 
    tXcX = thr_copy_X.partition_S(blkCrd)[(0, None), None, None]

    # allocate fragments for gmem->rmem
    tXrX, tXrO = [cute.make_fragment_like(thr) for thr in (tXgX, tXgO)]

    if cluster_n > 1:
        if tidx == 0:
            cute.arch.mbarrier_init_arrive_cnt(max_val_mbar_ptr, 1)
            cute.arch.mbarrier_init_arrive_cnt(denom_mbar_ptr, 1)
        cute.arch.mbarrier_init_fence()
        if tidx == 0:
            cute.arch.mbarrier_init_tx_bytes(max_val_mbar_ptr, num_warps * cluster_n * cutlass.Float32.width // 8)
            cute.arch.mbarrier_init_tx_bytes(denom_mbar_ptr, num_warps * cluster_n * cutlass.Float32.width // 8)
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

    ######## phase 1, get row max and softmax denominator (online softmax approach)
    max_x       = x.reduce(cute.ReductionOp.MAX, init_val=float('-inf'), reduction_profile=0)
    max_x       = warp_reduce(
        max_x, cute.arch.fmax,
        width=minimum(tv_layout.shape[0][0], cute.arch.WARP_SIZE),
    )

    if cutlass.const_expr(warps_per_row * cluster_n) > 1:
        if cutlass.const_expr(cluster_n) == 1:
            max_x = block_reduce(max_x, cute.arch.fmax, max_val_reduction_buffer, init_val=max_x)           
        else:
            cute.arch.cluster_wait()
            max_x = cluster_reduce(max_x, cute.arch.fmax, max_val_reduction_buffer, max_val_mbar_ptr, init_val=max_x)

    nom   = exp_math(x - max_x)
    denom = nom.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0)
    denom = warp_reduce(
        denom, operator.add,
        width=minimum(tv_layout.shape[0][0], cute.arch.WARP_SIZE),
    )
    
    if cutlass.const_expr(warps_per_row * cluster_n) > 1:
        if cutlass.const_expr(cluster_n) == 1:            
            denom     = block_reduce(denom, operator.add, denom_reduction_buffer, init_val=0.0)
        else:            
            denom     = cluster_reduce(denom, operator.add, denom_reduction_buffer, denom_mbar_ptr, init_val=0.0)


    ######## phase 2, actual softmax computation    
    y   = nom / denom

    tXrO.store(y.to(tXrO.element_type))
    tOcX = thr_copy_O.partition_S(blkCrd)[(0, None), None, None]
    tOpO = cute.make_fragment_like(tXgO[(0, None), None, None], cutlass.Boolean)
    for i in range(cute.size(tOpO)):
        tOpO[i] = cute.elem_less(tOcX[i][1], shape[1])
    if tXcX[0][0] < shape[0]:
        cute.copy(copy_atom_store_O, tXrO, tXgO, pred=tOpO)


@cute.jit
def softmax(
    # mX_: cute.Tensor,
    mX: cute.Tensor,
    mOut: cute.Tensor,
    stream: cuda.CUstream,
    N: cutlass.Constexpr,
    copy_bits: cutlass.Constexpr = 128
):
    N = mX.shape[1]
    vecsize = copy_bits // mX.element_type.width
    assert N % vecsize == 0, f"Input N {N} is not divisible by vector size {vecsize}"
    num_threads = 128 if N <= 16384 else 256
    num_warps = num_threads // cute.arch.WARP_SIZE
    assert num_threads % cute.arch.WARP_SIZE == 0
    threads_per_row = 8 if N <= 64 else (16 if N <= 128 else (32 if N <= 3072 else (64 if N <= 6144 else (128 if N <= 16384 else 256))))
    # cluster_n = 4 is faster and cluster_n = 2 for N=64k for some reason
    # Similarly cluster_n = 8 is faster for N=128k
    cluster_n = 1 if N <= 8 * 1024 else (2 if N <= 16 * 1024 else (8 if N <= 64 * 1024 else 16))

    num_blocks_N = cute.ceil_div(N // vecsize, threads_per_row * cluster_n)

    cols_per_block = num_threads // threads_per_row
    tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_row)  # This rounds up N
    tv_layout = cute.make_layout(
        ((threads_per_row, cols_per_block), (vecsize, num_blocks_N)),
        stride=((vecsize * cols_per_block, 1), (cols_per_block, cols_per_block * vecsize * threads_per_row))
    )
    print(tv_layout)

    print(f"[DSL INFO] Input Tensors:")
    print(f"[DSL INFO]   mX = {mX.type}")
    print(f"[DSL INFO]   mOut = {mOut.type}")

    print(f"[DSL INFO] Tiling Parameters:")
    print(f"[DSL INFO]   tiler_mn = {tiler_mn} per thread block")
    print(f"[DSL INFO]   tv_layout = {tv_layout}")

    idX = cute.make_identity_tensor(mX.shape)
    gX, gO, cX = [cute.zipped_divide(mT, tiler_mn) for mT in (mX, mOut, idX)]  # ((TileM,TileN),(RestM,RestN))
    print(f"[DSL INFO] Tiled Tensors:")
    print(f"[DSL INFO]   gX = {gX.type}")
    print(f"[DSL INFO]   gO = {gO.type}")
    print(f"[DSL INFO]   coord tensor = {cX.type}")
    
    smem_allocated = cute.size_in_bytes(mX.element_type, gX.layout[0]) + 2 * num_warps * cluster_n * (cutlass.Float32.width // 8) + 2 * (cutlass.Int64.width // 8)
    softmax_kernel(gX, gO, cX, mX.shape, tv_layout, tiler_mn, cluster_n).launch(
        grid=[cute.size(gX, mode=[1, 0]), cluster_n, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
        # Launching with cluster=[1, 1, 1] instead of None slows down the kernel by ~8us
        cluster=[1, cluster_n, 1] if cluster_n > 1 else None,
        smem=smem_allocated,
        stream=stream,
    )


def run_softmax(
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
    x = 0.1 * torch.randn(M, N, device=device, dtype=torch_dtype)
    out = torch.empty_like(x)

    print(f"Input tensor shapes:")
    print(f"x: {x.shape}, dtype: {x.dtype}")
    print(f"out: {out.shape}, dtype: {out.dtype}")

    convert_from_dlpack = lambda x: (
        from_dlpack(x, assumed_align=128)
        .mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
    )
    x_tensor, out_tensor = [convert_from_dlpack(tensor) for tensor in (x, out)]

    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    print("Compiling kernel with cute.compile ...")
    compiled_func = cute.compile(softmax, x_tensor, out_tensor, stream, x.shape[1])
    print("Executing kernel...")

    compiled_func(x_tensor, out_tensor, stream)

    compiled_func_ref = torch.compile(lambda x: F.softmax(x, dim=-1))
    if not skip_ref_check:
        # compiled_func(x_tensor, w_tensor, out_tensor, rstd_tensor, stream, eps)
        print("Verifying results...")
        out_ref = compiled_func_ref(x)
        if dtype == cutlass.BFloat16:
            torch.testing.assert_close(out_ref, out, atol=1e-3, rtol=1e-3)
        elif dtype == cutlass.Float32:
            torch.testing.assert_close(out_ref, out, atol=1e-4, rtol=1e-4)
        else:
            raise NotImplementedError()
        print("Results verified successfully!")

    
    if benchmark:
        fn = lambda: compiled_func(x_tensor, out_tensor, stream)
        time.sleep(0.5)
        avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
        mem_bw = round(2 * x.numel() * dtype.width // 8 / (avg_time / 1000) / 1e9)
        print(f"Kernel execution time: {avg_time:.4f} ms")
        print(f"Mem throughput: {mem_bw:.2f} GB/s")

        fn = lambda: compiled_func_ref(x)
        for _ in range(5): fn()  # warm up
        time.sleep(0.5)
        avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
        mem_bw_ref = round(2 * x.numel() * dtype.width // 8 / (avg_time / 1000) / 1e9)
        print(f"Ref kernel execution time: {avg_time:.4f} ms")
        print(f"Ref mem throughput: {mem_bw_ref:.2f} GB/s")

        return mem_bw, mem_bw_ref



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="example of elementwise add to demonstrate the numpy/pytorch as input for kernels"
    )
    parser.add_argument("--M", default=16384, type=int)
    parser.add_argument("--N", default=16384, type=int)
    parser.add_argument("--warmup_iterations", default=10, type=int)
    parser.add_argument("--iterations", default=100, type=int)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument("--benchmark", action="store_true")

    args = parser.parse_args()
    torch.manual_seed(0)
    run_softmax(
        args.M,
        args.N,
        dtype=cutlass.Float32,
        skip_ref_check=args.skip_ref_check,
        benchmark=args.benchmark,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
    )
    
    N_vals = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    results = []
    for N in N_vals:
        res = run_softmax(
            args.M,
            N,
            dtype=cutlass.Float32,
            skip_ref_check=False,
            benchmark=True,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
        )
        results.append(res)
    print(results)
    print("\nPASS")
    
    # BF16:
    # [(1363, 154), (1814, 304), (2257, 603), (2597, 1183), (2796, 1486), (2930, 1691), (2841, 1806), (2643, 1539), (2747, 1217), (2458, 1205)]
    
    # FP32:
    # [(1880, 1120), (2312, 2058), (2643, 2426), (2787, 2550), (2900, 1894), (2968, 1528), (2986, 1480), (3042, 1468), (3020, 1450), (3034, 1495)]