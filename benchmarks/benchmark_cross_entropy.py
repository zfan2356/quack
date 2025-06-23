import argparse
import time
from typing import Type

import torch
import torch.nn.functional as F
from triton.testing import do_bench

import cutlass
import cutlass.torch as cutlass_torch

from quack.cross_entropy import cross_entropy


def run_cross_entropy(
    M,
    N,
    dtype: Type[cutlass.Numeric],
    warmup_iterations=2,
    iterations=200,
):
    if not torch.cuda.is_available():
        raise RuntimeError(f"Ampere GPU is required to run this example!")

    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"Input Data type: {dtype}")

    torch_dtype = cutlass_torch.dtype(dtype)

    device = "cuda"
    x = 0.1 * torch.randn(M, N, device=device, dtype=torch_dtype)
    target = torch.randint(0, N, (M,), device=device, dtype=torch.int64)

    loss = cross_entropy(x, target)

    compiled_func_ref = torch.compile(lambda x, target: F.cross_entropy(x, target, reduction='none'))

    fn = lambda: cross_entropy(x, target)
    time.sleep(0.5)
    avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
    # Memory bandwidth calculation: read x (M*N elements) + read target (M elements) + write loss (M elements)
    mem_bytes = (M * N + M + M) * dtype.width // 8
    mem_bw = round(mem_bytes / (avg_time / 1000) / 1e9)
    print(f"Kernel execution time: {avg_time:.4f} ms")
    print(f"Mem throughput: {mem_bw:.2f} GB/s")

    fn = lambda: compiled_func_ref(x, target)
    for _ in range(5): fn()  # warm up
    time.sleep(0.5)
    avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
    mem_bw_ref = round(mem_bytes / (avg_time / 1000) / 1e9)
    print(f"Ref kernel execution time: {avg_time:.4f} ms")
    print(f"Ref mem throughput: {mem_bw_ref:.2f} GB/s")

    return mem_bw, mem_bw_ref


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="benchmark cross entropy kernel implementation"
    )
    parser.add_argument("--M", default=8192, type=int)
    parser.add_argument("--N", default=16384, type=int)
    parser.add_argument("--dtype", type=cutlass.dtype, choices=[cutlass.BFloat16, cutlass.Float16, cutlass.Float32], default=cutlass.BFloat16)
    parser.add_argument("--warmup_iterations", default=10, type=int)
    parser.add_argument("--iterations", default=100, type=int)

    args = parser.parse_args()
    torch.manual_seed(0)

    run_cross_entropy(
        args.M,
        args.N,
        dtype=args.dtype,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
    )

    # MN_pairs = [(32768, 256), (32768, 512), (32768, 1024), (32768, 2048), (32768, 4096), (32768, 8192), (32768, 16384), (32768, 32768), (32768, 65536), (16384, 131072), (8192, 262144)]
    # # MN_pairs = [(32768, 1024)]
    # results = []
    # for M, N in MN_pairs:
    #     res = run_cross_entropy(
    #         M,
    #         N,
    #         dtype=args.dtype,
    #         warmup_iterations=args.warmup_iterations,
    #         iterations=args.iterations,
    #     )
    #     results.append(res)
    # # print(results)
    # print([x for x, _ in results])
