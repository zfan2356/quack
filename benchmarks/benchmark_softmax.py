import argparse
import time
from typing import Type

import torch
import torch.nn.functional as F
from triton.testing import do_bench

import cutlass
import cutlass.torch as cutlass_torch

from quack.softmax import softmax


def run_softmax(
    M,
    N,
    dtype: Type[cutlass.Numeric],
    warmup_iterations=10,
    iterations=1000,
):
    if not torch.cuda.is_available():
        raise RuntimeError(f"Ampere GPU is required to run this example!")

    print(f"Tensor dimensions: [{M}, {N}]")
    print(f"Input and Output Data type: {dtype}")

    torch_dtype = cutlass_torch.dtype(dtype)

    device = "cuda"
    x = 0.1 * torch.randn(M, N, device=device, dtype=torch_dtype)

    print(f"Input tensor shapes:")
    print(f"x: {x.shape}, dtype: {x.dtype}")

    print("Executing kernel...")
    out = softmax(x)

    print(f"Output tensor shapes:")
    print(f"out: {out.shape}, dtype: {out.dtype}")

    compiled_func_ref = torch.compile(lambda x: F.softmax(x, dim=-1))
    fn = lambda: softmax(x)
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
    parser.add_argument("--M", default=8192, type=int)
    parser.add_argument("--N", default=16384, type=int)
    parser.add_argument("--warmup_iterations", default=10, type=int)
    parser.add_argument("--iterations", default=100, type=int)

    args = parser.parse_args()
    torch.manual_seed(0)
    run_softmax(
        args.M,
        args.N,
        # dtype=cutlass.Float32,
        dtype=cutlass.BFloat16,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
    )

    # MN_pairs = [(32768, 1024)]
    # results = []
    # for M, N in MN_pairs:
    #     res = run_softmax(
    #         M,
    #         N,
    #         dtype=cutlass.Float32,
    #         skip_ref_check=False,
    #         benchmark=True,
    #         warmup_iterations=args.warmup_iterations,
    #         iterations=args.iterations,
    #     )
    #     results.append(res)
    # print(results)
    # print("\nPASS")
