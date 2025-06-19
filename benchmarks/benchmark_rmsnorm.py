import argparse
import time
from typing import Type

import torch
from triton.testing import do_bench

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

from quack.rmsnorm import rmsnorm, rmsnorm_ref, rstd_ref, rmsnorm_bwd, rmsnorm_bwd_ref

try:
    import cudnn
    from quack.rmsnorm import rmsnorm_cudnn_setup
except ImportError:
    cudnn = None


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
        dtype=cutlass.Float32,
        skip_ref_check=args.skip_ref_check,
        benchmark=args.benchmark,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
    )
    MN_pairs = [(32768, 256), (32768, 512), (32768, 1024), (32768, 2048), (32768, 4096), (32768, 8192), (32768, 16384), (32768, 32768), (32768, 65536), (16384, 131072), (8192, 262144)]
    results = []
    for M, N in MN_pairs:
        res = run_rmsnorm(
            M,
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
