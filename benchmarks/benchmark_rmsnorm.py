import argparse
import time
from typing import Type

import torch
from triton.testing import do_bench

import cutlass
import cutlass.torch as cutlass_torch

from quack.rmsnorm import rmsnorm, rmsnorm_ref, rstd_ref

try:
    import cudnn
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

    print(f"Input tensor shapes:")
    print(f"x: {x.shape}, dtype: {x.dtype}")
    print(f"w: {w.shape}, dtype: {w.dtype}")

    eps = 1e-6

    print("Executing kernel...")
    out, rstd = rmsnorm(x, w, eps=eps, return_rstd=True)

    compiled_func_ref = torch.compile(rmsnorm_ref)
    if not skip_ref_check:
        print("Verifying results...")
        out_ref = compiled_func_ref(x, w, eps=eps)
        torch.testing.assert_close(out_ref, out)
        torch.testing.assert_close(rstd_ref(x, eps=eps), rstd)
        print("Results verified successfully!")

    if benchmark:
        fn = lambda: rmsnorm(x, w, eps=eps)
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
            run_cudnn = rmsnorm_cudnn_setup(M, N, torch_dtype)
            time.sleep(0.5)
            avg_time = do_bench(run_cudnn, warmup=warmup_iterations, rep=iterations)
            mem_bw_cudnn = (2 * x.numel() * dtype.width // 8) / (avg_time / 1000) / 1e9
            print(f"Cudnn kernel execution time: {avg_time:.4f} ms")
            print(f"Cudnn mem throughput: {mem_bw_cudnn:.2f} GB/s")

        return mem_bw, mem_bw_ref


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
    # MN_pairs = [(32768, 256), (32768, 512), (32768, 1024), (32768, 2048), (32768, 4096), (32768, 8192), (32768, 16384), (32768, 32768), (32768, 65536), (16384, 131072), (8192, 262144)]
    MN_pairs = [(32768, 2048)]
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
