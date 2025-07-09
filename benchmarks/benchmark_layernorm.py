import argparse
import time
from typing import Type

import torch
from triton.testing import do_bench

import cutlass
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from quack.layernorm import layernorm, layernorm_ref, rstd_ref, mean_ref
import cutlass.cute as cute

try:
    import cudnn
except ImportError:
    cudnn = None


def run_layernorm(
    M,
    N,
    dtype: Type[cutlass.Numeric],
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
    out, rstd, mean = layernorm(x, w, eps=eps, return_rstd=True, return_mean=True)

    compiled_func_ref = torch.compile(layernorm_ref)

    fn = lambda: layernorm(x, w, eps=eps)
    time.sleep(0.5)
    avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
    mem_bw = (2 * x.numel() * dtype.width // 8) / (avg_time / 1000) / 1e9
    print(f"Kernel execution time: {avg_time:.4f} ms")
    print(f"Mem throughput: {mem_bw:.2f} GB/s")

    fn = lambda: compiled_func_ref(x, w, eps=eps)
    for _ in range(5):
        fn()  # warm up
    time.sleep(0.5)
    avg_time = do_bench(fn, warmup=warmup_iterations, rep=iterations)
    mem_bw_ref = (2 * x.numel() * dtype.width // 8) / (avg_time / 1000) / 1e9
    print(f"Ref kernel execution time: {avg_time:.4f} ms")
    print(f"Ref mem throughput: {mem_bw_ref:.2f} GB/s")

    if cudnn is not None:
        run_cudnn = layernorm_cudnn_setup(M, N, torch_dtype)
        time.sleep(0.5)
        avg_time = do_bench(run_cudnn, warmup=warmup_iterations, rep=iterations)
        mem_bw_cudnn = (2 * x.numel() * dtype.width // 8) / (avg_time / 1000) / 1e9
        print(f"Cudnn kernel execution time: {avg_time:.4f} ms")
        print(f"Cudnn mem throughput: {mem_bw_cudnn:.2f} GB/s")

    return mem_bw, mem_bw_ref


def layernorm_cudnn_setup(M, N, dtype):
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
        name="layernorm",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="example of elementwise add to demonstrate the numpy/pytorch as input for kernels"
    )
    parser.add_argument("--M", default=32768, type=int)
    parser.add_argument("--N", default=16384, type=int)
    parser.add_argument("--warmup_iterations", default=10, type=int)
    parser.add_argument("--iterations", default=100, type=int)

    args = parser.parse_args()

    run_layernorm(
        args.M,
        args.N,
        dtype=cutlass.BFloat16,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
    )
