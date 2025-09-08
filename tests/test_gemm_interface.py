import math
import pytest
from typing import Optional
import torch

from quack.gemm_interface import (
    gemm_tuned,
    gemm_ref,
    gemm_act,
    gemm_relu_ref,
    gemm_dact,
    gemm_drelu_ref,
)

def gemm(
    A: torch.Tensor, B: torch.Tensor, out_dtype: Optional[torch.dtype] = None, dynamic_scheduler: bool = False
) -> torch.Tensor:
    return gemm_tuned(A=A, B=B, C=None, out_dtype=out_dtype, dynamic_scheduler=dynamic_scheduler, config=None)


def test_gemm(n, k, input_dtype):
    device = "cuda"
    torch.random.manual_seed(0)
    m = 1920
    x = torch.randn((m, k), device=device, dtype=input_dtype, requires_grad=True)
    w = (
        torch.randn((n, k), device=device, dtype=input_dtype)
        / math.sqrt(k)
    ).requires_grad_()
    out = gemm(x, w.T)  # Disable tuning for faster test
    out_ref = gemm_ref(x.float(), w.float().T)
    out_pt = gemm_ref(x, w.T)
    assert out.shape == out_ref.shape, f"Output shape mismatch: {out.shape} vs {out_ref.shape}"
    assert out.shape == out_pt.shape, f"Output shape mismatch: {out.shape} vs {out_pt.shape}"
    assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-6
    assert x.ndim == 2
    assert w.ndim == 2


if __name__ == "__main__":
    test_gemm(1504, 4096, torch.bfloat16)