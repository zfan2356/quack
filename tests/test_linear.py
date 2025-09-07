# Copyright (C) 2025, Tri Dao.
import math
import pytest
import torch
import torch.nn.functional as F

from quack.linear import linear_func, linear_act_func


@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("out_features", [1504, 2048])
@pytest.mark.parametrize("in_features", [736, 4096])
# @pytest.mark.parametrize("out_features", [2048])
# @pytest.mark.parametrize("in_features", [4096])
def test_linear(in_features, out_features, input_dtype):
    device = "cuda"
    torch.random.manual_seed(0)
    m = 1920
    x = torch.randn((m, in_features), device=device, dtype=input_dtype, requires_grad=True)
    x = x[::2]  # Testing non-contiguous
    w = (
        torch.randn((out_features, in_features), device=device, dtype=input_dtype)
        / math.sqrt(in_features)
    ).requires_grad_()
    out = linear_func(x, w, tuned=False)  # Disable tuning for faster test
    out_ref = F.linear(x.float(), w.float())
    out_pt = F.linear(x, w)
    assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-6
    dout = torch.randn_like(out)
    dx, dw = torch.autograd.grad(out, (x, w), dout)
    dx_ref, dw_ref = torch.autograd.grad(out_ref, (x, w), dout)
    dx_pt, dw_pt = torch.autograd.grad(out_pt, (x, w), dout)
    assert (dx - dx_ref).abs().max() < 2 * (dx_pt - dx_ref).abs().max() + 1e-6
    assert (dw - dw_ref).abs().max() < 2 * (dw_pt - dw_ref).abs().max() + 1e-6


@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("out_features", [1504, 2048])
@pytest.mark.parametrize("in_features", [736, 4096])
# @pytest.mark.parametrize("out_features", [2048])
# @pytest.mark.parametrize("in_features", [4096])
def test_linear_act(in_features, out_features, input_dtype):
    device = "cuda"
    torch.random.manual_seed(0)
    m = 1920
    x = torch.randn((m, in_features), device=device, dtype=input_dtype, requires_grad=True)
    x = x[::2]  # Testing non-contiguous
    w = (
        torch.randn((out_features, in_features), device=device, dtype=input_dtype)
        / math.sqrt(in_features)
    ).requires_grad_()
    _, out = linear_act_func(
        x, w, "gelu_tanh_approx", tuned=False
    )  # Disable tuning for faster test
    out_ref = F.gelu(F.linear(x.float(), w.float()), approximate="tanh")
    out_pt = F.gelu(F.linear(x, w), approximate="tanh")
    assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-6
