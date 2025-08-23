# Copyright (c) 2025, Tri Dao
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import custom_fwd, custom_bwd


from gemm_cublas import gemm as gemm_cb, gemm_add_ as gemm_add_cb_
# from gemm_cublas.interface import gemm_tuned as gemm_cb, gemm_add_tuned_ as gemm_add_cb_

from quack import gemm, gemm_lse  # TODO: implement these


def linear_fwd_convert_type(*tensors):
    autocast_dtype = torch.get_autocast_dtype("cuda")
    if torch.is_autocast_enabled():
        tensors = tuple(t.to(dtype=autocast_dtype) for t in tensors)
    return tensors


def linear_fwd_postprocess(ctx, x, weight, weight_og, needs_input_grad, needs_weight_grad):
    if not needs_input_grad:
        weight, weight_og = None, None
    if not needs_weight_grad:
        x = None
    ctx.save_for_backward(x, weight, weight_og if ctx.fuse_grad_accum else None)


def linear_bwd_compute_input_grad(ctx, dout, weight, use_tuned_gemm=True, sm_carveout=0):
    if ctx.needs_input_grad[0]:
        assert weight is not None
        # return gemm(dout, weight) if use_tuned_gemm else (dout @ weight)
        return (
            gemm(dout, weight, sm_carveout=sm_carveout)
            if use_tuned_gemm
            else gemm_cb(dout, weight, sm_carveout=sm_carveout)
        )
    else:
        return None


def linear_bwd_compute_weight_grad(ctx, dout, x, weight_og, sm_carveout=0):
    if ctx.needs_input_grad[1]:
        assert x is not None
        x = x.reshape(-1, x.shape[-1])
        # fuse_grad_accum is not compatible with torch.compile
        if not ctx.fuse_grad_accum or weight_og.grad is None or torch.compiler.is_compiling():
            dweight = gemm_cb(dout.T, x, out_dtype=ctx.weight_dtype, sm_carveout=sm_carveout)
        else:
            # print("Using fuse grad accum in Linear", dout.shape, x.shape, weight_og.grad.shape)
            gemm_add_cb_(dout.T, x, weight_og.grad, sm_carveout=sm_carveout)
            dweight = weight_og.grad
            weight_og.grad = None  # So that pytorch doesn't add dweight to weight_og.grad again
    else:
        dweight = None
    return dweight


class LinearFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, x, weight, fuse_grad_accum=False):
        """
        x: (..., in_features)
        weight: (out_features, in_features)
        out: (..., out_features)
        """
        ctx.weight_dtype = weight.dtype
        ctx.fuse_grad_accum = fuse_grad_accum
        weight_og = weight
        x, weight = linear_fwd_convert_type(x, weight)
        batch_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        # out = F.linear(x, weight)
        out = gemm(x, weight.T)
        linear_fwd_postprocess(
            ctx,
            x,
            weight,
            weight_og,
            needs_input_grad=ctx.needs_input_grad[0],
            needs_weight_grad=ctx.needs_input_grad[1],
        )
        return out.reshape(*batch_shape, out.shape[-1])

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, dout):
        """
        dout: (..., out_features)
        """
        x, weight, weight_og = ctx.saved_tensors  # weight_og is None if not ctx.fuse_grad_accum
        batch_shape = dout.shape[:-1]
        dout = dout.reshape(-1, dout.shape[-1])
        dx = linear_bwd_compute_input_grad(ctx, dout, weight, use_tuned_gemm=True)
        dx = dx.reshape(*batch_shape, dx.shape[-1]) if dx is not None else None
        dweight = linear_bwd_compute_weight_grad(ctx, dout, x, weight_og)
        return dx, dweight, None


def linear_func(x, weight, fuse_grad_accum=False):
    return LinearFunc.apply(x, weight, fuse_grad_accum)


class LinearLSEFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, x, weight, fuse_grad_accum=False):
        """
        x: (..., in_features)
        weight: (out_features, in_features)
        out: (..., out_features)
        """
        needs_weight_grad = weight.requires_grad
        needs_input_grad = x.requires_grad
        ctx.weight_dtype = weight.dtype
        ctx.fuse_grad_accum = fuse_grad_accum
        weight_og = weight
        x, weight = linear_fwd_convert_type(x, weight)
        batch_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        out, lse = gemm_lse(x, weight.T)
        lse = lse.reshape(*batch_shape)
        linear_fwd_postprocess(ctx, x, weight, weight_og, needs_weight_grad, needs_input_grad)
        ctx.mark_non_differentiable(lse)
        return out.reshape(*batch_shape, out.shape[-1]), lse

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, dout, dlse_ignored):
        """
        dout: (..., out_features)
        """
        x, weight, weight_og = ctx.saved_tensors  # weight_og is None if not ctx.fuse_grad_accum
        batch_shape = dout.shape[:-1]
        dout = dout.reshape(-1, dout.shape[-1])
        # cuBLAS seems faster for this so we just use it instead of cutlass gemm
        dx = linear_bwd_compute_input_grad(ctx, dout, weight, use_tuned_gemm=False)
        dx = dx.reshape(*batch_shape, dx.shape[-1]) if dx is not None else None
        dweight = linear_bwd_compute_weight_grad(ctx, dout, x, weight_og)
        return dx, dweight, None


def linear_lse_func(x, weight, fuse_grad_accum=False):
    return LinearLSEFunc.apply(x, weight, fuse_grad_accum)


class Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
        fuse_grad_accum: bool = False,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.fuse_grad_accum = fuse_grad_accum

    def forward(self, input: Tensor) -> Tensor:
        if self.bias is None and input.is_cuda:
            return linear_func(input, self.weight, fuse_grad_accum=self.fuse_grad_accum)
        else:
            return F.linear(input, self.weight, self.bias)


class LinearLSE(Linear):
    def forward(self, input: Tensor) -> Tensor:
        if self.bias is None and input.is_cuda:
            return linear_lse_func(input, self.weight, fuse_grad_accum=self.fuse_grad_accum)
        else:
            out = F.linear(input, self.weight, self.bias)
            lse = torch.logsumexp(out, dim=-1)
            return out, lse
