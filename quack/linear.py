# Copyright (c) 2025, Tri Dao
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import custom_fwd, custom_bwd


# from gemm_cublas import gemm as gemm_cb, gemm_add_ as gemm_add_cb_
# from gemm_cublas.interface import gemm_tuned as gemm_cb, gemm_add_tuned_ as gemm_add_cb_

from quack.gemm_interface import (
    gemm,
    gemm_tuned,
    gemm_act,
    gemm_act_tuned,
    gemm_dact,
    gemm_dact_tuned,
)


def linear_fwd_convert_type(*tensors):
    autocast_dtype = torch.get_autocast_dtype("cuda")
    if torch.is_autocast_enabled():
        tensors = tuple(t.to(dtype=autocast_dtype) for t in tensors)
    return tensors


def linear_fwd_postprocess(ctx, x, weight, weight_og, needs_x_w_grad):
    needs_input_grad, needs_weight_grad = needs_x_w_grad
    if not needs_input_grad:
        weight, weight_og = None, None
    if not needs_weight_grad:
        x = None
    ctx.save_for_backward(x, weight, weight_og if ctx.fuse_grad_accum else None)


def linear_bwd_compute_input_grad(ctx, dout, weight, matmul_fn):
    if ctx.needs_input_grad[0]:
        assert weight is not None
        return matmul_fn(dout, weight)
    else:
        return None


def linear_bwd_compute_weight_grad(ctx, dout, x, weight_og, matmul_fn):
    if ctx.needs_input_grad[1]:
        assert x is not None
        x = x.reshape(-1, x.shape[-1])
        # fuse_grad_accum is not compatible with torch.compile
        # if not ctx.fuse_grad_accum or weight_og.grad is None or torch.compiler.is_compiling():
        if True:
            dweight = matmul_fn(dout.T, x, out_dtype=ctx.weight_dtype)
        else:
            # print("Using fuse grad accum in Linear", dout.shape, x.shape, weight_og.grad.shape)
            # TODO: support gemm_add_
            # gemm_add_(dout.T, x, weight_og.grad)
            dweight = weight_og.grad
            weight_og.grad = None  # So that pytorch doesn't add dweight to weight_og.grad again
    else:
        dweight = None
    return dweight


class LinearFunc(torch.autograd.Function):
    matmul_fwd_fn = gemm
    matmul_bwd_dx = partial(gemm, dynamic_scheduler=True)
    matmul_bwd_dw = partial(gemm, dynamic_scheduler=True)

    # Use classmethod instead of staticmethod to allow inheritance
    @classmethod
    @custom_fwd(device_type="cuda")
    def forward(cls, ctx, x, weight, fuse_grad_accum=False):
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
        out = cls.matmul_fwd_fn(x, weight.T)
        linear_fwd_postprocess(ctx, x, weight, weight_og, needs_x_w_grad=ctx.needs_input_grad[:2])
        return out.reshape(*batch_shape, out.shape[-1])

    @classmethod
    @custom_bwd(device_type="cuda")
    def backward(cls, ctx, dout, *args):
        """
        dout: (..., out_features)
        """
        x, weight, weight_og = ctx.saved_tensors  # weight_og is None if not ctx.fuse_grad_accum
        batch_shape = dout.shape[:-1]
        dout = dout.reshape(-1, dout.shape[-1])
        dx = linear_bwd_compute_input_grad(ctx, dout, weight, cls.matmul_bwd_dx)
        dx = dx.reshape(*batch_shape, dx.shape[-1]) if dx is not None else None
        dweight = linear_bwd_compute_weight_grad(ctx, dout, x, weight_og, cls.matmul_bwd_dw)
        # return extra Nones for other classes that inherit from LinearFunc
        return dx, dweight, *([None] * 10)


class LinearUntunedFunc(LinearFunc):
    # Passing in config=None to disable tuning at runtime
    matmul_fwd_fn = partial(gemm_tuned.fn, config=None)
    matmul_bwd_dx = partial(gemm_tuned.fn, dynamic_scheduler=True, config=None)
    matmul_bwd_dw = partial(gemm_tuned.fn, dynamic_scheduler=True, config=None)


def linear_func(x, weight, fuse_grad_accum=False, tuned=True):
    if tuned:
        return LinearFunc.apply(x, weight, fuse_grad_accum)
    else:
        return LinearUntunedFunc.apply(x, weight, fuse_grad_accum)


class LinearActFunc(LinearFunc):
    matmul_fwd_fn = gemm_act

    # Use classmethod instead of staticmethod to allow inheritance
    @classmethod
    @custom_fwd(device_type="cuda")
    def forward(cls, ctx, x, weight, activation, store_preact=True, fuse_grad_accum=False):
        """
        x: (..., in_features)
        weight: (out_features, in_features)
        out: (..., out_features)
        Return both out and post-activation, but only out is differentiable.
        """
        ctx.weight_dtype = weight.dtype
        ctx.fuse_grad_accum = fuse_grad_accum
        weight_og = weight
        x, weight = linear_fwd_convert_type(x, weight)
        batch_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        out, postact = cls.matmul_fwd_fn(
            x, weight.T, activation=activation, store_preact=store_preact
        )
        linear_fwd_postprocess(ctx, x, weight, weight_og, needs_x_w_grad=ctx.needs_input_grad[:2])
        if out is not None:
            out = out.reshape(*batch_shape, out.shape[-1])
        ctx.mark_non_differentiable(postact)
        ctx.set_materialize_grads(False)  # We don't want to materialize grads for postact
        return out, postact.reshape(*batch_shape, postact.shape[-1])


class LinearActUntunedFunc(LinearActFunc):
    # Passing in config=None to disable tuning at runtime
    matmul_fwd_fn = partial(gemm_act_tuned.fn, config=None)
    matmul_bwd_dx = partial(gemm_tuned.fn, dynamic_scheduler=True, config=None)
    matmul_bwd_dw = partial(gemm_tuned.fn, dynamic_scheduler=True, config=None)


def linear_act_func(x, weight, activation, store_preact=True, fuse_grad_accum=False, tuned=True):
    if tuned:
        return LinearActFunc.apply(x, weight, activation, store_preact, fuse_grad_accum)
    else:
        return LinearActUntunedFunc.apply(x, weight, activation, store_preact, fuse_grad_accum)


class DActLinearFunc(LinearFunc):
    matmul_bwd_dx = partial(gemm_dact, dynamic_scheduler=True)

    # Use classmethod instead of staticmethod to allow inheritance
    @classmethod
    @custom_fwd(device_type="cuda")
    def forward(cls, ctx, preact, weight, x, activation, fuse_grad_accum=False):
        """
        x: (..., in_features)
        weight: (out_features, in_features)
        out: (..., out_features)
        Takes in an extra preact argument which is the pre-activation, to be used in the backward pass.
        """
        ctx.weight_dtype = weight.dtype
        ctx.fuse_grad_accum = fuse_grad_accum
        weight_og = weight
        x, weight = linear_fwd_convert_type(x, weight)
        batch_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        out = cls.matmul_fwd_fn(x, weight.T)
        # Store preact instead of x, we will recompute x in the backward pass
        linear_fwd_postprocess(
            ctx, preact, weight, weight_og, needs_x_w_grad=ctx.needs_input_grad[:2]
        )
        ctx.activation = activation
        return out.reshape(*batch_shape, out.shape[-1])

    @classmethod
    @custom_bwd(device_type="cuda")
    def backward(cls, ctx, dout):
        """
        dout: (..., out_features)
        """
        # weight_og is None if not ctx.fuse_grad_accum
        preact, weight, weight_og = ctx.saved_tensors
        batch_shape = dout.shape[:-1]
        dout = dout.reshape(-1, dout.shape[-1])
        preact = preact.reshape(-1, preact.shape[-1])
        if ctx.needs_input_grad[0]:
            assert weight is not None
            dpreact, x = cls.matmul_bwd_dx(dout, weight, preact, activation=ctx.activation)
        else:
            dpreact, x = None, None
        dpreact = dpreact.reshape(*batch_shape, dpreact.shape[-1]) if dpreact is not None else None
        dweight = linear_bwd_compute_weight_grad(ctx, dout, x, weight_og, cls.matmul_bwd_dw)
        return dpreact, dweight, *([None] * 3)


class DActLinearUntunedFunc(DActLinearFunc):
    # Passing in config=None to disable tuning at runtime
    matmul_fwd_fn = partial(gemm_tuned.fn, config=None)
    matmul_bwd_dx = partial(gemm_dact_tuned.fn, dynamic_scheduler=True, config=None)
    matmul_bwd_dw = partial(gemm_tuned.fn, dynamic_scheduler=True, config=None)


def act_linear_func(preact, weight, x, activation, fuse_grad_accum=False, tuned=True):
    if tuned:
        return DActLinearFunc.apply(preact, weight, x, activation, fuse_grad_accum)
    else:
        return DActLinearUntunedFunc.apply(preact, weight, x, activation, fuse_grad_accum)


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
