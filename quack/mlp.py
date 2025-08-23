# Copyright (c) 2025, Tri Dao
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import custom_fwd, custom_bwd

from einops import rearrange

from gemm_cublas import gemm as gemm_cb, gemm_add_ as gemm_add_cb_
# from gemm_cublas.interface import gemm_tuned as gemm_cb, gemm_add_tuned_ as gemm_add_cb_

from quack import gemm, gemm_swiglu, gemm_dswiglu  # TODO: implement these


class MLPSwiGLUFunc(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, x, weight1, weight2, fuse_grad_accum=False):
        """
        x: (..., in_features)
        weight1: (2 * intermediate_features, in_features)
        weight2: (out_features, intermediate_features)
        out: (..., out_features)
        Note that we do swiglu on the even and odd indices of the intermediate output,
        i.e. silu(y[..., ::2]) * y[..., 1::2].
        This is different from the usual swiglu implementation that does: y1, y2 = y.chunk(2, dim=-1); silu(y1) * y2
        """
        needs_weight1_grad = weight1.requires_grad
        needs_weight2_grad = weight2.requires_grad
        needs_input_grad = x.requires_grad
        ctx.weight1_dtype = weight1.dtype
        ctx.weight2_dtype = weight2.dtype
        autocast_dtype = torch.get_autocast_dtype("cuda")
        if torch.is_autocast_enabled():
            x = x.to(dtype=autocast_dtype)
        weight1_og = weight1
        weight2_og = weight2
        if torch.is_autocast_enabled():
            weight1 = weight1.to(dtype=autocast_dtype)
            weight2 = weight2.to(dtype=autocast_dtype)
        batch_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        # don't need preact if not computing gradient
        store_preact = needs_input_grad or needs_weight1_grad or needs_weight2_grad
        # (batch, inter_dim) & (batch, 2 * inter_dim)
        y, preact = gemm_swiglu(x, weight1.T, store_preact=store_preact)
        # out = F.linear(y, weight2)
        out = gemm(y, weight2.T)
        if not needs_input_grad:
            weight1, weight1_og = None, None
        if not needs_weight1_grad:
            x = None
        if not needs_input_grad and not needs_weight1_grad and not needs_weight2_grad:
            weight2, weight2_og = None, None
            preact = None
        ctx.save_for_backward(
            x,
            preact,
            weight1,
            weight2,
            *((weight1_og, weight2_og) if fuse_grad_accum else (None, None)),
        )
        ctx.fuse_grad_accum = fuse_grad_accum
        return out.reshape(*batch_shape, out.shape[-1])

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, dout):
        """
        dout: (..., out_features)
        """
        if not torch.compiler.is_dynamo_compiling():
            assert dout.stride(-1) == 1
        # weight1_og and weight2_og are None if not ctx.fused_grad_accum
        x, preact, weight1, weight2, weight1_og, weight2_og = ctx.saved_tensors
        batch_shape = dout.shape[:-1]
        dout = dout.reshape(-1, dout.shape[-1])
        if (
            not ctx.needs_input_grad[0]
            and not ctx.needs_weight1_grad[0]
            and not ctx.needs_weight2_grad[0]
        ):
            return (None,) * 4
        assert preact is not None
        # (batch, 2 * inter_dim) and (batch, inter_dim)
        # dpreact, y = gemm_dswiglu(dout, weight2, preact)
        dpreact, y = gemm_dswiglu(dout, weight2, preact, sm_carveout=16)
        if ctx.needs_input_grad[2]:
            # fuse_grad_accum is not compatible with torch.compile
            if not ctx.fuse_grad_accum or weight2_og.grad is None or torch.compiler.is_compiling():
                dweight2 = gemm_cb(dout.T, y, out_dtype=ctx.weight2_dtype)
                # dweight2 = gemm_cb(dout.T, y, out_dtype=ctx.weight2_dtype, sm_carveout=16)
            else:
                # print("Using fuse grad accum in MLP 2", dout.shape, y.shape, weight2_og.grad.shape)
                gemm_add_cb_(dout.T, y, weight2_og.grad)
                # gemm_add_cb_(dout.T, y, weight2_og.grad, sm_carveout=16)
                dweight2 = weight2_og.grad
                weight2_og.grad = (
                    None  # So that pytorch doesn't add dweight to weight2_og.grad again
                )
        else:
            dweight2 = None
        if ctx.needs_input_grad[0]:
            dx = dpreact @ weight1  # (batch, in_features)
            # dx = gemm(dpreact, weight1)  # (batch, in_features)
            dx = dx.reshape(*batch_shape, dx.shape[-1])
        else:
            dx = None
        if ctx.needs_input_grad[1]:
            # fuse_grad_accum is not compatible with torch.compile
            if not ctx.fuse_grad_accum or weight1_og.grad is None or torch.compiler.is_compiling():
                dweight1 = gemm_cb(dpreact.T, x, out_dtype=ctx.weight1_dtype)
            else:
                # print("Using fuse grad accum in MLP 1", dpreact.shape, x.shape, weight1_og.grad.shape)
                gemm_add_cb_(dpreact.T, x, weight1_og.grad)
                dweight1 = weight1_og.grad
                weight1_og.grad = (
                    None  # So that pytorch doesn't add dweight to weight1_og.grad again
                )
        else:
            dweight1 = None
        return dx, dweight1, dweight2, None


def mlp_swiglu_func(x, weight1, weight2, fuse_grad_accum=False):
    return MLPSwiGLUFunc.apply(x, weight1, weight2, fuse_grad_accum)


class MLPSwiGLU(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        bias1=False,
        bias2=False,
        multiple_of=128,
        device=None,
        dtype=None,
        fuse_grad_accum: bool = False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias1, **factory_kwargs)
        self.fc1.weight._muon_reshape_functions = (
            lambda w: rearrange(w, "(d two) e -> two d e", two=2),
            lambda w: rearrange(w, "two d e -> (d two) e"),
        )
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2, **factory_kwargs)
        self.fuse_grad_accum = fuse_grad_accum

    def forward(self, input: Tensor) -> Tensor:
        if (
            self.fc1.bias is None
            and self.fc2.bias is None
            and input.is_cuda
            and input.stride(-1) == 1
            and self.fc1.in_features % 8 == 0
            and self.fc1.out_features % 16 == 0
            and self.fc2.out_features % 8 == 0
        ):
            return mlp_swiglu_func(
                input,
                self.fc1.weight,
                self.fc2.weight,
                fuse_grad_accum=self.fuse_grad_accum,
            )
        else:
            y = self.fc1(input)
            return self.fc2(F.silu(y[..., ::2]) * y[..., 1::2])


class MLPSwiGLURef(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        bias1=False,
        bias2=False,
        multiple_of=128,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias1, **factory_kwargs)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2, **factory_kwargs)

    def forward(self, input: Tensor) -> Tensor:
        y = self.fc1(input)
        y1, y2 = y.chunk(2, dim=-1)
        return self.fc2(F.silu(y1) * y2)
