# Copyright (c) 2025, Tri Dao
from typing import Optional, Tuple, Literal
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor

from quack.gemm_config import GemmConfig, get_all_configs

from quack.autotuner import autotune, AutotuneConfig
from quack.dense_gemm_sm90 import gemm_sm90
from quack.gemm_act_sm90 import gemm_act_sm90
from quack.gemm_dact_sm90 import gemm_dact_sm90


# Dictionary mapping activation names to PyTorch functions
act_to_pytorch_fn_map = {
    None: lambda x: x,
    "relu": F.relu,
    "relu_sq": lambda x: F.relu(x).square(),
    "gelu_tanh_approx": partial(F.gelu, approximate="tanh"),
}


# Dictionary mapping gated activation names to their forward functions
# Each function takes (gate, up) and returns postact
gated_to_pytorch_fn_map = {
    "swiglu": lambda gate, up: F.silu(gate) * up,
    "swiglu_oai": lambda gate, up: gate * torch.sigmoid(1.702 * gate) * (up + 1),
    "reglu": lambda gate, up: F.relu(gate) * up,
    "geglu": lambda gate, up: F.gelu(gate, approximate="tanh") * up,
    "glu": lambda gate, up: torch.sigmoid(gate) * up,
}


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_configs()],
    key=["dynamic_scheduler"],
)
def gemm_tuned(
    A: Tensor,  # (M, K)
    B: Tensor,  # (K, N)
    out: Tensor,  # (M, N) - required output tensor
    C: Optional[Tensor] = None,  # (M, N)
    alpha: float | Tensor = 1.0,  # (1,)
    beta: float | Tensor = 1.0,  # (1,)
    dynamic_scheduler: bool = False,
    config: Optional[GemmConfig] = None,
) -> None:
    if config is None:
        config = GemmConfig(tile_m=128, tile_n=192, cluster_m=2, cluster_n=1, pingpong=True)
    A, B = A.unsqueeze(0), B.mT.unsqueeze(0)  # (1, M, K), (1, N, K)
    if C is not None:
        C = C.unsqueeze(0)  # (1, M, N)
    assert out.shape == (
        A.shape[1],
        B.shape[1],
    ), f"out shape mismatch: {out.shape} vs {(A.shape[1], B.shape[1])}"
    out = out.unsqueeze(0)
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device) if dynamic_scheduler else None
    )
    gemm_sm90(
        A if not config.swap_ab else B,
        B if not config.swap_ab else A,
        out if not config.swap_ab else out.mT,
        (C if not config.swap_ab else C.mT) if C is not None else None,
        tile_count_semaphore,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        config.pingpong,
        alpha=alpha,
        beta=beta,
    )


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_configs()],
    key=["activation"],
)
def gemm_act_tuned(
    A: Tensor,  # (M, K)
    B: Tensor,  # (K, N)
    preact_out: Optional[Tensor],  # (M, N) - None if not storing preact
    postact_out: Tensor,  # (M, N)
    C: Optional[Tensor] = None,  # (M, N)
    activation: Literal[None, "relu", "relu_sq", "gelu_tanh_approx"] = None,
    config: Optional[GemmConfig] = None,
) -> None:
    if config is None:
        config = GemmConfig(tile_m=128, tile_n=192, cluster_m=2, cluster_n=1, pingpong=True)
    A, B = A.unsqueeze(0), B.mT.unsqueeze(0)  # (1, M, K), (1, N, K)
    if C is not None:
        C = C.unsqueeze(0)  # (1, M, N)
    if preact_out is not None:
        assert preact_out.shape == (A.shape[1], B.shape[1])
        D = preact_out.unsqueeze(0)
    else:
        D = None
    assert postact_out.shape == (A.shape[1], B.shape[1])
    PostAct = postact_out.unsqueeze(0)
    gemm_act_sm90(
        A if not config.swap_ab else B,
        B if not config.swap_ab else A,
        (D if not config.swap_ab else D.mT) if D is not None else None,
        (C if not config.swap_ab else C.mT) if C is not None else None,
        PostAct if not config.swap_ab else PostAct.mT,
        activation,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        config.pingpong,
    )


@autotune(
    configs=[AutotuneConfig(config=c) for c in get_all_configs()],
    key=["activation", "dynamic_scheduler"],
)
def gemm_dact_tuned(
    A: Tensor,  # (M, K)
    B: Tensor,  # (K, N)
    PreAct: Tensor,  # (M, N)
    dx_out: Tensor,  # (M, N)
    postact_out: Tensor,  # (M, N)
    activation: Literal[None, "relu", "relu_sq", "gelu_tanh_approx"] = None,
    dynamic_scheduler: bool = True,
    config: Optional[GemmConfig] = None,
) -> None:
    if config is None:
        config = GemmConfig(tile_m=128, tile_n=192, cluster_m=2, cluster_n=1, pingpong=True)
    A, B = A.unsqueeze(0), B.mT.unsqueeze(0)  # (1, M, K), (1, N, K)
    PreAct = PreAct.unsqueeze(0)  # (1, M, N)
    assert dx_out.shape == (A.shape[1], B.shape[1])
    D = dx_out.unsqueeze(0)
    assert postact_out.shape == (A.shape[1], B.shape[1])
    PostAct = postact_out.unsqueeze(0)
    tile_count_semaphore = (
        torch.zeros(1, dtype=torch.int32, device=A.device) if dynamic_scheduler else None
    )
    gemm_dact_sm90(
        A if not config.swap_ab else B,
        B if not config.swap_ab else A,
        D if not config.swap_ab else D.mT,
        PreAct if not config.swap_ab else PreAct.mT,
        PostAct if not config.swap_ab else PostAct.mT,
        tile_count_semaphore,
        activation,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        config.pingpong,
    )


def gemm(
    A: Tensor,
    B: Tensor,
    out: Optional[Tensor] = None,
    alpha: float | Tensor = 1.0,
    out_dtype: Optional[torch.dtype] = None,
    dynamic_scheduler: bool = False,
    tuned: bool = True,
) -> Tensor:
    """GEMM with optional output tensor and tuning control."""
    if out is None:
        out_dtype = A.dtype if out_dtype is None else out_dtype
        out = torch.empty((A.shape[0], B.shape[1]), dtype=out_dtype, device=A.device)
    gemm_out(A, B, out, alpha=alpha, dynamic_scheduler=dynamic_scheduler, tuned=tuned)
    return out


@torch.library.custom_op(
    "quack::gemm_out",
    mutates_args=("out",),
    device_types="cuda",
    # Pretend alpha and beta are float to make torch.library happy
    schema="(Tensor A, Tensor B, Tensor(a2!) out, float alpha=1.0, bool dynamic_scheduler=False, bool tuned=True) -> ()",
)
def gemm_out(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    alpha: float | Tensor = 1.0,
    dynamic_scheduler: bool = False,
    tuned: bool = True,
) -> None:
    """GEMM with pre-allocated output tensor."""
    fn = gemm_tuned if tuned else partial(gemm_tuned.fn, config=None)
    fn(A, B, out, C=None, alpha=alpha, dynamic_scheduler=dynamic_scheduler)


def gemm_ref(
    A: Tensor,
    B: Tensor,
    out: Optional[Tensor] = None,
    alpha: float | Tensor = 1.0,
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Reference implementation for GEMM with pre-allocated output."""
    # The out_dtype argument requires torch >= 2.8
    out = torch.mm(A, B, out_dtype=out_dtype, out=out)
    if not isinstance(alpha, float) or alpha != 1.0:
        out = out * alpha
    return out


def gemm_add(
    A: Tensor,
    B: Tensor,
    C: Tensor,
    out: Optional[Tensor] = None,
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
    out_dtype: Optional[torch.dtype] = None,
    dynamic_scheduler: bool = False,
    tuned: bool = True,
) -> Tensor:
    """GEMM with addition and optional output tensor."""
    if out is None:
        out_dtype = A.dtype if out_dtype is None else out_dtype
        out = torch.empty((A.shape[0], B.shape[1]), dtype=out_dtype, device=A.device)
    gemm_add_out(A, B, C, out, alpha, beta, dynamic_scheduler=dynamic_scheduler, tuned=tuned)
    return out


@torch.library.custom_op(
    "quack::gemm_add_out",
    mutates_args=("out",),
    device_types="cuda",
    # Pretend alpha and beta are float to make torch.library happy
    schema="(Tensor A, Tensor B, Tensor C, Tensor(a3!) out, float alpha=1.0, float beta=1.0, bool dynamic_scheduler=False, bool tuned=True) -> ()",
)
def gemm_add_out(
    A: Tensor,
    B: Tensor,
    C: Tensor,
    out: Tensor,
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
    dynamic_scheduler: bool = False,
    tuned: bool = True,
) -> None:
    """GEMM with addition and pre-allocated output tensor."""
    fn = gemm_tuned if tuned else partial(gemm_tuned.fn, config=None)
    fn(A, B, out, C, alpha=alpha, beta=beta, dynamic_scheduler=dynamic_scheduler)


def gemm_add_ref(
    A: Tensor,
    B: Tensor,
    C: Tensor,
    out: Optional[Tensor] = None,
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
    out_dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Reference implementation for GEMM with addition and pre-allocated output."""
    if isinstance(alpha, float) and isinstance(beta, float):
        return torch.addmm(C, A, B, out_dtype=out_dtype, alpha=alpha, beta=beta, out=out)
    else:
        out_dtype = (
            out.dtype if out is not None else (out_dtype if out_dtype is not None else A.dtype)
        )
        result = (alpha * (A @ B) + beta * C).to(out_dtype)
        if out is not None:
            out.copy_(result)
        return result


@torch.library.custom_op(
    "quack::gemm_add_inplace",
    mutates_args=("out",),
    device_types="cuda",
    schema="(Tensor A, Tensor B, Tensor(a2!) out, float alpha=1.0, float beta=1.0, bool dynamic_scheduler=False, bool tuned=True) -> ()",
)
def gemm_add_inplace(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
    dynamic_scheduler: bool = False,
    tuned: bool = True,
) -> None:
    """In-place GEMM with addition: out = alpha * A @ B + beta * out.
    Args:
        A: (M, K) input tensor
        B: (K, N) input tensor
        out: (M, N) tensor to accumulate into (modified in-place)
        alpha: Scalar multiplier for A @ B
        beta: Scalar multiplier for out
        dynamic_scheduler: Whether to use dynamic scheduler
        tuned: Whether to use autotuned configuration
    """
    fn = gemm_tuned if tuned else partial(gemm_tuned.fn, config=None)
    # Use C as both input bias and output
    fn(A, B, out, out, alpha=alpha, beta=beta, dynamic_scheduler=dynamic_scheduler)


def gemm_act(
    A: Tensor,
    B: Tensor,
    C: Optional[Tensor] = None,
    activation: Literal[None, "relu", "relu_sq", "gelu_tanh_approx"] = None,
    preact_out: Optional[Tensor] = None,
    postact_out: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
    store_preact: bool = True,
    tuned: bool = True,
) -> Tuple[Optional[Tensor], Tensor]:
    """GEMM with activation and optional output tensors."""
    out_dtype = A.dtype if out_dtype is None else out_dtype
    postact_dtype = A.dtype if postact_dtype is None else postact_dtype
    if preact_out is None and store_preact:
        preact_out = torch.empty((A.shape[0], B.shape[1]), dtype=out_dtype, device=A.device)
    if postact_out is None:
        postact_out = torch.empty((A.shape[0], B.shape[1]), dtype=postact_dtype, device=A.device)
    gemm_act_out(A, B, preact_out, postact_out, C, activation, tuned)
    return preact_out, postact_out


@torch.library.custom_op(
    "quack::gemm_act_out",
    mutates_args=("preact_out", "postact_out"),
    device_types="cuda",
    schema="(Tensor A, Tensor B, Tensor(a2!)? preact_out, Tensor(a3!) postact_out, Tensor? C=None, str? activation=None, bool tuned=True) -> ()",
)
def gemm_act_out(
    A: Tensor,
    B: Tensor,
    preact_out: Optional[Tensor],
    postact_out: Tensor,
    C: Optional[Tensor] = None,
    activation: Literal[None, "relu", "relu_sq", "gelu_tanh_approx"] = None,
    tuned: bool = True,
) -> None:
    """GEMM with activation and pre-allocated output tensors."""
    fn = gemm_act_tuned if tuned else partial(gemm_act_tuned.fn, config=None)
    fn(A, B, preact_out, postact_out, C, activation)


def gemm_act_ref(
    A: Tensor,
    B: Tensor,
    C: Optional[Tensor] = None,
    activation: Literal[None, "relu", "relu_sq", "gelu_tanh_approx"] = None,
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
    store_preact: bool = True,
) -> Tuple[Optional[Tensor], Tensor]:
    out_dtype = A.dtype if out_dtype is None else out_dtype
    postact_dtype = A.dtype if postact_dtype is None else postact_dtype
    out = torch.mm(A, B) if C is None else C + torch.mm(A, B)
    postact = act_to_pytorch_fn_map[activation](out).to(postact_dtype)
    return out.to(out_dtype) if store_preact else None, postact


def gemm_dact(
    A: Tensor,
    B: Tensor,
    PreAct: Tensor,
    activation: Literal[None, "relu", "relu_sq", "gelu_tanh_approx"] = None,
    dx_out: Optional[Tensor] = None,
    postact_out: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
    dynamic_scheduler: bool = True,
    tuned: bool = True,
) -> Tuple[Tensor, Tensor]:
    """GEMM with activation gradient and optional output tensors."""
    out_dtype = A.dtype if out_dtype is None else out_dtype
    postact_dtype = PreAct.dtype if postact_dtype is None else postact_dtype
    if dx_out is None:
        dx_out = torch.empty((A.shape[0], B.shape[1]), dtype=out_dtype, device=A.device)
    if postact_out is None:
        postact_out = torch.empty((A.shape[0], B.shape[1]), dtype=postact_dtype, device=A.device)
    gemm_dact_out(A, B, PreAct, dx_out, postact_out, activation, dynamic_scheduler, tuned)
    return dx_out, postact_out


@torch.library.custom_op(
    "quack::gemm_dact_out",
    mutates_args=("dx_out", "postact_out"),
    device_types="cuda",
    schema="(Tensor A, Tensor B, Tensor PreAct, Tensor(a3!) dx_out, Tensor(a4!) postact_out, str? activation=None, bool dynamic_scheduler=True, bool tuned=True) -> ()",
)
def gemm_dact_out(
    A: Tensor,
    B: Tensor,
    PreAct: Tensor,
    dx_out: Tensor,
    postact_out: Tensor,
    activation: Literal[None, "relu", "relu_sq", "gelu_tanh_approx"] = None,
    dynamic_scheduler: bool = True,
    tuned: bool = True,
) -> None:
    """GEMM with activation gradient and pre-allocated output tensors."""
    fn = gemm_dact_tuned if tuned else partial(gemm_dact_tuned.fn, config=None)
    fn(A, B, PreAct, dx_out, postact_out, activation, dynamic_scheduler)


def gemm_dact_ref(
    A: Tensor,
    B: Tensor,
    PreAct: Tensor,
    activation: Literal[None, "relu", "relu_sq", "gelu_tanh_approx"] = None,
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor]:
    """Reference implementation for GEMM with activation gradient."""
    out_dtype = A.dtype if out_dtype is None else out_dtype
    postact_dtype = PreAct.dtype if postact_dtype is None else postact_dtype
    dout = torch.mm(A, B).to(out_dtype)
    postact = act_to_pytorch_fn_map[activation](PreAct)
    # Compute gradient using autograd
    if activation is None:
        dx = dout
    else:
        PreAct_requires_grad = PreAct.requires_grad
        PreAct.requires_grad_(True)
        postact_for_grad = act_to_pytorch_fn_map[activation](PreAct)
        dx = torch.autograd.grad(postact_for_grad, PreAct, dout, create_graph=False)[0]
        PreAct.requires_grad_(PreAct_requires_grad)
    return dx.to(out_dtype), postact.to(postact_dtype)


def gemm_gated_ref(
    A: Tensor,
    B: Tensor,
    C: Optional[Tensor] = None,
    activation: Literal["glu", "swiglu", "swiglu_oai", "reglu", "geglu"] = "swiglu",
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
    store_preact: bool = True,
) -> Tuple[Optional[Tensor], Tensor]:
    """Reference implementation for GEMM with gated activation forward.

    Args:
        A: (M, K) - input tensor
        B: (K, 2*N) - weight tensor with gate and up projections
        C: (M, 2*N) - optional bias tensor
        activation: Type of gated activation
        out_dtype: Output dtype for preact
        postact_dtype: Output dtype for postact
        store_preact: Whether to return the pre-activation

    Returns:
        (preact, postact) where:
        - preact: (M, 2*N) pre-activation (if store_preact=True, else None)
        - postact: (M, N) post-activation output
    """
    out_dtype = A.dtype if out_dtype is None else out_dtype
    postact_dtype = A.dtype if postact_dtype is None else postact_dtype
    preact = torch.mm(A, B) if C is None else C + torch.mm(A, B)
    # Split preact into gate and up projections
    gate = preact[..., ::2]  # (M, N)
    up = preact[..., 1::2]  # (M, N)
    postact = gated_to_pytorch_fn_map[activation](gate, up)
    return preact.to(out_dtype) if store_preact else None, postact.to(postact_dtype)


def gemm_dgated_ref(
    A: Tensor,
    B: Tensor,
    PreAct: Tensor,
    activation: Literal["glu", "swiglu", "swiglu_oai", "reglu", "geglu"],
    out_dtype: Optional[torch.dtype] = None,
    postact_dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor]:
    """Reference implementation for GEMM with gated activation gradient.

    Args:
        A: (M, K) - dout input tensor
        B: (K, N) - weight tensor
        PreAct: (M, 2*N) - pre-activation tensor with gate and up projections interleaved
        activation: Type of gated activation
        out_dtype: Output dtype for dx
        postact_dtype: Output dtype for postact

    Returns:
        (dx, postact) where:
        - dx: (M, 2*N) gradient w.r.t. PreAct
        - postact: (M, N) post-activation output
    """
    out_dtype = A.dtype if out_dtype is None else out_dtype
    postact_dtype = PreAct.dtype if postact_dtype is None else postact_dtype
    dout = torch.mm(A, B).to(out_dtype)
    # Split PreAct into gate and up projections
    gate = PreAct[..., ::2]  # (M, N)
    up = PreAct[..., 1::2]  # (M, N)
    postact = gated_to_pytorch_fn_map[activation](gate, up)
    # Use autograd to compute gradients w.r.t. gate and up
    dgate, dup = torch.autograd.grad(postact, [gate, up], dout, create_graph=False)
    # Interleave gradients back
    dx = torch.stack([dgate, dup], dim=-1).reshape(PreAct.shape)
    return dx.to(out_dtype), postact.to(postact_dtype)


# TODO: this is not quite right, do we need to register gemm_add not gemm_add_out?
# try:
#     from torch._inductor.fx_passes.reinplace import InplaceableOp
#     torch._inductor.fx_passes.reinplace.inplaceable_ops.update({
#         torch.ops.quack.gemm_add_out.default:
#         InplaceableOp(torch.ops.quack.gemm_add_inplace.default, mutated_arg=2)
#     })
# except ImportError:
#     pass
