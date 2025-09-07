# Copyright (c) 2025, Tri Dao.

import math
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def tanh(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def relu(x: Float32, *, loc=None, ip=None) -> Float32:
    return cute.arch.fmax(x, Float32(0.0))


@dsl_user_op
def drelu(x: Float32, dout: Float32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    x_pos = cutlass.Boolean(x > 0)
    return dout if x_pos else Float32(0.0), cute.arch.fmax(x, Float32(0.0))


@dsl_user_op
def relu_sq(x: Float32, *, loc=None, ip=None) -> Float32:
    return cute.arch.fmax(x, Float32(0.0)) * x


@dsl_user_op
def drelu_sq(x: Float32, dout: Float32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    """
    ReLU squared backward pass: computes gradient w.r.t. x and recomputes forward
    Given: relu_sq_out = max(x, 0) * x, and dout = grad w.r.t. relu_sq_out
    Returns: (dx, relu_sq_out) where:
    - dx = dout * 2 * x if x > 0, else 0
    - relu_sq_out = max(x, 0) * x
    """
    x_pos = cutlass.Boolean(x > 0)
    relu_sq_out = cute.arch.fmax(x, Float32(0.0)) * x
    # Derivative: d/dx[max(x,0) * x] = 2*x if x > 0, else 0
    dx = (2.0 * dout * x) if x_pos else Float32(0.0)
    return dx, relu_sq_out


@dsl_user_op
def gelu_tanh_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    """
    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            = 0.5 * x * (1 + tanh(x * (0.797885 + 0.0356774 * x * x)))
    """
    sqrt_2_over_pi = math.sqrt(2 / math.pi)  # ~0.797885
    sqrt_2_over_pi_coeff = 0.044715 * sqrt_2_over_pi  # ~0.0356774
    return 0.5 * (x * (1 + tanh(x * (sqrt_2_over_pi + sqrt_2_over_pi_coeff * (x * x)))))


@dsl_user_op
def dgelu_tanh_approx(x: Float32, dout: Float32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    """
    GELU tanh approximation backward pass: computes gradient w.r.t. x and recomputes forward
    Given: gelu_out = 0.5 * x * (1 + tanh(x * (c1 + c2 * x^2))), and dout = grad w.r.t. gelu_out
    Returns: (dx, gelu_out)

    Derivative uses the chain rule:
    d/dx[gelu(x)] = 0.5 * (1 + tanh(z)) + 0.5 * x * sech^2(z) * dz/dx
    where z = x * (c1 + c2 * x^2), dz/dx = c1 + 3 * c2 * x^2
    and sech^2(z) = 1 - tanh^2(z)
    """
    sqrt_2_over_pi = math.sqrt(2 / math.pi)  # c1 ~0.797885
    sqrt_2_over_pi_coeff = 0.044715 * sqrt_2_over_pi  # c2 ~0.0356774
    sqrt_2_over_pi_coeff_3 = 3.0 * sqrt_2_over_pi_coeff  # c3 ~0.01070322

    # Compute z = x * (c1 + c2 * x^2)
    x_sq = x * x
    tanh_z = tanh(x * (sqrt_2_over_pi + sqrt_2_over_pi_coeff * x_sq))
    gelu_out = 0.5 * (x * (1 + tanh_z))

    # Compute gradient
    # sech^2(z) = 1 - tanh^2(z)
    sech2_z = 1 - tanh_z * tanh_z
    # dz/dx = c1 + 3 * c2 * x^2
    dz_dx = sqrt_2_over_pi + sqrt_2_over_pi_coeff_3 * x_sq
    # d/dx[gelu(x)] = 0.5 * (1 + tanh(z)) + 0.5 * x * sech^2(z) * dz/dx
    dgelu = (0.5 + 0.5 * tanh_z) + 0.5 * x * sech2_z * dz_dx

    dx = dout * dgelu
    return dx, gelu_out


@dsl_user_op
def silu(y: Float32, *, loc=None, ip=None) -> Float32:
    """
    silu(y) = y * sigmoid(y) = y * (1 + tanh(y / 2)) / 2 = (0.5 * y) * tanh(0.5 * y) + (0.5 * y)
    This compiles down to 3 SASS instructions: FMUL to get 0.5 * y, MUFU.TANH, and FFMA.
    """
    y_half = 0.5 * y
    return y_half * tanh(y_half) + y_half


@dsl_user_op
def swiglu(x: Float32, y: Float32, *, loc=None, ip=None) -> Float32:
    return x * silu(y)


@dsl_user_op
def dswiglu(
    x: Float32, y: Float32, dout: Float32, *, loc=None, ip=None
) -> Tuple[Float32, Float32, Float32]:
    """
    SwiGLU backward pass: computes gradients w.r.t. x (up projection) and y (gate)
    Given: swiglu_out = x * silu(y), and dout = grad w.r.t. swiglu_out
    Returns: (dx, dy, swiglu_out) where dx = dout * silu(y), dy = dout * x * d_silu(y)

    d_silu(y) = sigmoid(y) * (1 + y * (1 - sigmoid(y)))

    This has been optimized to use fewer instructions (i.e. we expand things out
    to use FFMA instead of FADD and FMUL).
    """
    # Compute sigmoid(y) using tanh: sigmoid(y) = 0.5 * (1 + tanh(0.5 * y))
    y_half = 0.5 * y  # FMUL
    sigmoid_y = 0.5 + 0.5 * tanh(y_half)  # MUFU.TANH, then FFMA
    silu_y = y * sigmoid_y  # FMUL
    silu_y_dout = silu_y * dout  # FMUL
    #   d_silu(y) * dout
    # = sigmoid_y * (1 + y * (1 - sigmoid_y)) * dout
    # = (sigmoid_y + sigmoid_y * y * (1 - sigmoid_y)) * dout
    # = (sigmoid_y + silu_y * (1 - sigmoid_y)) * dout
    # = (sigmoid_y + silu_y - silu_y * sigmoid_y) * dout
    # = (sigmoid_y - silu_y * sigmoid_y) * dout + silu_y * dout
    d_silu_y_dout = (sigmoid_y - silu_y * sigmoid_y) * dout + silu_y_dout  # FFMA, FFMA
    dx = silu_y_dout
    dy = d_silu_y_dout * x  # FMUL
    swiglu_out = x * silu_y  # FMUL
    # Overall it's 1 MUFU.TANH, 5 FMUL, 3 FFMA
    return dx, dy, swiglu_out


@dsl_user_op
def swiglu_oai(x: Float32, y: Float32, alpha: float = 1.072, *, loc=None, ip=None) -> Float32:
    """The swiglu variant used in gpt-oss, which has a scaling factor on y and bias of 1 to x.
    https://github.com/openai/gpt-oss/blob/7be9334950053a888e24887a57dac797a17d6e00/gpt_oss/torch/model.py#L249
    (x + 1) * y * sigmoid(alpha * y)
    Compile down to FMUL, FMUL, TANH, FFMA, FFMA
    """
    # Compute sigmoid(alpha * y) using tanh: sigmoid(z) = 0.5 * (1 + tanh(z/2))
    y_half = 0.5 * y
    silu_y = y_half * tanh(alpha * y_half) + y_half
    return x * silu_y + silu_y


@dsl_user_op
def dswiglu_oai(
    x: Float32, y: Float32, dout: Float32, alpha: float = 1.072, *, loc=None, ip=None
) -> Tuple[Float32, Float32, Float32]:
    """
    Swiglu OAI backward pass: computes gradients w.r.t. x and y
    Given: silu_oai_out = (x + 1) * y * sigmoid(alpha * y), and dout = grad w.r.t. silu_oai_out
    Returns: (dx, dy, silu_oai_out)
    It's the same as dswiglu, dx formula stays the same, for dy we just replace x by x + 1
    """
    # Compute sigmoid(alpha * y) using tanh: sigmoid(z) = 0.5 * (1 + tanh(z/2))
    alpha_y_half = (0.5 * alpha) * y  # FMUL
    sigmoid_alpha_y = 0.5 + 0.5 * tanh(alpha_y_half)  # MUFU.TANH, then FFMA
    silu_y = y * sigmoid_alpha_y  # FMUL
    silu_y_dout = silu_y * dout  # FMUL
    d_silu_y_dout = (sigmoid_alpha_y - silu_y * sigmoid_alpha_y) * dout + silu_y_dout  # FFMA, FFMA
    dx = silu_y_dout
    dy = d_silu_y_dout * x + d_silu_y_dout  # FFMA, instead of multiply by x + 1
    swiglu_out = x * silu_y + silu_y  # FFMA, instead of multiply by x + 1
    # Overall it's 1 MUFU.TANH, 3 FMUL, 5 FFMA
    return dx, dy, swiglu_out
