# Copyright (c) 2025, Tri Dao.
# TODO: we probably dont' need this kernel, just use torch.logsumexp
import torch

import triton
import triton.language as tl


@triton.jit
def _lse_kernel(
    lse_ptr,
    logits_ptr,
    n_rows,
    n_cols,
    logits_row_stride,
    logits_col_stride,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    row_start = tl.program_id(0) * BLOCK_SIZE_M
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    cols = tl.arange(0, BLOCK_SIZE_N)
    logits = tl.load(
        logits_ptr + rows[:, None] * logits_row_stride + cols[None, :] * logits_col_stride,
        mask=(rows[:, None] < n_rows) & (cols[None, :] < n_cols),
        other=-float("inf"),
    ).to(tl.float32)
    m = tl.max(logits, 1)
    lse = tl.log(tl.sum(tl.exp(logits - m[:, None]), 1)) + m
    tl.store(lse_ptr + rows, lse, mask=rows < n_rows)


def logsumexp(logits):
    n_rows, n_cols = logits.shape
    BLOCK_SIZE_M = 32 if logits.stride(1) != 1 else 1
    MAX_BLOCK_SIZE = 64 * 1024
    # BLOCK_SIZE_N = min(triton.next_power_of_2(n_cols), MAX_BLOCK_SIZE // BLOCK_SIZE_M)
    BLOCK_SIZE_N = triton.next_power_of_2(n_cols)
    assert (
        BLOCK_SIZE_M * BLOCK_SIZE_N <= MAX_BLOCK_SIZE
    ), f"Only support max dimension {MAX_BLOCK_SIZE // BLOCK_SIZE_M}"
    num_warps = (
        4
        if BLOCK_SIZE_N < 2048
        else (8 if BLOCK_SIZE_N < 8192 else (16 if BLOCK_SIZE_N < 128 * 1024 else 32))
    )
    lse = torch.empty(n_rows, dtype=torch.float, device=logits.device)
    # Need this, otherwise Triton tries to launch from cuda:0 and we get
    # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
    with torch.cuda.device(logits.device.index):
        _lse_kernel[(triton.cdiv(n_rows, BLOCK_SIZE_M),)](
            lse,
            logits,
            n_rows,
            n_cols,  # shapes
            logits.stride(0),  # strides
            logits.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,  # constants
            BLOCK_SIZE_N=BLOCK_SIZE_N,  # constants
            num_warps=num_warps,
        )
    return lse
