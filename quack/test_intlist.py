import enum
from typing import Tuple, Type, Callable, Optional, Union
from functools import partial
import math

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Int32, Boolean, const_expr
from quack.cute_dsl_utils import IntList
import torch
from cutlass.cute.runtime import from_dlpack

@cute.jit
def test_intlist(
    seqs: Tuple[Int32, ...],
    t: IntList,
    stream: cuda.CUstream,
):
    kernel(seqs, t).launch(
        grid=[1, 1, 1], 
        block=[1, 1, 1],
        stream=stream,
    )


@cute.kernel
def kernel(seqs: Tuple[Int32, ...], t: IntList):
    tidx, _, _ = cute.arch.thread_idx()
    cute.printf("seqs = {}", max(seqs))

    t1_val = t.ints[tidx]
    cute.printf("t1 = {}", t1_val)


if __name__ == "__main__":
    a = list(range(5))
    ten = IntList(ints=tuple(a))
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    t = cute.compile(test_intlist, (1, 2, 3), ten, current_stream)
    t((1, 2, 3), ten, current_stream)