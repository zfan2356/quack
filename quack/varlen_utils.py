# Copyright (c) 2025, Tri Dao.

from typing import Optional, List
from dataclasses import dataclass

import cutlass.cute as cute
import cutlass
from quack.cute_dsl_utils import ArgumentsBase


# Grouping arguments together that should be passed to __call__
@dataclass
class VarlenArguments(ArgumentsBase):
    mCuSeqlensMTensor: Optional[cute.Tensor] = None
    mCuSeqlensMList: Optional[List[cutlass.Int32]] = None
    mCuSeqlen: cutlass.Constexpr[cutlass.Int32] = cutlass.const_expr(0)
    mTensormaps: Optional[cute.Tensor] = None

    # def __post_init__(self):
    #     varlen_m = (self.mCuSeqlensMTensor is not None) or (self.mCuSeqlensMList is not None)
    #     assert all(x is None for x in [varlen_m, self.mTensormaps]) or all(
    #         x is not None for x in [varlen_m, self.mTensormaps]
    #     ), "All two fields (mCuSeqlensM, mTensormaps) must be either all None or all not None"
