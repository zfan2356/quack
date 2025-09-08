# Copyright (c) 2025, Tri Dao.

from typing import Optional
from dataclasses import dataclass

import cutlass.cute as cute

from quack.cute_dsl_utils import ArgumentsBase


# Grouping arguments together that should be passed to __call__
@dataclass
class VarlenArguments(ArgumentsBase):
    mCuSeqlensM: Optional[cute.Tensor] = None
    mTensormaps: Optional[cute.Tensor] = None

    def __post_init__(self):
        assert all(x is None for x in [self.mCuSeqlensM, self.mTensormaps]) or all(
            x is not None for x in [self.mCuSeqlensM, self.mTensormaps]
        ), "All two fields (mCuSeqlensM, mTensormaps) must be either all None or all not None"
