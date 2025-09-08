from quack.cute_dsl_utils import ArgumentsBase
from typing import Optional
from cutlass.cute import Tensor


class VarlenArguments(ArgumentsBase):
    mCuSeqlensM: Optional[Tensor] = None
    mTensormaps: Optional[Tensor] = None
    mCuSeqlensK: Optional[Tensor] = None
