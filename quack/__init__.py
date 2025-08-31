__version__ = "0.1.11"

from quack.rmsnorm import rmsnorm
from quack.softmax import softmax
from quack.cross_entropy import cross_entropy

# ruff: noqa
import quack.cute_dsl_utils  # Patch cute.compile to optionally dump SASS

__all__ = [
    "rmsnorm",
    "softmax",
    "cross_entropy",
]
