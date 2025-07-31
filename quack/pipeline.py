# Copyright (c) 2025, Tri Dao.

from cutlass.cutlass_dsl import Int32
from cutlass.pipeline import PipelineState, PipelineUserType


class PipelineStateWAdvance(PipelineState):
    def advance_iters(self, num_iterations: Int32):
        self._count += Int32(num_iterations)
        new_index = self._index + Int32(num_iterations)
        # How many times did we cross the stages boundary
        num_crossings = new_index // self.stages
        self._phase ^= num_crossings
        self._index = new_index % self.stages

    # This can be overridden by derived classes
    def __new_from_mlir_values__(self, values):
        return PipelineStateWAdvance(
            self.stages, Int32(values[0]), Int32(values[1]), Int32(values[2])
        )


def make_pipeline_state(type: PipelineUserType, stages: int):
    """
    Creates a pipeline state. Producers are assumed to start with an empty buffer and have a flipped phase bit of 1.
    """
    if type is PipelineUserType.Producer:
        return PipelineStateWAdvance(
            stages,
            Int32(0),
            Int32(0),
            Int32(1),
        )
    elif type is PipelineUserType.Consumer:
        return PipelineStateWAdvance(
            stages,
            Int32(0),
            Int32(0),
            Int32(0),
        )
    else:
        assert False, "Error: invalid PipelineUserType specified for make_pipeline_state."
