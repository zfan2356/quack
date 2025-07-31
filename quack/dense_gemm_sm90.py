# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import enum
from typing import Tuple, Type, Callable, Optional
from functools import partial
import math
import cuda.bindings.driver as cuda

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Int32, const_expr

from quack.tile_scheduler import (
    TileSchedulerArguments,
    StaticTileScheduler,
    ParamsBase,
    RasterOrderOption,
)

# return PipelineStateWAdvance instead of PipelineState
from quack.pipeline import make_pipeline_state

"""
A high-performance batched dense GEMM (C = A * B) example for the NVIDIA Hopper architecture
using CUTE DSL.
- Matrix A is MxKxL, L is batch dimension, A can be row-major("K") or column-major("M")
- Matrix B is NxKxL, L is batch dimension, B can be row-major("N") or column-major("K")
- Matrix C is MxNxL, L is batch dimension, C can be row-major("N") or column-major("M")

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Hopper's WGMMA for matrix multiply-accumulate (MMA) operations
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Supports multi-stage pipeline to overlap computation and memory access

This GEMM works as follows:
1. Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. Perform matrix multiply-accumulate (MMA) operations using WGMMA instruction.
3. Store results from registers (RMEM) to shared memory (SMEM), then to global memory (GMEM) with TMA operations.

Hopper WGMMA instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Perform MMA operation and store the result in Accumulator(register)

To run this example:

.. code-block:: bash

    python examples/hopper/dense_gemm.py                                   \
      --mnkl 8192,8192,8192,1 --tile_shape_mnk 128,256,64                  \
      --cluster_shape_mn 1,1 --a_dtype Float16 --b_dtype Float16           \
      --d_dtype Float16 --acc_dtype Float32                                \
      --a_major k --b_major k --d_major n

The above example command compute batched gemm with M=8192, N=8192, K=8192,
batch_count=1. The Hopper WGMMA tile shape is 128x256x64 and the cluster shape
is (1,1). The input, mma accumulator and output data type are set as fp16, fp32
and fp16, respectively.

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/hopper/dense_gemm.py                               \
      --mnkl 8192,8192,8192,1 --tile_shape_mnk 128,256,64                  \
      --cluster_shape_mn 1,1 --a_dtype Float16 --b_dtype Float16           \
      --d_dtype Float16 --acc_dtype Float32                                \
      --a_major k --b_major k --d_major n

Constraints:
* Supported input data types: fp16, fp8 (e4m3fn, e5m2)
* For fp16 types, A and B must have the same data type
* For fp8 types, A and B can have different types (e4m3fn or e5m2) but both must be 8-bit
* Fp8 types only support k-major layout
* Only fp32 accumulation is supported in this example
* CTA tile shape M must be 64/128
* CTA tile shape N must be 64/128/256
* CTA tile shape K must be 64
* Cluster shape M/N must be positive and power of 2, total cluster size <= 4
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 8, 16 for Float16, and Float8, respectively.
* OOB tiles are not allowed when TMA store is disabled
"""


# /////////////////////////////////////////////////////////////////////////////
#  Helpers to parse args
# /////////////////////////////////////////////////////////////////////////////
def parse_comma_separated_ints(s: str):
    try:
        return tuple([int(x.strip()) for x in s.split(",")])
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid format. Expected comma-separated integers.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example of MxNxKxL GEMM on Hopper.")

    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(4096, 4096, 4096, 1),
        help="mnkl dimensions (comma-separated)",
    )
    parser.add_argument(
        "--tile_shape_mnk",
        type=parse_comma_separated_ints,
        default=(128, 256, 64),
        help="Cta tile shape (comma-separated)",
    )
    parser.add_argument(
        "--cluster_shape_mn",
        type=parse_comma_separated_ints,
        choices=[(1, 1), (2, 1), (1, 2), (2, 2)],
        default=(1, 1),
        help="Cluster shape (comma-separated)",
    )
    parser.add_argument(
        "--a_dtype",
        type=cutlass.dtype,
        default=cutlass.BFloat16,
    )
    parser.add_argument(
        "--b_dtype",
        type=cutlass.dtype,
        default=cutlass.BFloat16,
    )
    parser.add_argument(
        "--d_dtype",
        type=cutlass.dtype,
        default=cutlass.BFloat16,
    )
    parser.add_argument(
        "--acc_dtype",
        type=cutlass.dtype,
        default=cutlass.Float32,
    )
    parser.add_argument("--a_major", choices=["k", "m"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k", "n"], type=str, default="k")
    parser.add_argument("--d_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument("--tolerance", type=float, default=1e-01, help="Tolerance for validation")
    parser.add_argument("--warmup_iterations", type=int, default=0, help="Warmup iterations")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run the kernel",
    )
    parser.add_argument("--persistent", action="store_true", help="Persistent kernel")
    parser.add_argument("--pingpong", action="store_true", help="Pingpong kernel")
    parser.add_argument("--skip_ref_check", action="store_true", help="Skip reference checking")
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )

    args = parser.parse_args()

    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")
    if len(args.tile_shape_mnk) != 3:
        parser.error("--tile_shape_mnk must contain exactly 3 values")
    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    return args


# /////////////////////////////////////////////////////////////////////////////
#  Host setup and device kernel launch
# /////////////////////////////////////////////////////////////////////////////


class NamedBarrierPingpong(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    MmaWG0 = enum.auto()
    MmaWG1 = enum.auto()
    EpiWG0 = enum.auto()
    EpiWG1 = enum.auto()


class HopperWgmmaGemmKernel:
    """
    This class implements batched matrix multiplication (C = A x B) with support for various data types
    and architectural features specific to Hopper GPUs.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param tile_shape_mnk: Shape of the CTA tile (M,N,K)
    :type tile_shape_mnk: Tuple[int, int, int]
    :param cluster_shape_mnk: Cluster dimensions (M,N,K) for parallel processing
    :type cluster_shape_mnk: Tuple[int, int, int]

    :note: Data type requirements:
        - For 16-bit types: A and B must have the same data type
        - For 8-bit types: A and B can have different types (Float8E4M3FN/Float8E5M2) as long as both are 8-bit
        - Float8 types only support k-major layout

    :note: Supported data types:
        - Float16
        - BFloat16
        - Float8E4M3FN/Float8E5M2

    :note: Supported accumulation types:
        - Float32 (for all floating point inputs)

    :note: Constraints:
        - CTA tile M must be 64/128
        - CTA tile N must be 64/128/256
        - CTA tile K must be 64
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 4

    Example:
        >>> gemm = HopperWgmmaGemmKernel(
        ...     acc_dtype=cutlass.Float32,
        ...     tile_shape_mnk=(128, 256, 64),
        ...     cluster_shape_mnk=(1, 1, 1)
        ... )
        >>> gemm(a_tensor, b_tensor, c_tensor, stream)
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        pingpong: bool = False,
        is_persistent: bool = True,
    ):
        """
        Initializes the configuration for a Hopper dense GEMM kernel.

        This configuration includes data types for operands, tile shape, cluster configuration,
        and thread layout.

        :param acc_dtype: Data type for accumulation during computation
        :type acc_dtype: type[cutlass.Numeric]
        :param tile_shape_mnk: Shape of the CTA tile (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param cluster_shape_mnk: Cluster dimensions (M,N,K) for parallel processing
        :type cluster_shape_mnk: Tuple[int, int, int]
        """

        self.acc_dtype = acc_dtype
        self.pingpong = pingpong
        self.is_persistent = is_persistent
        if self.pingpong:
            assert self.is_persistent, "Pingpong gemm requires persistent scheduler"

        self.cluster_shape_mnk = cluster_shape_mnk
        self.tile_shape_mnk = tuple(tile_shape_mnk)
        tile_M, tile_N = tile_shape_mnk[0], tile_shape_mnk[1]
        # check the cta tile shape
        if not self.pingpong:
            if tile_M not in [64, 128, 192, 256, 320]:
                raise ValueError("CTA tile shape M must be 64/128/192/256/320")
            if tile_M in [192, 320]:  # special case
                tile_N_max = 256 if tile_M == 192 else 160
                if not (tile_N % 32 == 0 and tile_N <= tile_N_max):
                    raise ValueError(
                        f"If tile_m == {tile_M}, CTA tile shape N must be divisible by 32 and <= {tile_N_max}"
                    )
            else:
                if not (
                    (tile_N % 16 == 0 and tile_N <= 256) or (tile_N % 32 == 0 and tile_N <= 512)
                ):
                    raise ValueError(
                        "CTA tile shape N must be divisible by 16 and <= 256, or divisible by 32 and <= 512"
                    )
        else:
            if tile_M not in [64, 128, 192]:
                raise ValueError("CTA tile shape M must be 64/128/192 if pingpong")
            tile_N_max = 256 if tile_M == 64 else (208 if tile_M == 128 else 128)
            if not (tile_N % 16 == 0 and tile_N <= tile_N_max):
                raise ValueError(f"CTA tile shape N must be divisible by 16 and <= {tile_N_max}")
        if not self.tile_shape_mnk[2] % 16 == 0:
            raise ValueError("CTA tile shape K must be divisible by 16")

        if not self.pingpong:
            if tile_M in [192, 320]:  # Special case
                atom_layout_m, atom_layout_n = 1, 2
            else:
                atom_layout_m = tile_shape_mnk[0] // 64 if tile_shape_mnk[0] < 256 else 2
                atom_layout_n = 1
            assert atom_layout_m in [1, 2] and atom_layout_n in [1, 2]
        else:
            atom_layout_m, atom_layout_n = 1, 1
        self.atom_layout_mnk = (atom_layout_m, atom_layout_n, 1)

        self.num_mcast_ctas_a = self.cluster_shape_mnk[1]
        self.num_mcast_ctas_b = self.cluster_shape_mnk[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.occupancy = 1
        self.mma_warp_groups = math.prod(self.atom_layout_mnk) * (1 if not self.pingpong else 2)
        if self.pingpong:
            assert self.mma_warp_groups == 2
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = (self.mma_warp_groups + 1) * self.num_threads_per_warp_group
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")
        self.num_mma_threads = (
            self.mma_warp_groups if not self.pingpong else 1
        ) * self.num_threads_per_warp_group
        self.num_epi_threads = (
            self.mma_warp_groups if not self.pingpong else 1
        ) * self.num_threads_per_warp_group
        self.tma_warp_id = self.mma_warp_groups * 4

        regs_per_thread = math.prod(self.tile_shape_mnk[:2]) // self.num_mma_threads
        heavy_register_pressure = regs_per_thread >= 208
        self.num_regs_load = 40 if not heavy_register_pressure else 24
        self.num_regs_mma = 232 if not heavy_register_pressure else 240

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None

        self.shared_storage = None
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        """

        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        self.epi_tile = self._sm90_compute_tile_shape_or_override(
            self.tile_shape_mnk,
            self.atom_layout_mnk,
            self.d_dtype,
        )

        # Compute stage before compute smem layout
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            # epi_smem will reuse smem ab if not persistent.
            self.epi_tile if self.is_persistent else None,
            self.a_dtype,
            self.b_dtype,
            self.d_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.d_dtype,
            self.d_layout,
            self.epi_stage,
        )

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: cute.Tensor,
        max_active_clusters: Int32,
        stream: cuda.CUstream,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes
        - Setup TMA load/store atoms and tensors
        - Compute grid size
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param mA: Input tensor A
        :type mA: cute.Tensor
        :param mB: Input tensor B
        :type mB: cute.Tensor
        :param mD: Output tensor D
        :type mD: cute.Tensor
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        """

        # setup static attributes before smem/grid/tma computation
        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.d_dtype = mD.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(mA)
        self.b_layout = utils.LayoutEnum.from_tensor(mB)
        self.d_layout = utils.LayoutEnum.from_tensor(mD)

        if const_expr(self.a_dtype.width == 16 and self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if const_expr(self.a_dtype.width != self.b_dtype.width):
            raise TypeError(f"Type width mismatch: {self.a_dtype.width} != {self.b_dtype.width}")
        if const_expr(self.a_dtype.width != 16 and self.a_dtype.width != 8):
            raise TypeError("a_dtype should be float16 or float8")

        self._setup_attributes()

        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1] // self.atom_layout_mnk[1]),
        )
        if const_expr(self.atom_layout_mnk[1] > 1):
            # If N dimension is split among 2 WGs, we need to permute the N dimension so
            # that in the epilogue, WG0 and WG1 can write to epi smem of size e.g. (64, 32)
            # containing accumulators that are next to each other in the N dimension.
            # Without permutation WG0 would write to epi smem of size (64, 16) and
            # WG1 would write to a separate epi smem of size (64, 16) that's far away.
            atom_n = self.atom_layout_mnk[1]
            permutation_n = cute.make_ordered_layout(
                (8, self.tile_shape_mnk[1] // atom_n // 8, atom_n), order=(0, 2, 1)
            )
            tiled_mma = cute.make_tiled_mma(
                cute.make_mma_atom(tiled_mma.op),
                self.atom_layout_mnk,
                permutation_mnk=(None, permutation_n, None),
            )

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            mA,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            self.cluster_shape_mnk[1],
        )

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            mB,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.cluster_shape_mnk[0],
        )

        tma_atom_d, tma_tensor_d = self._make_tma_store_atoms_and_tensors(
            mD,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        problem_shape_ntile_mnl = cute.ceil_div(mD.shape[:2], self.tile_shape_mnk[:2]) + (
            mD.shape[2],
        )
        TileScheduler = StaticTileScheduler
        tile_sched_args = TileSchedulerArguments(
            problem_shape_ntile_mnl=problem_shape_ntile_mnl,
            raster_order=RasterOrderOption.Heuristic,
            group_size=8,
            cluster_shape_mnk=self.cluster_shape_mnk,
            is_persistent=self.is_persistent,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid = TileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)

        epi_smem_size = cute.cosize(self.epi_smem_layout_staged) if self.is_persistent else 0

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            sD: cute.struct.Align[
                cute.struct.MemRange[self.d_dtype, epi_smem_size],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_d,
            tma_tensor_d,
            tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
            TileScheduler,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )
        return

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        mD_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
    ):
        """
        GPU device kernel performing the batched GEMM computation.

        :param tma_atom_a: TMA copy atom for A tensor
        :type tma_atom_a: cute.CopyAtom
        :param mA_mkl: Input tensor A
        :type mA_mkl: cute.Tensor
        :param tma_atom_b: TMA copy atom for B tensor
        :type tma_atom_b: cute.CopyAtom
        :param mB_nkl: Input tensor B
        :type mB_nkl: cute.Tensor
        :param tma_atom_d: TMA copy atom for D tensor
        :type tma_atom_d: cute.CopyAtom
        :param mD_mnl: Output tensor D
        :type mD_mnl: cute.Tensor
        :param tiled_mma: Tiled MMA object
        :type tiled_mma: cute.TiledMma
        :param cta_layout_mnk: CTA layout
        :type cta_layout_mnk: cute.Layout
        :param a_smem_layout_staged: Shared memory layout for A
        :type a_smem_layout_staged: cute.ComposedLayout
        :param b_smem_layout_staged: Shared memory layout for B
        :type b_smem_layout_staged: cute.ComposedLayout
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch Tma desc
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_d)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout) + cute.size_in_bytes(
            self.b_dtype, b_smem_layout
        )

        # /////////////////////////////////////////////////////////////////////////////
        #  Alloc and init AB full/empty + ACC full mbar (pipeline)
        # /////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Threads/warps participating in this pipeline
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        # Each warp will constribute to the arrive count with the number of mcast size
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = mcast_size * (self.num_mma_threads // cute.arch.WARP_SIZE)
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mainloop_pipeline_array_ptr.data_ptr(),
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Generate smem tensor A/B
        # ///////////////////////////////////////////////////////////////////////////////
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        if const_expr(not self.is_persistent):
            sD_ptr = cute.recast_ptr(sA.iterator, epi_smem_layout_staged.inner, dtype=self.d_dtype)
            sD = cute.make_tensor(sD_ptr, epi_smem_layout_staged.outer)
        else:
            sD = storage.sD.get_tensor(
                epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
            )

        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        k_tile_cnt = cute.ceil_div(cute.size(mA_mkl.shape[1]), self.tile_shape_mnk[2])

        if warp_idx >= self.tma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_load)
            if warp_idx == self.tma_warp_id:
                # ///////////////////////////////////////////////////////////////////////////////
                # Get mcast mask
                # ///////////////////////////////////////////////////////////////////////////////
                cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)
                a_mcast_mask = cute.make_layout_image_mask(
                    cta_layout_mnk, cluster_coord_mnk, mode=1
                )
                b_mcast_mask = cute.make_layout_image_mask(
                    cta_layout_mnk, cluster_coord_mnk, mode=0
                )
                a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
                b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0
                mainloop_producer_state = make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                tile_scheduler = TileSchedulerCls()
                work_tile = tile_scheduler.initial_work_tile_info()
                while work_tile.is_valid_tile:
                    tile_coord_mnkl = work_tile.tile_idx
                    # ///////////////////////////////////////////////////////////////////////////
                    #  Local_tile partition global tensors
                    # ///////////////////////////////////////////////////////////////////////////
                    # (bM, bK, RestK)
                    gA_mkl = cute.local_tile(
                        mA_mkl, self.tile_shape_mnk, tile_coord_mnkl, proj=(1, None, 1)
                    )
                    # (bN, bK, RestK)
                    gB_nkl = cute.local_tile(
                        mB_nkl, self.tile_shape_mnk, tile_coord_mnkl, proj=(None, 1, 1)
                    )
                    # //////////////////////////////////////////////////////////////////////////
                    #  Partition shared tensor for TMA load A/B
                    # //////////////////////////////////////////////////////////////////////////
                    #  TMA load A partition_S/D
                    a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
                    a_cta_crd = cluster_coord_mnk[1]
                    tAsA, tAgA_mkl = cpasync.tma_partition(
                        tma_atom_a,
                        a_cta_crd,
                        a_cta_layout,
                        cute.group_modes(sA, 0, 2),
                        cute.group_modes(gA_mkl, 0, 2),
                    )
                    # TMA load B partition_S/D
                    b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
                    b_cta_crd = cluster_coord_mnk[0]
                    tBsB, tBgB_nkl = cpasync.tma_partition(
                        tma_atom_b,
                        b_cta_crd,
                        b_cta_layout,
                        cute.group_modes(sB, 0, 2),
                        cute.group_modes(gB_nkl, 0, 2),
                    )
                    # /////////////////////////////////////////////////////////////////////////
                    # TMA load
                    # /////////////////////////////////////////////////////////////////////////
                    for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                        # Wait for A/B buffers to be empty before loading into them
                        # Also sets the transaction barrier for the A/B buffers
                        mainloop_pipeline.producer_acquire(mainloop_producer_state)
                        cute.copy(
                            tma_atom_a,
                            tAgA_mkl[None, k_tile],
                            tAsA[None, mainloop_producer_state.index],
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                mainloop_producer_state
                            ),
                            mcast_mask=a_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_nkl[None, k_tile],
                            tBsB[None, mainloop_producer_state.index],
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                mainloop_producer_state
                            ),
                            mcast_mask=b_mcast_mask,
                        )
                        # Mainloop pipeline's producer commit is a NOP
                        mainloop_pipeline.producer_commit(mainloop_producer_state)
                        mainloop_producer_state.advance()
                    tile_scheduler.prefetch_next_work()
                    tile_scheduler.advance_to_next_work()
                    work_tile = tile_scheduler.get_current_work()
                    # End of persistent scheduler loop
                mainloop_pipeline.producer_tail(mainloop_producer_state)

        if warp_idx < self.tma_warp_id:
            cute.arch.warpgroup_reg_alloc(self.num_regs_mma)
            # //////////////////////////////////////////////////////////////////////////////
            #  Partition global tensor for TiledMMA_A/B/C
            # //////////////////////////////////////////////////////////////////////////////
            tidx, _, _ = cute.arch.thread_idx()
            warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
            if const_expr(self.pingpong):
                tidx = tidx % self.num_threads_per_warp_group
            warp_group_thread_layout = cute.make_layout(
                self.mma_warp_groups if not self.pingpong else 1,
                stride=self.num_threads_per_warp_group,
            )
            thr_mma = tiled_mma.get_slice(
                warp_group_thread_layout(warp_group_idx if not self.pingpong else 0)
            )

            # //////////////////////////////////////////////////////////////////////////////
            #  Make fragments
            # //////////////////////////////////////////////////////////////////////////////
            tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
            tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))

            acc_shape = tiled_mma.partition_shape_C(cute.select(self.tile_shape_mnk, mode=[0, 1]))
            acc = cute.make_fragment(acc_shape, self.acc_dtype)

            if const_expr(self.pingpong):
                if warp_group_idx == 0:
                    # WG0 needs a start signal at the very beginning
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="mma")
                    self.pingpong_barrier_arrive(warp_group_idx=0, stage="epi")

            mainloop_consumer_read_state = make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            tile_scheduler = TileSchedulerCls()
            if const_expr(self.pingpong):
                if warp_idx >= 4:
                    # Advance 2nd Math WG to the next work tile for the startup
                    tile_scheduler.advance_to_next_work()
                    # Advance 2nd Math WG pipeline states to the end of 1st Math WG
                    mainloop_consumer_read_state.advance_iters(k_tile_cnt)
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                tile_coord_mnkl = work_tile.tile_idx
                # /////////////////////////////////////////////////////////////////////////////
                #  Prologue MMAs
                # /////////////////////////////////////////////////////////////////////////////
                k_pipe_mmas = 1
                mainloop_consumer_release_state = mainloop_consumer_read_state.clone()
                num_prologue_mma = min(k_pipe_mmas, k_tile_cnt)
                if const_expr(self.pingpong):
                    self.pingpong_barrier_sync(warp_group_idx, stage="mma")
                peek_ab_full_status = cutlass.Boolean(1)
                if 0 < k_tile_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_read_state
                    )
                tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
                num_k_blocks = cute.size(tCrA, mode=[2])
                for k_tile in cutlass.range(num_prologue_mma):
                    # Wait for A/B buffer to be ready
                    mainloop_pipeline.consumer_wait(
                        mainloop_consumer_read_state, peek_ab_full_status
                    )
                    warpgroup.fence()
                    for k_blk_idx in cutlass.range(num_k_blocks, unroll_full=True):
                        k_blk_coord = (None, None, k_blk_idx, mainloop_consumer_read_state.index)
                        cute.gemm(tiled_mma, acc, tCrA[k_blk_coord], tCrB[k_blk_coord], acc)
                        tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                    warpgroup.commit_group()
                    mainloop_consumer_read_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if k_tile + 1 < k_tile_cnt:
                        peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                            mainloop_consumer_read_state
                        )

                # /////////////////////////////////////////////////////////////////////////////
                #  MAINLOOP
                # /////////////////////////////////////////////////////////////////////////////
                for k_tile in cutlass.range(num_prologue_mma, k_tile_cnt, unroll=1):
                    # Wait for TMA copies to complete
                    mainloop_pipeline.consumer_wait(
                        mainloop_consumer_read_state, peek_ab_full_status
                    )
                    # WGMMA
                    warpgroup.fence()
                    for k_blk_idx in cutlass.range(num_k_blocks, unroll_full=True):
                        k_blk_coord = (None, None, k_blk_idx, mainloop_consumer_read_state.index)
                        cute.gemm(tiled_mma, acc, tCrA[k_blk_coord], tCrB[k_blk_coord], acc)
                    warpgroup.commit_group()
                    # Wait on the wgmma barrier for previous k_pipe_mmas wgmmas to complete
                    warpgroup.wait_group(k_pipe_mmas)
                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    mainloop_consumer_read_state.advance()
                    mainloop_consumer_release_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if k_tile + 1 < k_tile_cnt:
                        peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                            mainloop_consumer_read_state
                        )
                if const_expr(self.pingpong):
                    # Cue for next WG's MMA to start
                    self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
                warpgroup.wait_group(0)
                for k_tile in cutlass.range(k_pipe_mmas, unroll=1):
                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    mainloop_consumer_release_state.advance()
                if const_expr(self.pingpong):
                    # Update starting mainloop pipeline state for the next tile
                    mainloop_consumer_read_state.advance_iters(k_tile_cnt)

                # /////////////////////////////////////////////////////////////////////////////
                #  EPILOGUE
                # /////////////////////////////////////////////////////////////////////////////
                if const_expr(self.pingpong):
                    self.pingpong_barrier_sync(warp_group_idx, "epi")
                # Wait for all warp groups in the thread block to finish, because smem for tensor A in
                # the mainloop is reused in the epilogue if not persistent.
                if const_expr(not self.is_persistent):
                    cute.arch.barrier(barrier_id=1, number_of_threads=self.num_mma_threads)

                copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                    self.d_layout,
                    elem_ty_d=self.d_dtype,
                    elem_ty_acc=self.acc_dtype,
                )
                copy_atom_C = cute.make_copy_atom(
                    warp.StMatrix8x8x16bOp(self.d_layout.is_m_major_c(), 4),
                    self.d_dtype,
                )
                tiled_copy_C_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
                tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_atom)
                # (R2S, R2S_M, R2S_N, PIPE_D)
                tRS_sD = tiled_copy_r2s.get_slice(tidx).partition_D(sD)
                # (R2S, R2S_M, R2S_N)
                tRS_rAcc = tiled_copy_r2s.retile(acc)

                # (bM, bN)
                gD_mnl = cute.local_tile(
                    mD_mnl, self.tile_shape_mnk, tile_coord_mnkl, proj=(1, 1, None)
                )
                tcgc_for_tma_partition = cute.zipped_divide(gD_mnl, self.epi_tile)
                bSG_sD, bSG_gD = cpasync.tma_partition(
                    tma_atom_d,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sD, 0, 2),
                    tcgc_for_tma_partition,
                )

                epi_tile_num = const_expr(cute.size(tcgc_for_tma_partition, mode=[1]))
                epi_tile_shape = tcgc_for_tma_partition.shape[1]
                num_prev_subtiles = tile_scheduler.num_tiles_executed * epi_tile_num
                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    # Copy from acc to D registers
                    tRS_rD = cute.make_fragment_like(tRS_sD[None, None, None, 0], self.acc_dtype)
                    for epi_v in cutlass.range_constexpr(cute.size(tRS_rD)):
                        tRS_rD[epi_v] = tRS_rAcc[epi_idx * cute.size(tRS_rD) + epi_v]
                    # Type conversion
                    tRS_rD_out = cute.make_fragment_like(tRS_rD, self.d_dtype)
                    tRS_rD_out.store(tRS_rD.load().to(self.d_dtype))
                    # Copy from D registers to shared memory
                    epi_buffer = (num_prev_subtiles + epi_idx) % cute.size(tRS_sD, mode=[3])
                    cute.copy(tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)])
                    # Fence and barrier to make sure shared memory store is visible to TMA store
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
                    )
                    cute.arch.barrier(barrier_id=1, number_of_threads=self.num_epi_threads)
                    # Get the global memory coordinate for the current epi tile.
                    epi_tile_layout = cute.make_layout(
                        epi_tile_shape, stride=(epi_tile_shape[1], 1)
                    )
                    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                    # Copy from shared memory to global memory
                    if (not self.pingpong and warp_idx == 0) or (
                        self.pingpong and (warp_idx == 0 or warp_idx == 4)
                    ):
                        cute.copy(tma_atom_d, bSG_sD[None, epi_buffer], bSG_gD[None, gmem_coord])
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(self.epi_stage - 1, read=True)
                    cute.arch.barrier(barrier_id=1, number_of_threads=self.num_epi_threads)

                if const_expr(self.pingpong):
                    # With pingpong, 2 WGs write two different output tiles to the same smem,
                    # so we have to make sure the smem content is done reading before signalling
                    # the next WG's epilogue.
                    if warp_idx == 0 or warp_idx == 4:
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                    self.pingpong_barrier_arrive(1 - warp_group_idx, stage="epi")

                tile_scheduler.advance_to_next_work(
                    advance_count=1 if not self.pingpong else self.mma_warp_groups
                )
                work_tile = tile_scheduler.get_current_work()
                # End of persistent scheduler loop

            if const_expr(not self.pingpong):
                if warp_idx == 0:
                    cute.arch.cp_async_bulk_wait_group(0, read=True)

    def pingpong_barrier_sync(self, warp_group_idx: Int32, stage: str):
        assert stage in ["mma", "epi"]
        barrier = NamedBarrierPingpong.MmaWG0 if stage == "mma" else NamedBarrierPingpong.EpiWG0
        cute.arch.barrier(
            barrier_id=int(barrier) + warp_group_idx,
            number_of_threads=2 * self.num_threads_per_warp_group,
        )

    def pingpong_barrier_arrive(self, warp_group_idx: Int32, stage: str):
        assert stage in ["mma", "epi"]
        barrier = NamedBarrierPingpong.MmaWG0 if stage == "mma" else NamedBarrierPingpong.EpiWG0
        cute.arch.barrier_arrive(
            barrier_id=int(barrier) + warp_group_idx,
            number_of_threads=2 * self.num_threads_per_warp_group,
        )

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: Tuple[int, int, int],
        epi_tile: Optional[Tuple[int, int]],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: Tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (A/B operand stages, epilogue stages)
        :rtype: Tuple[int, int]
        """

        epi_stage = 4
        # epi_smem will reuse smem ab if not persistent.
        epi_bytes = 0 if epi_tile is None else cute.size(epi_tile) * d_dtype.width // 8 * epi_stage

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8 + cute.size(b_shape) * b_dtype.width // 8
        )
        mbar_helpers_bytes = 1024

        ab_stage = (
            (smem_capacity - occupancy * 1024) // occupancy - mbar_helpers_bytes - epi_bytes
        ) // ab_bytes_per_stage
        return ab_stage, epi_stage

    @staticmethod
    def _sm90_compute_tile_shape_or_override(
        tile_shape_mnk: Tuple[int, int, int],
        atom_layout_mnk: Tuple[int, int, int],
        element_type: Type[cutlass.Numeric],
        epi_tile_override: Tuple[int, int] | None = None,
    ) -> Tuple[int, int]:
        """Compute the epilogue tile shape or use override if provided.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param element_type: Data type of elements
        :type element_type: type[cutlass.Numeric]
        :param is_cooperative: Whether to use cooperative approach
        :type is_cooperative: bool
        :param epi_tile_override: Optional override for epilogue tile shape
        :type epi_tile_override: Tuple[int, int] or None

        :return: Computed epilogue tile shape
        :rtype: Tuple[int, int]
        """
        if epi_tile_override is not None:
            return epi_tile_override
        if tile_shape_mnk[0] % 128 == 0 and atom_layout_mnk[0] > 1:
            tile_m = math.gcd(128, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = math.gcd(32, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)
        else:
            # In the case of tile shape 128 x N but atom_layout 1 x 2, we need to set
            # epi_tile_m = 64. If epi_tile_m = 128, the epilogue would iterate along the
            # M dimension first, then move to the N dimension. But the accumulator in registers
            # iterate along the N dimension first, then move to the M dimension.
            # We could change the epilogue to accommodate this,
            # but it's easier to just set epi_tile_m = 64.
            n_perf = 64 if element_type.width == 8 else 32
            tile_m = math.gcd(64, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = math.gcd(n_perf, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: Tuple[int, int, int],
        epi_tile: Tuple[int, int],
        a_dtype: Type[cutlass.Numeric],
        a_layout: utils.LayoutEnum,
        b_dtype: Type[cutlass.Numeric],
        b_layout: utils.LayoutEnum,
        ab_stage: int,
        d_dtype: Type[cutlass.Numeric],
        d_layout: utils.LayoutEnum,
        epi_stage: int,
    ) -> Tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:
        """Create shared memory layouts for A, B, and C tensors.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param a_dtype: Data type for matrix A
        :type a_dtype: type[cutlass.Numeric]
        :param a_layout: Layout enum for matrix A
        :type a_layout: utils.LayoutEnum
        :param b_dtype: Data type for matrix B
        :type b_dtype: type[cutlass.Numeric]
        :param b_layout: Layout enum for matrix B
        :type b_layout: utils.LayoutEnum
        :param ab_stage: Number of stages for A/B tensors
        :type ab_stage: int
        :param d_dtype: Data type for output matrix C
        :type d_dtype: type[cutlass.Numeric]
        :param d_layout: Layout enum for the output matrix C
        :type d_layout: utils.LayoutEnum
        :param epi_stage: Number of epilogue stages
        :type epi_stage: int

        :return: Tuple of shared memory layouts for A, B, and C
        :rtype: Tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]
        """
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))

        a_is_k_major = a_layout.sm90_mma_major_mode() == warpgroup.OperandMajorMode.K
        b_is_k_major = b_layout.sm90_mma_major_mode() == warpgroup.OperandMajorMode.K
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]
        a_smem_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                a_layout,
                a_dtype,
                a_major_mode_size,
            ),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))

        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                b_layout,
                b_dtype,
                b_major_mode_size,
            ),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

        d_smem_shape = epi_tile
        d_major_mode_size = epi_tile[1] if d_layout.is_n_major_c() else epi_tile[0]
        d_smem_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                d_layout,
                d_dtype,
                d_major_mode_size,
            ),
            d_dtype,
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            d_smem_layout_atom,
            cute.append(d_smem_shape, epi_stage),
            order=(1, 0, 2) if d_layout.is_m_major_c() else (0, 1, 2),
        )

        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_d: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: Tuple[int, int],
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for C tensor storage.

        :param tensor_d: Output tensor D
        :type tensor_d: cute.Tensor
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]

        :return: TMA atom and tensor for C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        c_cta_v_layout = cute.composition(cute.make_identity_layout(tensor_d.shape), epi_tile)
        tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            tensor_d,
            epi_smem_layout,
            c_cta_v_layout,
        )

        return tma_atom_d, tma_tensor_d

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: Tuple[int, int],
        mcast_dim: int,
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors.

        :param tensor: Input tensor (A or B)
        :type tensor: cute.Tensor
        :param smem_layout_staged: Shared memory layout for the tensor
        :type smem_layout_staged: cute.ComposedLayout
        :param smem_tile: Shared memory tile shape
        :type smem_tile: Tuple[int, int]
        :param mcast_dim: Multicast dimension
        :type mcast_dim: int

        :return: TMA atom and tensor
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        op = (
            cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cpasync.CopyBulkTensorTileG2SMulticastOp()
        )

        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor

    @staticmethod
    def is_valid_dtypes(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        d_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
    ) -> bool:
        """
        Check if the dtypes are valid

        :param a_dtype: The data type of tensor A
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of tensor B
        :type b_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param d_dtype: The data type of the output tensor
        :type d_dtype: Type[cutlass.Numeric]
        :param a_major: major mode of tensor A
        :type a_major: str
        :param b_major: major mode of tensor B
        :type b_major: str

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        # tested a_dtype
        if a_dtype not in {
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        # tested b_dtype
        if b_dtype not in {
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        # tested acc_dtype
        if acc_dtype not in {cutlass.Float32, cutlass.Float16}:
            is_valid = False
        # tested d_dtype
        if d_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            is_valid = False
        # make sure a_dtype == b_dtype for Float16
        if a_dtype.width == 16 and a_dtype != b_dtype:
            is_valid = False
        # make sure a_dtype.width == b_dtype.width (i.e, Float8E4M3FN or Float8E5M2)
        if a_dtype.width != b_dtype.width:
            is_valid = False

        # for Float8 types, this implementation only supports k-major layout
        if (a_dtype.width == 8 and a_major != "k") or (b_dtype.width == 8 and b_major != "k"):
            is_valid = False

        return is_valid


def run(
    mnkl: Tuple[int, int, int, int],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    d_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    d_major: str,
    tile_shape_mnk: Tuple[int, int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float,
    warmup_iterations: int,
    iterations: int,
    skip_ref_check: bool,
    persistent: bool,
    pingpong: bool,
    use_cold_l2: bool = False,
    **kwargs,
):
    """
    Prepare A/B/C tensors, launch GPU kernel, and reference checking.

    :param mnkl: Problem size (M, N, K, L)
    :type mnkl: Tuple[int, int, int, int]
    :param a_dtype: Data type for input tensor A
    :type a_dtype: Type[cutlass.Numeric]
    :param b_dtype: Data type for input tensor B
    :type b_dtype: Type[cutlass.Numeric]
    :param d_dtype: Data type for output tensor C
    :type d_dtype: Type[cutlass.Numeric]
    :param acc_dtype: Data type for accumulation during matrix multiplication
    :type acc_dtype: Type[cutlass.Numeric]
    :param a_major/b_major/d_major: Memory layout of tensor A/B/C
    :type a_major/b_major/d_major: str
    :param tile_shape_mnk: CTA tile shape (M, N, K)
    :type tile_shape_mnk: Tuple[int, int, int]
    :param cluster_shape_mn: Cluster shape (M, N)
    :type cluster_shape_mn: Tuple[int, int]
    :param tolerance: Tolerance value for reference validation comparison
    :type tolerance: float
    :param warmup_iterations: Number of warmup iterations before benchmarking, defaults to 0
    :type warmup_iterations: int, optional
    :param iterations: Number of benchmark iterations to run, defaults to 1
    :type iterations: int, optional
    :param skip_ref_check: Whether to skip reference result validation, defaults to False
    :type skip_ref_check: bool, optional
    :param use_cold_l2: Whether to use circular buffer strategy to ensure cold L2 cache, defaults to False
    :type use_cold_l2: bool, optional
    :return: Execution time of the GEMM kernel in microseconds
    :rtype: float
    """

    print("Running Hopper Dense GEMM with:")
    print(f"mnkl: {mnkl}")
    print(f"A dtype: {a_dtype}, B dtype: {b_dtype}, C dtype: {d_dtype}, Acc dtype: {acc_dtype}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {d_major}")
    print(f"Tile Shape: {tile_shape_mnk}, Cluster Shape: {cluster_shape_mn}")
    print(f"Tolerance: {tolerance}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {use_cold_l2}")

    # Unpack parameters
    m, n, k, l = mnkl
    cluster_shape_mnk = (*cluster_shape_mn, 1)

    # Skip unsupported types
    if not HopperWgmmaGemmKernel.is_valid_dtypes(
        a_dtype, b_dtype, acc_dtype, d_dtype, a_major, b_major
    ):
        raise TypeError(
            f"Skipping due to unsupported combination of types and majors: {a_dtype}, {b_dtype}, {acc_dtype}, {d_dtype}, {a_major=}, {b_major=}"
        )

    # Prepare pytorch tensors: A, B (random from 0 to 2) and C (all zero)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    torch.manual_seed(1111)

    # Create and permute tensor A/B/C
    def create_and_permute_tensor(l, mode0, mode1, is_mode0_major, dtype, is_dynamic_layout=True):
        # is_mode0_major: (l, mode1, mode0) -> (mode0, mode1, l)
        # else : (l, mode0, mode1) -> (mode0, mode1, l)
        shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
        permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
        is_unsigned = dtype in {cutlass.Uint8}
        # Temporarily use uint8 as torch does not support fp8 type
        torch_dtype = (
            cutlass_torch.dtype(dtype)
            if dtype not in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}
            else torch.uint8
        )

        # Create dtype torch tensor (cpu)
        torch_tensor_cpu = cutlass.torch.create_and_permute_torch_tensor(
            shape,
            torch_dtype,
            permute_order=permute_order,
            # init_type=cutlass.torch.TensorInitType.RANDOM,
            # init_config=cutlass.torch.RandomInitConfig(
            #     min_val=0 if is_unsigned else -2, max_val=4 if is_unsigned else 2
            # ),
            init_type=cutlass.torch.TensorInitType.GAUSSIAN,
            init_config=cutlass.torch.GaussianInitConfig(std=k ** (-0.5), scale=1),
        )
        # Create dtype torch tensor (gpu)
        torch_tensor = torch_tensor_cpu.cuda()

        # Create f32 torch tensor (cpu)
        f32_torch_tensor = torch_tensor_cpu.to(dtype=torch.float32)

        # Create dtype cute tensor (gpu)
        cute_tensor = from_dlpack(torch_tensor, assumed_align=16)
        cute_tensor.element_type = dtype
        if is_dynamic_layout:
            cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=(0 if is_mode0_major else 1))
        cute_tensor = cutlass.torch.convert_cute_tensor(
            f32_torch_tensor,
            cute_tensor,
            dtype,
            is_dynamic_layout=is_dynamic_layout,
        )

        return f32_torch_tensor, cute_tensor, torch_tensor

    a, mA, a_torch = create_and_permute_tensor(l, m, k, a_major == "m", a_dtype)
    b, mB, b_torch = create_and_permute_tensor(l, n, k, b_major == "n", b_dtype)
    c, mC, c_torch = create_and_permute_tensor(l, m, n, d_major == "m", d_dtype)

    gemm = HopperWgmmaGemmKernel(
        acc_dtype,
        tile_shape_mnk,
        cluster_shape_mnk,
        pingpong=args.pingpong,
        is_persistent=args.persistent,
    )

    # Compute max active clusters on current device
    if args.persistent:
        max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(
            cluster_shape_mn[0] * cluster_shape_mn[1]
        )
    else:
        max_active_clusters = 0

    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    # compile gemm kernel
    compiled_gemm = cute.compile(gemm, mA, mB, mC, max_active_clusters, stream)

    if not skip_ref_check:
        # execution
        compiled_gemm(mA, mB, mC, max_active_clusters, stream)

        torch.cuda.synchronize()

        # Ref check
        ref = (torch.einsum("mkl,nkl->mnl", a, b)).cpu()

        if d_dtype in (cutlass.Float8E4M3FN, cutlass.Float8E5M2):
            # m major: (l, n, m) -> (m, n, l)
            # n major: (l, m, n) -> (m, n, l)
            permute_order = (1, 2, 0) if d_major == "n" else (2, 1, 0)
            shape = (l, m, n) if d_major == "n" else (l, n, m)
            f8_torch_tensor = cutlass_torch.create_and_permute_torch_tensor(
                shape,
                torch.uint8,
                permute_order=permute_order,
                init_type=cutlass_torch.TensorInitType.SKIP,
            ).cuda()
            # Create dtype cute tensor (gpu)
            ref_c_tensor = from_dlpack(f8_torch_tensor, assumed_align=16).mark_layout_dynamic(
                leading_dim=(1 if d_major == "n" else 0)
            )
            ref_c_tensor.element_type = d_dtype
            ref_c_tensor = cutlass_torch.convert_cute_tensor(
                ref,
                ref_c_tensor,
                d_dtype,
                is_dynamic_layout=True,
            )
            ref_c = f8_torch_tensor.cpu()
        else:
            ref_c = ref.to(cutlass_torch.dtype(d_dtype))

        torch.testing.assert_close(c_torch.cpu(), ref_c, atol=tolerance, rtol=1e-03)

    def generate_tensors():
        _, mA_workspace, _ = create_and_permute_tensor(l, m, k, a_major == "m", a_dtype)
        _, mB_workspace, _ = create_and_permute_tensor(l, n, k, b_major == "n", b_dtype)
        _, mC_workspace, _ = create_and_permute_tensor(l, m, n, d_major == "m", d_dtype)
        return testing.JitArguments(
            mA_workspace, mB_workspace, mC_workspace, max_active_clusters, stream
        )

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            a_torch.numel() * a_torch.element_size()
            + b_torch.numel() * b_torch.element_size()
            + c_torch.numel() * c_torch.element_size()
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = testing.benchmark(
        compiled_gemm,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    from triton.testing import do_bench

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    flops = 2 * m * n * k * l

    repeats = 30
    # repeats = 1
    warmup = 5

    import time

    time.sleep(0.5)
    fn = lambda: torch.matmul(a_torch.permute(2, 0, 1), b_torch.permute(2, 0, 1).mT)
    timing_cublas = do_bench(fn, warmup=warmup, rep=repeats)
    tflops_cublas = flops / (timing_cublas * 1e9)  # Convert to TFlops
    print(f"CuBLAS Average time: {timing_cublas:.3f} ms, TFLOPS: {tflops_cublas:.1f}")

    time.sleep(0.5)
    fn = lambda: compiled_gemm(mA, mB, mC, max_active_clusters, current_stream)
    timing = do_bench(fn, warmup=warmup, rep=repeats)
    tflops = flops / (timing * 1e9)  # Convert to TFlops
    print(f"Cute-DSL Average time: {timing:.3f} ms, TFLOPS: {tflops:.1f}")

    time.sleep(0.5)
    fn = lambda: torch.matmul(a_torch.permute(2, 0, 1), b_torch.permute(2, 0, 1).mT)
    timing_cublas = do_bench(fn, warmup=warmup, rep=repeats)
    tflops_cublas = flops / (timing_cublas * 1e9)  # Convert to TFlops
    print(f"CuBLAS Average time: {timing_cublas:.3f} ms, TFLOPS: {tflops_cublas:.1f}")

    return exec_time  # Return execution time in microseconds


if __name__ == "__main__":
    args = parse_arguments()
    run(
        args.mnkl,
        args.a_dtype,
        args.b_dtype,
        args.d_dtype,
        args.acc_dtype,
        args.a_major,
        args.b_major,
        args.d_major,
        args.tile_shape_mnk,
        args.cluster_shape_mn,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.persistent,
        args.pingpong,
        args.use_cold_l2,
    )
    print("PASS")
