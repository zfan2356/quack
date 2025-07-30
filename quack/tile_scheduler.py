# Copyright (c) 2025, Tri Dao.

from typing import Tuple
from dataclasses import dataclass, fields
from enum import IntEnum

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from quack.fast_math import FastDivmod


@dataclass
class ParamsBase:
    def __extract_mlir_values__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, cutlass.Constexpr)]
        values, self._values_pos = [], []
        for obj in non_constexpr_fields:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        constexpr_fields = {n: f for n, f in all_fields.items() if isinstance(f, cutlass.Constexpr)}
        non_constexpr_fields = {
            n: f for n, f in all_fields.items() if not isinstance(f, cutlass.Constexpr)
        }
        for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
            non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            values = values[n_items:]
        return self.__class__(**non_constexpr_fields, **constexpr_fields)


class RasterOrderOption(IntEnum):
    AlongM = 0
    AlongN = 1
    Heuristic = 2  # Pick AlongM if tiles_n > tiles_m, else AlongN


class RasterOrder(IntEnum):
    AlongM = 0
    AlongN = 1


@dataclass
class TileSchedulerArguments(ParamsBase):
    problem_shape_ntile_mnl: cute.Shape
    raster_order: RasterOrderOption
    group_size: Int32
    cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
    is_persistent: cutlass.Constexpr[bool] = False


class StaticTileScheduler:
    @dataclass
    class Params(ParamsBase):
        problem_shape_ncluster_mnl: cute.Shape
        raster_order: RasterOrder
        num_clusters_per_problem_divmod: FastDivmod
        num_groups_regular: Int32
        group_size_divmod: FastDivmod
        group_size_tail_divmod: FastDivmod
        num_clusters_in_group_divmod: FastDivmod
        cluster_shape_mn: cutlass.Constexpr[cute.Shape]
        is_persistent: cutlass.Constexpr[bool]

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "StaticTileScheduler.Params":
            cluster_shape_mn = cutlass.const_expr(cute.select(args.cluster_shape_mnk, mode=[0, 1]))
            problem_shape_ntile_mn = cute.select(args.problem_shape_ntile_mnl, mode=[0, 1])
            problem_shape_ncluster_mn = cute.ceil_div(problem_shape_ntile_mn, cluster_shape_mn)
            raster_order = RasterOrder.AlongM
            if raster_order == RasterOrderOption.Heuristic:
                problem_blocks_m = cute.round_up(problem_shape_ncluster_mn[0], args.group_size)
                problem_blocks_n = cute.round_up(problem_shape_ncluster_mn[1], args.group_size)
                raster_order = (
                    RasterOrder.AlongM
                    if problem_blocks_n > problem_blocks_m
                    else RasterOrder.AlongN
                )
            ncluster_fast = (
                problem_shape_ncluster_mn[0]
                if raster_order == RasterOrder.AlongM
                else problem_shape_ncluster_mn[1]
            )
            ncluster_slow = (
                problem_shape_ncluster_mn[1]
                if raster_order == RasterOrder.AlongM
                else problem_shape_ncluster_mn[0]
            )
            num_clusters_per_problem = cute.size(problem_shape_ncluster_mn)
            group_size = min(args.group_size, ncluster_fast)
            problem_shape_ncluster_mnl = problem_shape_ncluster_mn + (
                args.problem_shape_ntile_mnl[2],
            )
            group_size_tail = ncluster_fast % group_size
            num_groups_regular = ncluster_fast // group_size
            num_clusters_in_group = group_size * ncluster_slow
            return StaticTileScheduler.Params(
                problem_shape_ncluster_mnl,
                raster_order,
                FastDivmod.create(num_clusters_per_problem),
                num_groups_regular,
                FastDivmod.create(group_size),
                # Don't divide by 0
                FastDivmod.create(group_size_tail if group_size_tail > 0 else 1),
                FastDivmod.create(num_clusters_in_group),
                cluster_shape_mn,
                args.is_persistent,
            )

    def __init__(
        self,
        current_work_linear_idx: Int32,
        num_tiles_executed: Int32,
        params: Params,
        *,
        loc=None,
        ip=None,
    ):
        self._current_work_linear_idx = current_work_linear_idx
        self._num_tiles_executed = num_tiles_executed
        self.params = params
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return StaticTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(params: Params, *, loc=None, ip=None) -> "StaticTileScheduler":
        if cutlass.const_expr(not params.is_persistent):
            cidx, cidy, _ = cute.arch.cluster_idx()
            cdimx, _, _ = cute.arch.cluster_dim()
            cluster_id = cidx + cidy * cdimx
            current_work_linear_idx = Int32(cluster_id)
        else:
            _, _, bidz = cute.arch.block_idx()
            current_work_linear_idx = Int32(bidz)
        return StaticTileScheduler(
            current_work_linear_idx,
            Int32(0),  # num_tiles_executed
            params,
            loc=loc,
            ip=ip,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        num_ctas_mnl = tuple(
            x * y for x, y in zip(params.problem_shape_ncluster_mnl, params.cluster_shape_mn)
        ) + (params.problem_shape_ncluster_mnl[2],)
        if cutlass.const_expr(not params.is_persistent):
            return num_ctas_mnl
        else:
            num_ctas_in_problem = cute.size(num_ctas_mnl, loc=loc, ip=ip)
            num_ctas_per_cluster = cute.size(params.cluster_shape_mn, loc=loc, ip=ip)
            # Total ctas that can run in one wave
            num_ctas_per_wave = max_active_clusters * num_ctas_per_cluster
            num_persistent_ctas = cutlass.min(num_ctas_in_problem, num_ctas_per_wave)
            num_persistent_clusters = num_persistent_ctas // num_ctas_per_cluster
            return (*params.cluster_shape_mn, num_persistent_clusters)

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        params = self.params
        if cutlass.const_expr(not params.is_persistent):
            cluster_id_in_problem = self._current_work_linear_idx
            _, _, bidz = cute.arch.block_idx()
        else:
            bidz, cluster_id_in_problem = params.num_clusters_per_problem_divmod.divmod(
                self._current_work_linear_idx
            )
        # CTA Swizzle to promote L2 data reuse
        group_id, id_in_group = params.num_clusters_in_group_divmod.divmod(cluster_id_in_problem)
        cid_fast_in_group, cid_slow = Int32(0), Int32(0)
        if group_id < params.num_groups_regular:
            cid_slow, cid_fast_in_group = params.group_size_divmod.divmod(id_in_group)
        else:  # tail part
            cid_slow, cid_fast_in_group = params.group_size_tail_divmod.divmod(id_in_group)
        cid_fast = group_id * params.group_size_divmod.divisor + cid_fast_in_group
        cid_m, cid_n = cid_fast, cid_slow
        if params.raster_order == RasterOrder.AlongN:
            cid_n, cid_m = cid_fast, cid_slow

        # Get the pid from cluster id
        bidx_in_cluster = cute.arch.block_in_cluster_idx()
        pid_m = cid_m * params.cluster_shape_mn[0] + bidx_in_cluster[0]
        pid_n = cid_n * params.cluster_shape_mn[1] + bidx_in_cluster[1]
        tile_coord_mnkl = (pid_m, pid_n, None, bidz)
        if cutlass.const_expr(not params.is_persistent):
            is_valid = self._num_tiles_executed == 0
        else:
            is_valid = self._current_work_linear_idx < cute.size(params.problem_shape_ncluster_mnl)
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, advance_count: int = 1, loc=None, ip=None):
        if cutlass.const_expr(self.params.is_persistent):
            num_persistent_clusters = cute.arch.grid_dim()[2]
            self._current_work_linear_idx += advance_count * Int32(num_persistent_clusters)
        self._num_tiles_executed += Int32(1)

    @property
    def num_tiles_executed(self) -> Int32:
        return self._num_tiles_executed

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self._current_work_linear_idx,
            self._num_tiles_executed,
            self.params,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self._current_work_linear_idx,
                self._num_tiles_executed,
                self.params,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return StaticTileScheduler(*(tuple(obj_list)), loc=self._loc)
