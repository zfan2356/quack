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


class RasterOrder(IntEnum):
    AlongM = 0
    AlongN = 1


@dataclass
class TileSchedulerArguments(ParamsBase):
    problem_shape_ntile_mnl: cute.Shape
    raster_order: RasterOrder
    group_size: Int32
    cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
    is_persistent: cutlass.Constexpr[bool] = False


class SingleTileScheduler:
    @dataclass
    class Params(ParamsBase):
        problem_shape_ncluster_mnl: cute.Shape
        raster_order: RasterOrder
        num_groups_regular: Int32
        group_size_divmod: FastDivmod
        group_size_tail_divmod: FastDivmod
        num_clusters_in_group_divmod: FastDivmod
        cluster_shape_mn: cutlass.Constexpr[cute.Shape]

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileScheduler.Params":
            cluster_shape_mn = cutlass.const_expr(cute.select(args.cluster_shape_mnk, mode=[0, 1]))
            problem_shape_ntile_mn = cute.select(args.problem_shape_ntile_mnl, mode=[0, 1])
            problem_shape_ncluster_mn = cute.ceil_div(problem_shape_ntile_mn, cluster_shape_mn)
            group_size = min(args.group_size, problem_shape_ncluster_mn[int(args.raster_order)])
            problem_shape_ncluster_mnl = problem_shape_ncluster_mn + (
                args.problem_shape_ntile_mnl[2],
            )
            group_size_tail = problem_shape_ncluster_mn[int(args.raster_order)] % group_size
            num_groups_regular = problem_shape_ncluster_mn[int(args.raster_order)] // group_size
            num_clusters_in_group = group_size * problem_shape_ncluster_mn[1 - args.raster_order]
            return SingleTileScheduler.Params(
                problem_shape_ncluster_mnl,
                args.raster_order,
                num_groups_regular,
                FastDivmod.create(group_size),
                # Don't divide by 0
                FastDivmod.create(group_size_tail if group_size_tail > 0 else 1),
                FastDivmod.create(num_clusters_in_group),
                cluster_shape_mn,
            )

    def __init__(
        self,
        cluster_coord: cute.Coord,
        raster_order: RasterOrder,
        num_groups_regular: Int32,
        group_size_divmod: FastDivmod,
        group_size_tail_divmod: FastDivmod,
        num_clusters_in_group_divmod: FastDivmod,
        cluster_shape_mn: cutlass.Constexpr[cute.Shape],
        *,
        loc=None,
        ip=None,
    ):
        self._cluster_coord = cluster_coord
        self._is_first_block = True
        self.raster_order = raster_order
        self.num_groups_regular = num_groups_regular
        self.group_size_divmod = group_size_divmod
        self.group_size_tail_divmod = group_size_tail_divmod
        self.num_clusters_in_group_divmod = num_clusters_in_group_divmod
        self.cluster_shape_mn = cluster_shape_mn
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return SingleTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileScheduler":
        cidx, cidy, _ = cute.arch.cluster_idx()
        _, _, bidz = cute.arch.block_idx()
        cdimx, _, _ = cute.arch.cluster_dim()
        cluster_id = cidx + cidy * cdimx
        return SingleTileScheduler(
            (cluster_id, bidz),
            params.raster_order,
            params.num_groups_regular,
            params.group_size_divmod,
            params.group_size_tail_divmod,
            params.num_clusters_in_group_divmod,
            params.cluster_shape_mn,
            loc=loc,
            ip=ip,
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        num_ctas_mnl = tuple(
            x * y for x, y in zip(params.problem_shape_ncluster_mnl, params.cluster_shape_mn)
        ) + (params.problem_shape_ncluster_mnl[2],)
        return num_ctas_mnl

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        cluster_id, bidz = self._cluster_coord
        # CTA Swizzle to promote L2 data reuse
        group_id, id_in_group = self.num_clusters_in_group_divmod.divmod(cluster_id)
        cid_fast_in_group, cid_slow = Int32(0), Int32(0)
        if group_id < self.num_groups_regular:
            cid_slow, cid_fast_in_group = self.group_size_divmod.divmod(id_in_group)
        else:  # tail part
            cid_slow, cid_fast_in_group = self.group_size_tail_divmod.divmod(id_in_group)
        cid_fast = group_id * self.group_size_divmod.divisor + cid_fast_in_group
        cid_m, cid_n = (
            (cid_fast, cid_slow)
            if self.raster_order == RasterOrder.AlongM
            else (cid_slow, cid_fast)
        )
        # Get the pid from cluster id
        bidx_in_cluster = cute.arch.block_in_cluster_idx()
        pid_m = cid_m * self.cluster_shape_mn[0] + bidx_in_cluster[0]
        pid_n = cid_n * self.cluster_shape_mn[1] + bidx_in_cluster[1]
        tile_coord_mnkl = (pid_m, pid_n, None, bidz)
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, self._is_first_block)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._is_first_block = False

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self._cluster_coord,
            self.raster_order,
            self.num_groups_regular,
            self.group_size_divmod,
            self.group_size_tail_divmod,
            self.num_clusters_in_group_divmod,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self._cluster_coord,
                self.raster_order,
                self.num_groups_regular,
                self.group_size_divmod,
                self.group_size_tail_divmod,
                self.num_clusters_in_group_divmod,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return SingleTileScheduler(*(tuple(obj_list), self.cluster_shape_mn), loc=self._loc)
