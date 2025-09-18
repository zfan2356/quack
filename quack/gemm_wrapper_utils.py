# Copyright (c) 2025, Tri Dao.
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from torch import Tensor

import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack, make_ptr

from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.dense_gemm_sm90 import TileSchedulerOptions


@dataclass
class GemmTensorInfo:
    tensor: Optional[Tensor]
    dtype: Optional[Any] = None
    major: Optional[str] = None
    cute_tensor: Optional[cute.Tensor] = None


class GemmWrapperBase:
    @staticmethod
    def validate_tensor_3d(tensor: Tensor, name: str) -> None:
        assert tensor.dim() == 3 and tensor.is_cuda, f"{name} must be a 3D CUDA tensor"
        assert tensor.dtype in torch2cute_dtype_map, f"Unsupported dtype for {name}"

    @staticmethod
    def validate_shape(tensor: Tensor, expected_shape: Tuple[int, ...], name: str) -> None:
        assert tensor.shape == expected_shape, (
            f"{name} must have shape {expected_shape}, got {tensor.shape}"
        )

    @staticmethod
    def get_major_order(tensor: Tensor, dims: Tuple[str, str, str]) -> str:
        # Tensor is already permuted to (dims[0], dims[1], dims[2])
        # stride(1) == 1 means dims[1] is contiguous (innermost)
        return dims[1] if tensor.stride(1) == 1 else dims[0]

    @staticmethod
    def create_cute_tensor(
        tensor: Optional[Tensor],
        major: Optional[str],
        dims: Tuple[str, str, str],
        assumed_align: int = 16,
    ) -> Optional[cute.Tensor]:
        if tensor is None:
            return None
        # Tensor is already permuted to (dims[0], dims[1], dims[2])
        # If major is dims[1], leading_dim is 1; if major is dims[0], leading_dim is 0
        leading_dim = 1 if major == dims[1] else 0
        return from_dlpack(tensor.detach(), assumed_align=assumed_align).mark_layout_dynamic(
            leading_dim=leading_dim
        )

    @staticmethod
    def validate_and_prepare_tensors(
        A: Tensor,
        B: Tensor,
        D: Optional[Tensor] = None,
        C: Optional[Tensor] = None,
        additional_tensors: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[int, int, int, int, Dict[str, GemmTensorInfo]]:
        GemmWrapperBase.validate_tensor_3d(A, "A")
        L, M, K = A.shape
        GemmWrapperBase.validate_tensor_3d(B, "B")
        _, N, _ = B.shape
        assert B.dtype == A.dtype, "A and B must have the same dtype"
        GemmWrapperBase.validate_shape(B, (L, N, K), "B")
        tensors = {
            "A": GemmTensorInfo(A),
            "B": GemmTensorInfo(B),
            "D": GemmTensorInfo(D),
            "C": GemmTensorInfo(C),
        }
        if D is not None:
            GemmWrapperBase.validate_tensor_3d(D, "D")
            GemmWrapperBase.validate_shape(D, (L, M, N), "D")
        if C is not None:
            GemmWrapperBase.validate_tensor_3d(C, "C")
            GemmWrapperBase.validate_shape(C, (L, M, N), "C")
        if additional_tensors:
            for name, tensor in additional_tensors.items():
                if tensor is not None:
                    GemmWrapperBase.validate_tensor_3d(tensor, name)
                    GemmWrapperBase.validate_shape(tensor, (L, M, N), name)
                tensors[name] = GemmTensorInfo(tensor)

        return L, M, K, N, tensors

    @staticmethod
    def permute_tensors(tensors: Dict[str, GemmTensorInfo]) -> None:
        for info in tensors.values():
            if info.tensor is not None:
                info.tensor = info.tensor.permute(1, 2, 0)

    @staticmethod
    def extract_dtypes(tensors: Dict[str, GemmTensorInfo]) -> None:
        for info in tensors.values():
            if info.tensor is not None:
                info.dtype = torch2cute_dtype_map[info.tensor.dtype]

    @staticmethod
    def determine_major_orders(
        tensors: Dict[str, GemmTensorInfo], major_configs: Dict[str, Tuple[str, str, str]]
    ) -> None:
        for name, dims in major_configs.items():
            if name in tensors and tensors[name].tensor is not None:
                tensors[name].major = GemmWrapperBase.get_major_order(tensors[name].tensor, dims)

    @staticmethod
    def create_cute_tensors(
        tensors: Dict[str, GemmTensorInfo], major_configs: Dict[str, Tuple[str, str, str]]
    ) -> None:
        for name, info in tensors.items():
            if info.tensor is not None and name in major_configs:
                info.cute_tensor = GemmWrapperBase.create_cute_tensor(
                    info.tensor, info.major, major_configs[name]
                )

    @staticmethod
    def create_scheduler_args(
        max_active_clusters: int, tile_count_semaphore: Optional[Tensor] = None
    ) -> TileSchedulerOptions:
        return TileSchedulerOptions(
            Int32(max_active_clusters),
            tile_count_semaphore=make_ptr(
                Int32, tile_count_semaphore.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
            )
            if tile_count_semaphore is not None
            else None,
        )

    @staticmethod
    def get_compile_key(
        tensors: Dict[str, GemmTensorInfo],
        activation: Optional[str],
        tile_shape_mn: Tuple[int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        pingpong: bool,
        persistent: bool,
        has_semaphore: bool,
        *args,
        key_tensor_names: Tuple[str, ...] = ("A", "B", "D", "C"),
    ) -> Tuple:
        key_parts = []
        for name in key_tensor_names:
            if name in tensors:
                key_parts.append(tensors[name].dtype)
        key_parts.append(activation)
        key_parts.extend([tile_shape_mn, cluster_shape_mnk])
        for name in key_tensor_names:
            if name in tensors:
                key_parts.append(tensors[name].major)
        key_parts.extend([pingpong, persistent, has_semaphore])
        key_parts.extend(args)
        return tuple(key_parts)
