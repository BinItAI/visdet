# ruff: noqa
# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.
from .utils import (
    get_device,
    get_max_cuda_memory,
    is_cuda_available,
)

# Add lambda function that always returns False
# Required by mmcv
is_mlu_available = lambda: False
is_npu_available = lambda: False
is_musa_available = lambda: False
is_mps_available = lambda: False

__all__ = [
    "get_device",
    "get_max_cuda_memory",
    "is_cuda_available",
    "is_mlu_available",
    "is_npu_available",
    "is_musa_available",
    "is_mps_available",
]
