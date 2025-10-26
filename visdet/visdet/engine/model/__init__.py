# ruff: noqa
"""
Model utilities for visdet.

This module provides base model classes and utilities.
"""

from typing import List
import torch.nn as nn

from .base_module import BaseModule, BaseModel  # noqa: F401
from .data_preprocessor import BaseDataPreprocessor, ImgDataPreprocessor  # noqa: F401
from .weight_init import constant_init, trunc_normal_, trunc_normal_init  # noqa: F401


# Simple alias for ModuleList
class ModuleList(nn.ModuleList):
    """ModuleList for visdet.

    This is an alias for torch.nn.ModuleList with additional features.
    """

    pass


__all__ = [
    "BaseModule",
    "BaseModel",
    "ModuleList",
    "BaseDataPreprocessor",
    "ImgDataPreprocessor",
    "constant_init",
    "trunc_normal_",
    "trunc_normal_init",
]
