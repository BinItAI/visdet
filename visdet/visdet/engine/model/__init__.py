# ruff: noqa
"""
Model utilities for visdet.

This module provides base model classes and utilities.
"""

from typing import List
import torch.nn as nn

from .base_module import BaseModule  # noqa: F401


# Simple alias for ModuleList
class ModuleList(nn.ModuleList):
    """ModuleList for visdet.

    This is an alias for torch.nn.ModuleList with additional features.
    """

    pass


__all__ = ["BaseModule", "ModuleList"]
