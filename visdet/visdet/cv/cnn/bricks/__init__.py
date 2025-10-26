# ruff: noqa
"""
CNN building blocks module.

This module provides basic neural network components like convolutions,
normalizations, activations, and attention mechanisms.
"""

from .norm import build_norm_layer  # noqa: F401
from .transformer import *  # noqa: F401, F403

__all__ = ["build_norm_layer"]
