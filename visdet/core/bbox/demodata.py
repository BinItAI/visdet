# Copyright (c) OpenMMLab. All rights reserved.
"""Backward compatibility for bbox demodata utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# These are test utilities - provide stubs or imports if they exist
# TODO: Check if these exist in the codebase


def ensure_rng(rng: int | np.random.RandomState | None = None) -> np.random.RandomState:
    """Ensure we have a random number generator."""
    import numpy as np

    if rng is None:
        return np.random.RandomState()
    if isinstance(rng, np.random.RandomState):
        return rng
    return np.random.RandomState(rng)


def random_boxes(num_boxes, scale=1.0, rng=None, dtype="float32"):
    """Generate random boxes for testing."""
    import torch

    rng = ensure_rng(rng)

    # Generate random boxes in xyxy format
    x1 = torch.from_numpy(rng.rand(num_boxes) * scale).to(dtype=getattr(torch, dtype))
    y1 = torch.from_numpy(rng.rand(num_boxes) * scale).to(dtype=getattr(torch, dtype))
    x2 = x1 + torch.from_numpy(rng.rand(num_boxes) * scale).to(dtype=getattr(torch, dtype))
    y2 = y1 + torch.from_numpy(rng.rand(num_boxes) * scale).to(dtype=getattr(torch, dtype))

    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    return boxes


__all__ = ["ensure_rng", "random_boxes"]
