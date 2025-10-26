# Copyright (c) OpenMMLab. All rights reserved.
"""Backward compatibility for bbox demodata utilities."""

# These are test utilities - provide stubs or imports if they exist
# TODO: Check if these exist in the codebase


def ensure_rng(rng=None):
    """Ensure we have a random number generator."""
    import numpy as np

    if rng is None:
        return np.random.RandomState()
    return rng


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
