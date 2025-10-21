# Copyright (c) OpenMMLab. All rights reserved.
"""Test-time augmentation helpers for visdet."""

from __future__ import annotations

from typing import Iterable

import torch


def merge_aug_results(aug_batch_results, aug_batch_img_metas):
    """Merge augmented detection results.

    This remains unimplemented because TTA is not required for our minimal
    deployment footprint. The helper stays in place so upstream call sites fail
    loudly rather than silently producing incorrect outputs.
    """
    raise NotImplementedError(
        "Test-time augmentation (merge_aug_results) is not yet implemented. This feature is not essential for basic Swin Mask R-CNN operation."
    )


def merge_aug_masks(aug_masks: Iterable[torch.Tensor], img_meta: dict) -> torch.Tensor:
    """Merge masks produced from multiple cascade stages or test-time augs.

    We do not support spatial transforms for TTA, so the helper simply averages
    the provided tensors. This mirrors MMSeg/MMDet behaviour when no geometric
    transforms are applied.

    Args:
        aug_masks: Iterable of mask tensors for a single image. Each tensor
            shares the same spatial shape and comes from a cascade stage or
            identical augmentation pipeline.
        img_meta: Metadata dictionary for the image. Included for API parity
            with MMDetection; currently unused but retained for future support.

    Returns:
        torch.Tensor: Averaged mask tensor.
    """
    aug_masks = list(aug_masks)
    if not aug_masks:
        raise ValueError("merge_aug_masks expects at least one mask tensor.")
    if len(aug_masks) == 1:
        return aug_masks[0]
    stacked = torch.stack(aug_masks, dim=0)
    return stacked.mean(dim=0)
