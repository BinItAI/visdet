# Copyright (c) OpenMMLab. All rights reserved.
"""Backward compatibility for visdet.core.utils module."""

from visdet.structures.bbox import distance2bbox

# TODO: Add other utility functions if they exist
# filter_scores_and_topk, select_single_mlvl, etc.


def filter_scores_and_topk(*args, **kwargs):
    """Stub - check if this exists elsewhere in codebase."""
    raise NotImplementedError(
        "filter_scores_and_topk is not available. Check if this function "
        "has been moved or is specific to non-Mask-RCNN detectors."
    )


def select_single_mlvl(*args, **kwargs):
    """Stub - check if this exists elsewhere in codebase."""
    raise NotImplementedError(
        "select_single_mlvl is not available. Check if this function "
        "has been moved or is specific to non-Mask-RCNN detectors."
    )


__all__ = [
    "distance2bbox",
    "filter_scores_and_topk",
    "select_single_mlvl",
]
