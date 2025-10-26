# Copyright (c) OpenMMLab. All rights reserved.
"""Backward compatibility for visdet.core.evaluation submodule."""

from visdet.evaluation.functional import (
    eval_map,
    eval_recalls,
    print_map_summary,
)
from visdet.structures.bbox import bbox_overlaps

# Alias for backward compatibility
recall = eval_recalls

__all__ = [
    "bbox_overlaps",
    "eval_map",
    "eval_recalls",
    "print_map_summary",
    "recall",
]
