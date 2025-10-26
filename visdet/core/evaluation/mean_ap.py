# Copyright (c) OpenMMLab. All rights reserved.
"""Backward compatibility for mean_ap evaluation."""

from visdet.evaluation.functional import eval_map, print_map_summary

__all__ = [
    "eval_map",
    "print_map_summary",
]
