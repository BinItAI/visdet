# Copyright (c) OpenMMLab. All rights reserved.
"""Backward compatibility for visdet.core.post_processing module."""


def mask_matrix_nms(*args, **kwargs):
    """Stub for mask_matrix_nms - not in minimal build."""
    raise NotImplementedError(
        "mask_matrix_nms is not available in the minimal visdet build. "
        "This was used for SOLO/SOLOv2 instance segmentation."
    )


__all__ = ["mask_matrix_nms"]
