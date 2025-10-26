# Copyright (c) OpenMMLab. All rights reserved.
"""Builders for detector models."""

from visdet.registry import MODELS


def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build a detector model.

    Args:
        cfg (dict): Config for building detector
        train_cfg (dict, optional): Training config. Deprecated.
        test_cfg (dict, optional): Testing config. Deprecated.

    Returns:
        BaseDetector: The detector model
    """
    if train_cfg is not None or test_cfg is not None:
        import warnings

        warnings.warn(
            "train_cfg and test_cfg are deprecated, please specify them in the model config",
            UserWarning,
        )
    return MODELS.build(cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
