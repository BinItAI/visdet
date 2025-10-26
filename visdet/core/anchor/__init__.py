# Copyright (c) OpenMMLab. All rights reserved.
"""Anchor/prior generators - backward compatibility for visdet.core.anchor namespace."""

from visdet.models.task_modules.prior_generators import AnchorGenerator

# Alias for backward compatibility
build_anchor_generator = AnchorGenerator

__all__ = ["AnchorGenerator", "build_anchor_generator"]
