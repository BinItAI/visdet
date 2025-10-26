# Copyright (c) OpenMMLab. All rights reserved.
"""Test utilities for backbones - only Swin is in scope.

The utilities in utils.py are only needed for backbone tests that are
being skipped (ResNet, RegNet, etc.). Since those tests don't get
collected, we don't import these utilities to avoid import errors.
"""

__all__ = []
