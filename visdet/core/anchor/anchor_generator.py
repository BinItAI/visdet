# Copyright (c) OpenMMLab. All rights reserved.
"""Backward compatibility for specific anchor generator imports."""

from visdet.models.task_modules.prior_generators import AnchorGenerator


# Stubs for generators not in minimal build
class SSDAnchorGenerator:
    """Stub for SSDAnchorGenerator - not in minimal build."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SSDAnchorGenerator is not available in the minimal visdet build.")


class YOLOAnchorGenerator:
    """Stub for YOLOAnchorGenerator - not in minimal build."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("YOLOAnchorGenerator is not available in the minimal visdet build.")


__all__ = [
    "AnchorGenerator",
    "SSDAnchorGenerator",
    "YOLOAnchorGenerator",
]
