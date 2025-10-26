# Copyright (c) OpenMMLab. All rights reserved.
"""Backward compatibility for visdet.core.hook module."""

# Most hooks should come from visengine
# These specific hooks may not exist in minimal build - provide stubs


class ExpMomentumEMAHook:
    """Stub for ExpMomentumEMAHook - not in minimal build."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ExpMomentumEMAHook is not available in the minimal visdet build. This was a YOLOX-specific hook."
        )


class YOLOXLrUpdaterHook:
    """Stub for YOLOXLrUpdaterHook - not in minimal build."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "YOLOXLrUpdaterHook is not available in the minimal visdet build. This was a YOLOX-specific hook."
        )


class MemoryProfilerHook:
    """Stub for MemoryProfilerHook - not in minimal build."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("MemoryProfilerHook is not available in the minimal visdet build.")


__all__ = [
    "ExpMomentumEMAHook",
    "YOLOXLrUpdaterHook",
    "MemoryProfilerHook",
]
