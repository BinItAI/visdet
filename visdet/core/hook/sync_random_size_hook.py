# Copyright (c) OpenMMLab. All rights reserved.
"""Backward compatibility for SyncRandomSizeHook."""


class SyncRandomSizeHook:
    """Stub for SyncRandomSizeHook - not in minimal build."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "SyncRandomSizeHook is not available in the minimal visdet build. This was a YOLOX-specific hook."
        )


__all__ = ["SyncRandomSizeHook"]
