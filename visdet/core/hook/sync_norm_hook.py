# Copyright (c) OpenMMLab. All rights reserved.
"""Backward compatibility for SyncNormHook."""


class SyncNormHook:
    """Stub for SyncNormHook - not in minimal build."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SyncNormHook is not available in the minimal visdet build.")


__all__ = ["SyncNormHook"]
