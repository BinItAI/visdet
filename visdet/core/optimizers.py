# Copyright (c) OpenMMLab. All rights reserved.
"""Backward compatibility for visdet.core.optimizers module."""


class LearningRateDecayOptimizerConstructor:
    """Stub for LearningRateDecayOptimizerConstructor - not in minimal build."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "LearningRateDecayOptimizerConstructor is not available in the minimal visdet build. "
            "Use standard optimizer configurations instead."
        )


__all__ = ["LearningRateDecayOptimizerConstructor"]
