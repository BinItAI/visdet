# Copyright (c) OpenMMLab. All rights reserved.
"""Backward compatibility for visdet.core.bbox.samplers."""

from visdet.models.task_modules.builder import build_sampler
from visdet.models.task_modules.samplers import (
    PseudoSampler,
    RandomSampler,
    SamplingResult,
)


# Stubs for samplers not in minimal build
class OHEMSampler:
    """Stub for OHEMSampler - not in minimal build."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "OHEMSampler (Online Hard Example Mining) is not available in the minimal visdet build."
        )


class ScoreHLRSampler:
    """Stub for ScoreHLRSampler - not in minimal build."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("ScoreHLRSampler is not available in the minimal visdet build.")


__all__ = [
    "PseudoSampler",
    "RandomSampler",
    "SamplingResult",
    "OHEMSampler",
    "ScoreHLRSampler",
    "build_sampler",
]
