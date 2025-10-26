# Copyright (c) OpenMMLab. All rights reserved.
"""Bbox samplers - backward compatibility wrapper."""

from visdet.models.task_modules.samplers import PseudoSampler, RandomSampler, SamplingResult

__all__ = ["RandomSampler", "PseudoSampler", "SamplingResult"]
