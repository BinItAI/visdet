# ruff: noqa
# Copyright (c) OpenMMLab. All rights reserved.
from .det_inferencer import DetInferencer
from .inference import inference_detector, init_detector

__all__ = [
    "DetInferencer",
    "inference_detector",
    "init_detector",
]
