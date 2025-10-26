# Copyright (c) OpenMMLab. All rights reserved.
from .coco import CocoDataset
from .utils import get_loading_pipeline, replace_ImageToTensor

__all__ = ["CocoDataset", "get_loading_pipeline", "replace_ImageToTensor"]
