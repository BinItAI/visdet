# Copyright (c) OpenMMLab. All rights reserved.
from . import transforms as _transforms  # noqa: F401 - for pipeline registration
from visdet.datasets.coco import CocoDataset
from visdet.datasets.utils import get_loading_pipeline, replace_ImageToTensor

# Ensure pipelines are registered after builder is loaded
_transforms._register_pipelines()

__all__ = ["CocoDataset", "get_loading_pipeline", "replace_ImageToTensor"]
