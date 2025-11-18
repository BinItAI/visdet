# Copyright (c) OpenMMLab. All rights reserved.
"""Dataset wrapper aliases used across visdet."""

from visdet.engine.dataset.dataset_wrapper import (
    ClassBalancedDataset as _ClassBalancedDataset,
)
from visdet.engine.dataset.dataset_wrapper import (
    ConcatDataset as _ConcatDataset,
)
from visdet.engine.dataset.dataset_wrapper import (
    RepeatDataset as _RepeatDataset,
)

ClassBalancedDataset = _ClassBalancedDataset
ConcatDataset = _ConcatDataset
RepeatDataset = _RepeatDataset

__all__ = ["ClassBalancedDataset", "ConcatDataset", "RepeatDataset"]
