# Copyright (c) OpenMMLab. All rights reserved.
"""Collecting some commonly used type hint in mmdetection."""

from collections.abc import Sequence
from typing import Any, Union

from visdet.engine.config import ConfigDict
from visdet.engine.structures import InstanceData, PixelData

# TODO: Need to avoid circular import with assigner and sampler
# Type hint of config data
ConfigType = Union[ConfigDict, dict, str, dict[str, Any]]
OptConfigType = ConfigType | None
# Type hint of one or more config data
MultiConfig = Union[ConfigType, list[ConfigType]]
OptMultiConfig = MultiConfig | None

InstanceList = list[InstanceData]
OptInstanceList = InstanceList | None

PixelList = list[PixelData]
OptPixelList = PixelList | None

RangeType = Sequence[tuple[int, int]]
