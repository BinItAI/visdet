# ruff: noqa
# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.
"""Collecting some commonly used type hint in mmdetection."""

from collections.abc import Sequence
from typing import Optional, Union

from visengine.config import ConfigDict
from visengine.structures import InstanceData, PixelData

# TODO: Need to avoid circular import with assigner and sampler
# Type hint of config data
from typing import Dict, Any

ConfigType = Union[ConfigDict, dict, str, Dict[str, Any]]
OptConfigType = Optional[ConfigType]
# Type hint of one or more config data
MultiConfig = Union[ConfigType, list[ConfigType]]
OptMultiConfig = Optional[MultiConfig]

InstanceList = list[InstanceData]
OptInstanceList = Optional[InstanceList]

PixelList = list[PixelData]
OptPixelList = Optional[PixelList]

RangeType = Sequence[tuple[int, int]]
