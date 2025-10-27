# ruff: noqa
from typing import Union, Dict, Any
from pathlib import Path

# ConfigType is used for typing config objects
ConfigType = Union[str, Path, Dict[str, Any]]

from visdet.utils.misc import get_test_pipeline_cfg
from visdet.utils.typing_utils import (
    InstanceList,
    MultiConfig,
    OptConfigType,
    OptInstanceList,
    OptMultiConfig,
)
from visdet.utils.setup_env import register_all_modules

__all__ = [
    "ConfigType",
    "get_test_pipeline_cfg",
    "OptInstanceList",
    "InstanceList",
    "OptMultiConfig",
    "OptConfigType",
    "MultiConfig",
    "register_all_modules",
]
