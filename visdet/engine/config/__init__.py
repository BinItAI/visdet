# ruff: noqa
# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.
from .config import Config, ConfigDict, DictAction, read_base
from .yaml_loader import load_yaml_config

__all__ = ["Config", "ConfigDict", "DictAction", "read_base", "load_yaml_config"]
