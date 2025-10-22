# ruff: noqa
# Copyright (c) OpenMMLab. All rights reserved.
from visdet.engine.utils import digit_version

from .version import __version__, version_info

# Import models to register components
from . import models

# Import engine to register hooks
from . import engine

# Explicitly import hooks to register them (not done automatically by engine)
from .engine import hooks

# Import cv
from . import cv

# Import visualization to register components
from . import visualization

# Import datasets to register dataset classes
from . import datasets

# Import presets for string-based configuration
from . import presets

# Import SimpleRunner for easy training API
from .runner import Runner, SimpleRunner
