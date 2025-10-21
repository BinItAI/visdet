# ruff: noqa
# Copyright (c) OpenMMLab. All rights reserved.
import visdet.cv as viscv
import visdet.engine as visengine
from visdet.engine.utils import digit_version

from .version import __version__, version_info

# Import models to register components
from . import models

# Import engine to register hooks (also re-exports visengine)
from . import engine

# Import cv (re-exports viscv)
from . import cv

# Import visualization to register components
from . import visualization

# Import datasets to register dataset classes
from . import datasets

# Import presets for string-based configuration
from . import presets

# Import SimpleRunner for easy training API
from .runner import Runner, SimpleRunner
