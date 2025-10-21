# ruff: noqa
# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa

# Import version first to ensure it's available
from .version import __version__, version_info

# Import other modules
from .config import *
from .fileio import *
from .logging import *
from .registry import *
from .utils import *

# Re-export version info explicitly at module level
globals()["__version__"] = __version__
globals()["version_info"] = version_info
