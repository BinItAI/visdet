# ruff: noqa
# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa

# Import version first to ensure it's available
from visdet.version import __version__, version_info

# Import Config for convenience
from visdet.engine.config import Config, ConfigDict
from visdet.engine.registry import DefaultScope

# Re-export version info explicitly at module level
globals()["__version__"] = __version__
globals()["version_info"] = version_info

# Re-export config utilities
from .config import Config
