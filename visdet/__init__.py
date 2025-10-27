# ruff: noqa
# Copyright (c) OpenMMLab. All rights reserved.
from visdet.engine.utils import digit_version
from visdet.version import __version__, version_info

# Import models to register components
from visdet import cv, datasets, engine, models, presets, visualization

# Explicitly import hooks to register them (not done automatically by engine)
from visdet.engine import hooks
from visdet.runner import Runner, SimpleRunner
