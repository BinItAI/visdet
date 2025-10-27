# ruff: noqa
# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.
from visdet.engine._strategy.base import BaseStrategy
from visdet.engine._strategy.distributed import DDPStrategy
from visdet.engine._strategy.single_device import SingleDeviceStrategy

__all__ = [
    "BaseStrategy",
    "DDPStrategy",
    "SingleDeviceStrategy",
]
