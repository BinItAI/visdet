# ruff: noqa
# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.
from visdet.engine.utils.dl_utils import TORCH_VERSION
from visdet.engine.utils.version_utils import digit_version
from visdet.engine.model.wrappers.distributed import MMDataParallel, MMDistributedDataParallel
from visdet.engine.model.wrappers.seperate_distributed import MMSeparateDistributedDataParallel
from visdet.engine.model.wrappers.utils import is_model_wrapper

__all__ = [
    "MMDataParallel",
    "MMDistributedDataParallel",
    "MMSeparateDistributedDataParallel",
    "is_model_wrapper",
]

from visdet.engine.model.wrappers.fully_sharded_distributed import MMFullyShardedDataParallel

__all__.append("MMFullyShardedDataParallel")
