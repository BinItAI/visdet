# ruff: noqa
# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.
from visdet.engine.utils.dl_utils import TORCH_VERSION
from visdet.engine.utils.version_utils import digit_version
from .base_model import BaseDataPreprocessor, BaseModel, ImgDataPreprocessor
from .base_module import BaseModule, ModuleDict, ModuleList, Sequential
from .utils import (
    convert_sync_batchnorm,
    detect_anomalous_params,
    merge_dict,
    revert_sync_batchnorm,
    stack_batch,
)
from .weight_init import (
    BaseInit,
    Caffe2XavierInit,
    ConstantInit,
    KaimingInit,
    NormalInit,
    PretrainedInit,
    TruncNormalInit,
    UniformInit,
    XavierInit,
    bias_init_with_prob,
    caffe2_xavier_init,
    constant_init,
    initialize,
    kaiming_init,
    normal_init,
    trunc_normal_init,
    uniform_init,
    update_init_info,
    xavier_init,
)
from .wrappers import (
    MMDistributedDataParallel,
    MMSeparateDistributedDataParallel,
    is_model_wrapper,
)

__all__ = [
    "BaseDataPreprocessor",
    "BaseInit",
    "BaseModel",
    "BaseModule",
    "Caffe2XavierInit",
    "ConstantInit",
    # "ExponentialMovingAverage",  # Not imported - EMA is in hooks module
    "ImgDataPreprocessor",
    "KaimingInit",
    "MMDistributedDataParallel",
    "MMSeparateDistributedDataParallel",
    "ModuleDict",
    "ModuleList",
    # "MomentumAnnealingEMA",  # Not imported - EMA is in hooks module
    "NormalInit",
    "PretrainedInit",
    "Sequential",
    "TruncNormalInit",
    "UniformInit",
    "XavierInit",
    "bias_init_with_prob",
    "caffe2_xavier_init",
    "constant_init",
    "convert_sync_batchnorm",
    "detect_anomalous_params",
    "initialize",
    "is_model_wrapper",
    "kaiming_init",
    "merge_dict",
    "normal_init",
    "revert_sync_batchnorm",
    "stack_batch",
    "trunc_normal_init",
    "uniform_init",
    "update_init_info",
    "xavier_init",
]

from .wrappers import MMFullyShardedDataParallel

__all__.append("MMFullyShardedDataParallel")
