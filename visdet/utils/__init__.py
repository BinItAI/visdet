# ruff: noqa
from typing import Union, Dict, Any
from pathlib import Path

# ConfigType is used for typing config objects
ConfigType = Union[str, Path, Dict[str, Any]]

from visdet.utils.misc import get_test_pipeline_cfg, reduce_mean
from visdet.utils.typing_utils import (
    InstanceList,
    MultiConfig,
    OptConfigType,
    OptInstanceList,
    OptMultiConfig,
)
from visdet.utils.setup_env import register_all_modules

# FiftyOne utilities (optional - fiftyone must be installed)
from visdet.utils.fiftyone_utils import (
    detections_to_fiftyone,
    load_inference_results,
    add_predictions_to_dataset,
    create_coco_dataset,
    visualize_results,
)

__all__ = [
    "ConfigType",
    "get_test_pipeline_cfg",
    "reduce_mean",
    "OptInstanceList",
    "InstanceList",
    "OptMultiConfig",
    "OptConfigType",
    "MultiConfig",
    "register_all_modules",
    # FiftyOne utilities
    "detections_to_fiftyone",
    "load_inference_results",
    "add_predictions_to_dataset",
    "create_coco_dataset",
    "visualize_results",
]
