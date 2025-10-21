# ruff: noqa
# fmt: off
# isort: skip
# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.
"""visdet provides 17 registry nodes to support using modules across
projects. Each node is a child of the root registry in visengine.

More details can be found at
https://mmengine.readthedocs.io/en/latest/tutorials/registry.html.
"""

from visengine.registry import DATA_SAMPLERS as VISENGINE_DATA_SAMPLERS
from visengine.registry import DATASETS as VISENGINE_DATASETS
from visengine.registry import EVALUATOR as VISENGINE_EVALUATOR
from visengine.registry import HOOKS as VISENGINE_HOOKS
from visengine.registry import LOG_PROCESSORS as VISENGINE_LOG_PROCESSORS
from visengine.registry import LOOPS as VISENGINE_LOOPS
from visengine.registry import METRICS as VISENGINE_METRICS
from visengine.registry import MODEL_WRAPPERS as VISENGINE_MODEL_WRAPPERS
from visengine.registry import MODELS as VISENGINE_MODELS
from visengine.registry import OPTIM_WRAPPER_CONSTRUCTORS as VISENGINE_OPTIM_WRAPPER_CONSTRUCTORS
from visengine.registry import OPTIM_WRAPPERS as VISENGINE_OPTIM_WRAPPERS
from visengine.registry import OPTIMIZERS as VISENGINE_OPTIMIZERS
from visengine.registry import PARAM_SCHEDULERS as VISENGINE_PARAM_SCHEDULERS
from visengine.registry import RUNNER_CONSTRUCTORS as VISENGINE_RUNNER_CONSTRUCTORS
from visengine.registry import RUNNERS as VISENGINE_RUNNERS
from visengine.registry import TASK_UTILS as VISENGINE_TASK_UTILS
from visengine.registry import TRANSFORMS as VISENGINE_TRANSFORMS
from visengine.registry import VISBACKENDS as VISENGINE_VISBACKENDS
from visengine.registry import VISUALIZERS as VISENGINE_VISUALIZERS
from visengine.registry import WEIGHT_INITIALIZERS as VISENGINE_WEIGHT_INITIALIZERS
from visengine.registry import Registry

# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry("runner", parent=VISENGINE_RUNNERS, locations=["visengine.runner"])
# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    "runner constructor", parent=VISENGINE_RUNNER_CONSTRUCTORS, locations=["visengine.runner"]
)
# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry("loop", parent=VISENGINE_LOOPS, locations=["visengine.runner"])
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry("hook", parent=VISENGINE_HOOKS, locations=["visdet.engine.hooks"])

# manage data-related modules
DATASETS = Registry("dataset", parent=VISENGINE_DATASETS, locations=["visdet.datasets"])
DATA_SAMPLERS = Registry(
    "data sampler", parent=VISENGINE_DATA_SAMPLERS, locations=["visdet.datasets.samplers"]
)
TRANSFORMS = Registry(
    "transform", parent=VISENGINE_TRANSFORMS, locations=["visdet.datasets.transforms", "viscv.transforms"]
)

# manage all kinds of modules inheriting `nn.Module`
MODELS = Registry("model", parent=VISENGINE_MODELS, locations=["visdet.models"])
# manage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry(
    "model_wrapper", parent=VISENGINE_MODEL_WRAPPERS, locations=["visdet.models"]
)
# manage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry(
    "weight initializer", parent=VISENGINE_WEIGHT_INITIALIZERS, locations=["visdet.models"]
)

# manage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry(
    "optimizer", parent=VISENGINE_OPTIMIZERS, locations=["visdet.engine.optimizers"]
)
# manage optimizer wrapper
OPTIM_WRAPPERS = Registry(
    "optim_wrapper", parent=VISENGINE_OPTIM_WRAPPERS, locations=["visdet.engine.optimizers"]
)
# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    "optimizer constructor",
    parent=VISENGINE_OPTIM_WRAPPER_CONSTRUCTORS,
    locations=["visdet.engine.optimizers"],
)
# manage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry(
    "parameter scheduler", parent=VISENGINE_PARAM_SCHEDULERS, locations=["visengine.optim.scheduler.param_scheduler"]
)
# manage all kinds of metrics
METRICS = Registry("metric", parent=VISENGINE_METRICS, locations=["visdet.evaluation"])
# manage evaluator
EVALUATOR = Registry("evaluator", parent=VISENGINE_EVALUATOR, locations=["visdet.evaluation"])

# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry("task util", parent=VISENGINE_TASK_UTILS, locations=["visdet.models"])

# manage visualizer
VISUALIZERS = Registry("visualizer", parent=VISENGINE_VISUALIZERS, locations=["visdet.visualization"])
# manage visualizer backend
VISBACKENDS = Registry(
    "vis_backend", parent=VISENGINE_VISBACKENDS, locations=["visdet.visualization"]
)

# manage logprocessor
LOG_PROCESSORS = Registry(
    "log_processor",
    parent=VISENGINE_LOG_PROCESSORS,
    # TODO: update the location when visdet has its own log processor
    locations=["visdet.engine"],
)
