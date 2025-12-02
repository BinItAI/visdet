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

from visdet.engine.registry import DATA_SAMPLERS as VISENGINE_DATA_SAMPLERS
from visdet.engine.registry import DATASETS as VISENGINE_DATASETS
from visdet.engine.registry import EVALUATOR as VISENGINE_EVALUATOR
from visdet.engine.registry import HOOKS as VISENGINE_HOOKS
from visdet.engine.registry import LOG_PROCESSORS as VISENGINE_LOG_PROCESSORS
from visdet.engine.registry import LOOPS as VISENGINE_LOOPS
from visdet.engine.registry import METRICS as VISENGINE_METRICS
from visdet.engine.registry import MODEL_WRAPPERS as VISENGINE_MODEL_WRAPPERS
from visdet.engine.registry import MODELS as VISENGINE_MODELS
from visdet.engine.registry import OPTIM_WRAPPER_CONSTRUCTORS as VISENGINE_OPTIM_WRAPPER_CONSTRUCTORS
from visdet.engine.registry import OPTIM_WRAPPERS as VISENGINE_OPTIM_WRAPPERS
from visdet.engine.registry import OPTIMIZERS as VISENGINE_OPTIMIZERS
from visdet.engine.registry import PARAM_SCHEDULERS as VISENGINE_PARAM_SCHEDULERS
from visdet.engine.registry import RUNNER_CONSTRUCTORS as VISENGINE_RUNNER_CONSTRUCTORS
from visdet.engine.registry import RUNNERS as VISENGINE_RUNNERS
from visdet.engine.registry import TASK_UTILS as VISENGINE_TASK_UTILS
from visdet.engine.registry import TRANSFORMS as VISENGINE_TRANSFORMS
from visdet.engine.registry import VISBACKENDS as VISENGINE_VISBACKENDS
from visdet.engine.registry import VISUALIZERS as VISENGINE_VISUALIZERS
from visdet.engine.registry import WEIGHT_INITIALIZERS as VISENGINE_WEIGHT_INITIALIZERS
from visdet.engine.registry import Registry

# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry("runner", parent=VISENGINE_RUNNERS, locations=["visengine.runner"])
# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    "runner constructor", parent=VISENGINE_RUNNER_CONSTRUCTORS, locations=["visengine.runner"]
)
# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry("loop", parent=VISENGINE_LOOPS, locations=["visengine.runner"])
# manage all kinds of hooks like `CheckpointHook`
HOOKS = VISENGINE_HOOKS

# manage data-related modules
DATASETS = VISENGINE_DATASETS
DATA_SAMPLERS = VISENGINE_DATA_SAMPLERS
TRANSFORMS = VISENGINE_TRANSFORMS

# manage all kinds of modules inheriting `nn.Module`
MODELS = VISENGINE_MODELS
# manage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = VISENGINE_MODEL_WRAPPERS
# manage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = VISENGINE_WEIGHT_INITIALIZERS

# manage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = VISENGINE_OPTIMIZERS
# manage optimizer wrapper
OPTIM_WRAPPERS = VISENGINE_OPTIM_WRAPPERS
# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = VISENGINE_OPTIM_WRAPPER_CONSTRUCTORS
# manage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = VISENGINE_PARAM_SCHEDULERS
# manage all kinds of metrics
METRICS = VISENGINE_METRICS
# manage evaluator
EVALUATOR = VISENGINE_EVALUATOR

# manage task-specific modules like anchor generators and box coders
TASK_UTILS = VISENGINE_TASK_UTILS

# manage visualizer
VISUALIZERS = VISENGINE_VISUALIZERS
# manage visualizer backend
VISBACKENDS = VISENGINE_VISBACKENDS

# manage logprocessor
LOG_PROCESSORS = VISENGINE_LOG_PROCESSORS
