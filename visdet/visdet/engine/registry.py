# ruff: noqa
# type: ignore
"""
Registry module for visdet.

This module provides access to the registry system for managing models,
datasets, hooks, and other components.
"""

from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional


class Registry(dict):
    """Stub Registry class for visdet.

    This is a minimal implementation for type checking.
    In a full implementation, this would provide enhanced registry features.
    """

    def __init__(self, name: str = "", parent: Optional["Registry"] = None, locations: Optional[list] = None) -> None:
        """Initialize registry."""
        super().__init__()
        self.name = name
        self.parent = parent
        self.locations = locations or []

    def register(self, cls_or_func: Any = None, force: bool = False) -> Any:
        """Register a class or function."""

        def _register(obj: Any) -> Any:
            self[obj.__name__] = obj
            return obj

        if cls_or_func is None:
            return _register
        return _register(cls_or_func)

    def build(self, cfg: Dict) -> Any:
        """Build object from config."""
        if isinstance(cfg, dict):
            cfg = cfg.copy()
            obj_type = cfg.pop("type")
            return self[obj_type](**cfg)
        return cfg


# Create stub registry instances
DATA_SAMPLERS = Registry("data_sampler")
DATASETS = Registry("dataset")
EVALUATOR = Registry("evaluator")
HOOKS = Registry("hook")
LOG_PROCESSORS = Registry("log_processor")
LOOPS = Registry("loop")
METRICS = Registry("metric")
MODEL_WRAPPERS = Registry("model_wrapper")
MODELS = Registry("model")
OPTIM_WRAPPER_CONSTRUCTORS = Registry("optimizer_constructor")
OPTIM_WRAPPERS = Registry("optim_wrapper")
OPTIMIZERS = Registry("optimizer")
PARAM_SCHEDULERS = Registry("parameter_scheduler")
RUNNER_CONSTRUCTORS = Registry("runner_constructor")
RUNNERS = Registry("runner")
TASK_UTILS = Registry("task_util")
TRANSFORMS = Registry("transform")
VISBACKENDS = Registry("vis_backend")
VISUALIZERS = Registry("visualizer")
WEIGHT_INITIALIZERS = Registry("weight_initializer")


class DefaultScope:
    """Stub implementation of DefaultScope for registry management.

    This is a simplified version for the type checking phase.
    In a full implementation, this would come from mmengine.
    """

    _current_instance: Optional[str] = None
    _created_instances: set = set()

    def __init__(self, scope_name: str) -> None:
        """Initialize a DefaultScope instance."""
        self.scope_name = scope_name
        DefaultScope._created_instances.add(scope_name)

    @classmethod
    def get_instance(cls, instance_name: str, scope_name: str = "") -> "DefaultScope":
        """Get or create a DefaultScope instance."""
        cls._created_instances.add(scope_name)
        return cls(scope_name)

    @classmethod
    def get_current_instance(cls) -> Optional["DefaultScope"]:
        """Get the current DefaultScope instance."""
        if cls._current_instance is not None:
            return cls(cls._current_instance)
        return None

    @classmethod
    def check_instance_created(cls, scope_name: str) -> bool:
        """Check if a scope instance has been created."""
        return scope_name in cls._created_instances

    @classmethod
    @contextmanager
    def overwrite_default_scope(cls, scope_name: str) -> Generator[None, None, None]:
        """Context manager to temporarily set the default scope."""
        old_instance = cls._current_instance
        cls._current_instance = scope_name
        try:
            yield
        finally:
            cls._current_instance = old_instance


__all__ = ["DefaultScope"]
