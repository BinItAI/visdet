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

    def register_module(
        self,
        name: str = None,
        module: Any = None,
        force: bool = False,
    ) -> Any:
        """Register a module with a given name.

        Can be used as:
        - @registry.register_module() - uses class __name__
        - @registry.register_module(name="custom_name") - uses provided name
        - registry.register_module(name="name", module=obj) - direct registration
        """

        # Case 1: Direct registration with module argument
        if module is not None:
            key = name if name is not None else module.__name__
            self[key] = module
            return module

        # Case 2: Decorator usage - could be:
        # @registry.register_module() - name will be None
        # @registry.register_module(name="custom") - name will be string
        # or even @registry.register_module("CustomName") - name could be string (class passed as name arg)

        def _register(obj: Any) -> Any:
            # If name was provided as a string, use it; otherwise use obj's name
            key = name if isinstance(name, str) else obj.__name__
            self[key] = obj
            return obj

        # If name is None or a string, return decorator
        # If name is actually a class (used as @registry.register_module without parens), register it directly
        if name is not None and not isinstance(name, str):
            # name is actually a class being decorated
            return _register(name)

        return _register

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
