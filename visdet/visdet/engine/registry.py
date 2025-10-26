# ruff: noqa
# type: ignore
"""
Registry module for visdet.

This module provides access to the registry system for managing models,
datasets, hooks, and other components.
"""

from contextlib import contextmanager
from typing import Any, Generator, Optional


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
