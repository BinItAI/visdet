"""Distributed training utilities and decorators.

This module provides convenient decorators and helpers for distributed training,
helping centralize rank-specific operations and reduce boilerplate.
"""

import functools
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def rank_0_only(func: F) -> F:
    """Decorator that ensures a function only executes on rank 0."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        from visdet.engine.dist import is_main_process

        if not is_main_process():
            return None

        return func(*args, **kwargs)

    return wrapper  # type: ignore


def rank_0_only_method(func: F) -> F:
    """Decorator for class methods that should only execute on rank 0."""

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        from visdet.engine.dist import is_main_process

        if not is_main_process():
            return None

        return func(self, *args, **kwargs)

    return wrapper  # type: ignore


def broadcast_from_rank_0(obj: Any) -> Any:
    """Broadcast a picklable object from rank 0 to all ranks."""

    from visdet.engine.dist import broadcast_object_list, get_world_size

    if get_world_size() == 1:
        return obj

    obj_list = [obj]
    broadcast_object_list(obj_list, src=0)
    return obj_list[0]
