# Copyright (c) OpenMMLab. All rights reserved.
"""Utility functions for distributed training."""

import functools


def master_only(func):
    """Decorator to make a function only execute on master process."""
    from visdet.engine.dist import is_main_process

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
    return wrapper


__all__ = ['master_only']
