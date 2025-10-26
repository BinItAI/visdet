# Copyright (c) OpenMMLab. All rights reserved.
"""Builder functions for task modules (assigners, samplers, coders)."""

from visdet.registry import TASK_UTILS


def build_assigner(cfg, **default_args):
    """Builder of bbox assigner.

    Args:
        cfg (dict): Config dict to build assigner.
        default_args (dict, optional): Defaults to construct assigner.

    Returns:
        Assigner: Built assigner.
    """
    return TASK_UTILS.build(cfg, default_args=default_args)


def build_sampler(cfg, **default_args):
    """Builder of bbox sampler.

    Args:
        cfg (dict): Config dict to build sampler.
        default_args (dict, optional): Defaults to construct sampler.

    Returns:
        Sampler: Built sampler.
    """
    return TASK_UTILS.build(cfg, default_args=default_args)


def build_bbox_coder(cfg, **default_args):
    """Builder of bbox coder.

    Args:
        cfg (dict): Config dict to build coder.
        default_args (dict, optional): Defaults to construct coder.

    Returns:
        Coder: Built bbox coder.
    """
    return TASK_UTILS.build(cfg, default_args=default_args)


__all__ = ["build_assigner", "build_sampler", "build_bbox_coder"]
