"""Distributed Data Parallel (DDP) training utilities for visdet.

This module provides automatic DDP setup that detects multiple GPUs and
automatically spawns worker processes without requiring manual torchrun commands.

Example:
    >>> from visdet.ddp import auto_ddp_train
    >>> auto_ddp_train(
    ...     model='mask_rcnn_swin_s',
    ...     dataset='cmr_instance_segmentation',
    ...     epochs=12
    ... )
"""

from visdet.ddp.auto_ddp import auto_ddp_train, setup_distributed_env

__all__ = ["auto_ddp_train", "setup_distributed_env"]
