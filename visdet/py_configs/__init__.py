"""Python-first configuration API for visdet.

This package provides a programmatic way to create configurations
with full IDE autocomplete and type safety.

Key Features:
1. Builder functions for creating component configs
2. Pre-defined experiment presets
3. Sweep generators for hyperparameter search
4. Full IDE autocomplete via Pydantic models

Example:
    >>> from visdet.py_configs import mask_rcnn_swin_tiny_coco
    >>> # Get a complete experiment config with one line
    >>> cfg = mask_rcnn_swin_tiny_coco(data_root='/data/coco')
    >>>
    >>> # Customize with full autocomplete
    >>> cfg.train_cfg.max_epochs = 24
    >>> cfg.optim_wrapper.optimizer.lr = 2e-4
    >>>
    >>> # Use with SimpleRunner
    >>> from visdet import SimpleRunner
    >>> runner = SimpleRunner(config=cfg)
    >>> runner.train()
"""

# Builders - factory functions for components
from visdet.py_configs.builders import (
    # Backbones
    swin_tiny,
    swin_small,
    swin_base,
    resnet50,
    resnet101,
    # Necks
    fpn_for_swin,
    fpn_for_resnet,
    # Heads
    standard_rpn_head,
    standard_roi_head,
    # Models
    mask_rcnn,
    # Optimizers
    adamw_default,
    one_cycle_scheduler,
    # Data
    coco_train_pipeline,
    coco_test_pipeline,
    coco_dataset,
    train_dataloader,
    val_dataloader,
)

# Presets - ready-to-use experiment configs
from visdet.py_configs.presets import (
    mask_rcnn_swin_tiny_coco,
    mask_rcnn_swin_small_coco,
    # Sweeps
    lr_sweep,
    batch_size_sweep,
)

__all__ = [
    # Backbone builders
    "swin_tiny",
    "swin_small",
    "swin_base",
    "resnet50",
    "resnet101",
    # Neck builders
    "fpn_for_swin",
    "fpn_for_resnet",
    # Head builders
    "standard_rpn_head",
    "standard_roi_head",
    # Model builders
    "mask_rcnn",
    # Optimizer builders
    "adamw_default",
    "one_cycle_scheduler",
    # Data builders
    "coco_train_pipeline",
    "coco_test_pipeline",
    "coco_dataset",
    "train_dataloader",
    "val_dataloader",
    # Presets
    "mask_rcnn_swin_tiny_coco",
    "mask_rcnn_swin_small_coco",
    # Sweeps
    "lr_sweep",
    "batch_size_sweep",
]
