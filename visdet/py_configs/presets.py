"""Pre-defined experiment configurations.

This module provides ready-to-use experiment configs for common setups.
All presets return ExperimentConfig objects with full IDE autocomplete.

Example:
    >>> from visdet.py_configs import mask_rcnn_swin_tiny_coco
    >>> cfg = mask_rcnn_swin_tiny_coco(data_root='/data/coco')
    >>> # Customize with autocomplete
    >>> cfg.train_cfg.max_epochs = 24
    >>> cfg.optim_wrapper.optimizer.lr = 2e-4
"""

from typing import Optional

from visdet.py_configs.builders import (
    adamw_default,
    coco_dataset,
    coco_test_pipeline,
    coco_train_pipeline,
    mask_rcnn,
    one_cycle_scheduler,
    swin_base,
    swin_small,
    swin_tiny,
    train_dataloader,
    val_dataloader,
)
from visdet.schemas import (
    EpochBasedTrainLoopConfig,
    ExperimentConfig,
    ValLoopConfig,
)


def mask_rcnn_swin_tiny_coco(
    data_root: str = "/data/coco",
    train_ann_file: str = "annotations/instances_train2017.json",
    val_ann_file: Optional[str] = "annotations/instances_val2017.json",
    train_img_prefix: str = "train2017",
    val_img_prefix: str = "val2017",
    num_classes: int = 80,
    batch_size: int = 2,
    num_workers: int = 2,
    max_epochs: int = 12,
    lr: float = 1e-4,
    work_dir: str = "./work_dirs/mask_rcnn_swin_tiny_coco",
) -> ExperimentConfig:
    """Mask R-CNN with Swin-Tiny backbone on COCO.

    Standard configuration for instance segmentation training.

    Args:
        data_root: COCO data root directory.
        train_ann_file: Training annotation file path.
        val_ann_file: Validation annotation file path (None to skip validation).
        train_img_prefix: Training image directory prefix.
        val_img_prefix: Validation image directory prefix.
        num_classes: Number of object classes.
        batch_size: Batch size per GPU.
        num_workers: Data loading workers.
        max_epochs: Maximum training epochs.
        lr: Learning rate.
        work_dir: Output directory.

    Returns:
        Complete ExperimentConfig ready for training.

    Example:
        >>> cfg = mask_rcnn_swin_tiny_coco(
        ...     data_root='/my/coco',
        ...     max_epochs=24,
        ...     lr=2e-4
        ... )
        >>> from visdet import SimpleRunner
        >>> runner = SimpleRunner(config=cfg)
        >>> runner.train()
    """
    # Model
    model = mask_rcnn(backbone=swin_tiny(), num_classes=num_classes)

    # Training data
    train_dataset = coco_dataset(
        data_root=data_root,
        ann_file=train_ann_file,
        img_prefix=train_img_prefix,
        pipeline=coco_train_pipeline(),
    )
    train_dl = train_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Validation data
    val_dl = None
    val_cfg = None
    val_evaluator = None
    if val_ann_file:
        val_dataset = coco_dataset(
            data_root=data_root,
            ann_file=val_ann_file,
            img_prefix=val_img_prefix,
            pipeline=coco_test_pipeline(),
            test_mode=True,
        )
        val_dl = val_dataloader(
            dataset=val_dataset,
            batch_size=1,
            num_workers=num_workers,
        )
        val_cfg = ValLoopConfig()
        val_evaluator = {
            "type": "CocoMetric",
            "ann_file": f"{data_root}/{val_ann_file}",
            "metric": ["bbox", "segm"],
        }

    return ExperimentConfig(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        optim_wrapper=adamw_default(lr=lr),
        param_scheduler=one_cycle_scheduler(max_lr=lr * 10),
        train_cfg=EpochBasedTrainLoopConfig(max_epochs=max_epochs, val_interval=1),
        val_cfg=val_cfg,
        val_evaluator=val_evaluator,
        work_dir=work_dir,
        default_hooks={
            "timer": {"type": "IterTimerHook"},
            "logger": {"type": "LoggerHook", "interval": 50},
            "param_scheduler": {"type": "ParamSchedulerHook"},
            "checkpoint": {"type": "CheckpointHook", "interval": 1},
            "sampler_seed": {"type": "DistSamplerSeedHook"},
            "visualization": {"type": "DetVisualizationHook"},
        },
    )


def mask_rcnn_swin_small_coco(
    data_root: str = "/data/coco",
    train_ann_file: str = "annotations/instances_train2017.json",
    val_ann_file: Optional[str] = "annotations/instances_val2017.json",
    train_img_prefix: str = "train2017",
    val_img_prefix: str = "val2017",
    num_classes: int = 80,
    batch_size: int = 2,
    num_workers: int = 2,
    max_epochs: int = 12,
    lr: float = 1e-4,
    work_dir: str = "./work_dirs/mask_rcnn_swin_small_coco",
) -> ExperimentConfig:
    """Mask R-CNN with Swin-Small backbone on COCO.

    Deeper than Swin-Tiny (18 blocks vs 6 in stage 3).

    Args:
        data_root: COCO data root directory.
        train_ann_file: Training annotation file path.
        val_ann_file: Validation annotation file path.
        train_img_prefix: Training image directory prefix.
        val_img_prefix: Validation image directory prefix.
        num_classes: Number of object classes.
        batch_size: Batch size per GPU.
        num_workers: Data loading workers.
        max_epochs: Maximum training epochs.
        lr: Learning rate.
        work_dir: Output directory.

    Returns:
        Complete ExperimentConfig.
    """
    # Model with Swin-Small backbone
    model = mask_rcnn(backbone=swin_small(), num_classes=num_classes)

    # Training data
    train_dataset = coco_dataset(
        data_root=data_root,
        ann_file=train_ann_file,
        img_prefix=train_img_prefix,
        pipeline=coco_train_pipeline(),
    )
    train_dl = train_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Validation data
    val_dl = None
    val_cfg = None
    val_evaluator = None
    if val_ann_file:
        val_dataset = coco_dataset(
            data_root=data_root,
            ann_file=val_ann_file,
            img_prefix=val_img_prefix,
            pipeline=coco_test_pipeline(),
            test_mode=True,
        )
        val_dl = val_dataloader(
            dataset=val_dataset,
            batch_size=1,
            num_workers=num_workers,
        )
        val_cfg = ValLoopConfig()
        val_evaluator = {
            "type": "CocoMetric",
            "ann_file": f"{data_root}/{val_ann_file}",
            "metric": ["bbox", "segm"],
        }

    return ExperimentConfig(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        optim_wrapper=adamw_default(lr=lr),
        param_scheduler=one_cycle_scheduler(max_lr=lr * 10),
        train_cfg=EpochBasedTrainLoopConfig(max_epochs=max_epochs, val_interval=1),
        val_cfg=val_cfg,
        val_evaluator=val_evaluator,
        work_dir=work_dir,
        default_hooks={
            "timer": {"type": "IterTimerHook"},
            "logger": {"type": "LoggerHook", "interval": 50},
            "param_scheduler": {"type": "ParamSchedulerHook"},
            "checkpoint": {"type": "CheckpointHook", "interval": 1},
            "sampler_seed": {"type": "DistSamplerSeedHook"},
            "visualization": {"type": "DetVisualizationHook"},
        },
    )


# =============================================================================
# Sweep Generators
# =============================================================================


def lr_sweep(
    base_preset_fn,
    learning_rates: list[float] | None = None,
    **preset_kwargs,
):
    """Generate configs for learning rate sweep.

    Args:
        base_preset_fn: Preset function to use as base (e.g., mask_rcnn_swin_tiny_coco).
        learning_rates: List of learning rates to try.
        **preset_kwargs: Arguments passed to preset function.

    Yields:
        ExperimentConfig for each learning rate.

    Example:
        >>> from visdet.py_configs import mask_rcnn_swin_tiny_coco, lr_sweep
        >>> for cfg in lr_sweep(
        ...     mask_rcnn_swin_tiny_coco,
        ...     learning_rates=[1e-5, 1e-4, 1e-3],
        ...     data_root='/data/coco'
        ... ):
        ...     # Train each config
        ...     SimpleRunner(config=cfg).train()
    """
    if learning_rates is None:
        learning_rates = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]

    for lr in learning_rates:
        cfg = base_preset_fn(lr=lr, **preset_kwargs)
        cfg.work_dir = f"{cfg.work_dir}_lr{lr}"
        yield cfg


def batch_size_sweep(
    base_preset_fn,
    batch_sizes: list[int] | None = None,
    **preset_kwargs,
):
    """Generate configs for batch size sweep.

    Args:
        base_preset_fn: Preset function to use as base.
        batch_sizes: List of batch sizes to try.
        **preset_kwargs: Arguments passed to preset function.

    Yields:
        ExperimentConfig for each batch size.
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]

    for bs in batch_sizes:
        cfg = base_preset_fn(batch_size=bs, **preset_kwargs)
        cfg.work_dir = f"{cfg.work_dir}_bs{bs}"
        yield cfg
