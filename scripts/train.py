#!/usr/bin/env python3
"""
Simple training script using MMEngine Runner pattern.

This script demonstrates the recommended way to train models using visdet,
following the MMEngine tutorial pattern with explicit Runner initialization.

Usage:
    python scripts/train.py path/to/your_config.yaml

For distributed training or advanced options, use tools/train.py instead.
"""

import argparse

from visdet.engine.config import Config
from visdet.engine.runner import Runner
from visdet.registry import DATASETS, EVALUATOR, METRICS, MODELS


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a detector using MMEngine Runner pattern")
    parser.add_argument("config", help="Train config file path")
    args = parser.parse_args()
    return args


def build_dataloader(dataset_cfg, cfg):
    """Build dataloader from dataset config.

    Args:
        dataset_cfg: Dataset configuration dict
        cfg: Full config object

    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader

    # Build dataset using registry
    dataset = DATASETS.build(dataset_cfg)

    # Create dataloader (simplified - uses config values)
    from visdet.engine.dataset import pseudo_collate

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 2),
        shuffle=True,
        num_workers=cfg.get("num_workers", 2),
        persistent_workers=True if cfg.get("num_workers", 2) > 0 else False,
        collate_fn=pseudo_collate,
    )

    return dataloader


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration file
    cfg = Config.fromfile(args.config)

    # Build Model
    # Models must inherit from BaseModel and implement forward with mode argument
    print("Building model...")
    model = MODELS.build(cfg.model)

    # Build Train Dataloader
    print("Building train dataloader...")
    train_dataloader = build_dataloader(cfg.data.train, cfg)

    # Build Validation Dataloader (optional)
    val_dataloader = None
    if hasattr(cfg.data, "val"):
        print("Building validation dataloader...")
        val_dataloader = build_dataloader(cfg.data.val, cfg)

    # Build Evaluation Metrics
    val_evaluator = None
    if cfg.get("val_evaluator"):
        print("Building evaluator from val_evaluator config...")
        val_evaluator = EVALUATOR.build(cfg.val_evaluator)
    elif cfg.get("evaluation"):
        print("Building evaluator from evaluation config...")
        # Build evaluator from config
        if isinstance(cfg.evaluation.get("metric"), list):
            # Multiple metrics
            val_evaluator = [METRICS.build(dict(type=m)) for m in cfg.evaluation.metric]
        else:
            # Single metric
            val_evaluator = METRICS.build(dict(type=cfg.evaluation.metric))

    # Build and Run the Runner
    # This follows the MMEngine pattern shown in:
    # https://mmengine.readthedocs.io/en/latest/get_started/15_minutes.html
    print("Creating Runner...")
    runner = Runner(
        # Model (must inherit from BaseModel with forward(mode='loss'|'predict'))
        model=model,
        # Working directory for logs and checkpoints
        work_dir=cfg.work_dir,
        # Train dataloader (PyTorch DataLoader)
        train_dataloader=train_dataloader,
        # Optimizer wrapper config
        optim_wrapper=dict(
            optimizer=dict(type=cfg.optimizer.type, lr=cfg.optimizer.lr, **cfg.optimizer.get("optimizer_config", {}))
        ),
        # Training configuration (epochs, validation interval, etc.)
        train_cfg=dict(
            by_epoch=True,
            max_epochs=cfg.get("total_epochs", cfg.runner.get("max_epochs", 12)),
            val_interval=cfg.get("evaluation", {}).get("interval", 1) if cfg.get("evaluation") else 1,
        ),
        # Validation dataloader (optional)
        val_dataloader=val_dataloader,
        # Validation configuration
        val_cfg=dict() if val_dataloader else None,
        # Validation evaluator (Metrics that inherit from BaseMetric)
        val_evaluator=val_evaluator,
        default_scope="visdet",
    )

    # Start training
    print("Starting training...")
    runner.train()


if __name__ == "__main__":
    main()
