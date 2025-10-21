#!/usr/bin/env python3
"""
Simple training script using MMEngine Runner pattern.

This script demonstrates the recommended way to train models using visdet,
following the MMEngine tutorial pattern with explicit Runner initialization.

Usage:
    python scripts/train.py configs/your_config.py

For distributed training or advanced options, use tools/train.py instead.
"""

import argparse

from visdet.engine import Config, Runner


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a detector using MMEngine Runner pattern")
    parser.add_argument("config", help="Train config file path")
    args = parser.parse_args()
    return args


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration file
    cfg = Config.fromfile(args.config)

    # Create Runner instance with explicit initialization
    # This follows the MMEngine pattern shown in:
    # https://mmengine.readthedocs.io/en/latest/get_started/15_minutes.html
    runner = Runner(
        model=cfg.model,
        work_dir=cfg.work_dir,
        train_dataloader=cfg.train_dataloader,
        optim_wrapper=cfg.optim_wrapper,
        train_cfg=cfg.train_cfg,
        val_dataloader=cfg.get("val_dataloader"),
        val_cfg=cfg.get("val_cfg"),
        val_evaluator=cfg.get("val_evaluator"),
    )

    # Start training
    runner.train()


if __name__ == "__main__":
    main()
