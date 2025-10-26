#!/usr/bin/env python3
"""Simple training script for visdet with minimal configuration.

This script provides a user-friendly entry point for training object detection
and instance segmentation models using the visdet framework.

Example:
    Train Mask R-CNN on CMR dataset:
    $ python scripts/train_simple.py \\
        --model mask_rcnn_swin_s \\
        --dataset cmr_instance_segmentation \\
        --epochs 12 \\
        --work-dir ./work_dirs/cmr_training

    List available presets:
    $ python scripts/train_simple.py --list-models
    $ python scripts/train_simple.py --list-datasets
"""

import argparse
from pathlib import Path

# Import visdet to trigger all registrations
import visdet

# Now import SimpleRunner after registrations
from visdet import SimpleRunner


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Simple training script for visdet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Training arguments
    parser.add_argument(
        "--model",
        type=str,
        help="Model preset name (e.g., 'mask_rcnn_swin_s')",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset preset name (e.g., 'coco_instance_segmentation', 'cmr_instance_segmentation')",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw_default",
        help="Optimizer preset name (default: 'adamw_default')",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        help="Scheduler preset name (optional)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Number of training epochs (default: 12)",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="./work_dirs",
        help="Working directory for logs and checkpoints (default: './work_dirs')",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=1,
        help="Validation interval in epochs (default: 1)",
    )

    # Discovery arguments
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available model presets and exit",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available dataset presets and exit",
    )
    parser.add_argument(
        "--list-optimizers",
        action="store_true",
        help="List all available optimizer presets and exit",
    )
    parser.add_argument(
        "--list-schedulers",
        action="store_true",
        help="List all available scheduler presets and exit",
    )
    parser.add_argument(
        "--show-preset",
        type=str,
        help="Show configuration for a specific preset (use with --category)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="model",
        choices=["model", "dataset", "optimizer", "scheduler", "pipeline"],
        help="Category for --show-preset (default: 'model')",
    )

    # DDP arguments
    parser.add_argument(
        "--ddp-enabled",
        action="store_true",
        help="Enable automatic DDP (Distributed Data Parallel) for multi-GPU training. "
        "Automatically detects GPUs and spawns processes. Single-GPU training uses no DDP.",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Handle discovery commands
    if args.list_models:
        print("Available model presets:")
        for name in SimpleRunner.list_models():
            print(f"  - {name}")
        return

    if args.list_datasets:
        print("Available dataset presets:")
        for name in SimpleRunner.list_datasets():
            print(f"  - {name}")
        return

    if args.list_optimizers:
        print("Available optimizer presets:")
        for name in SimpleRunner.list_optimizers():
            print(f"  - {name}")
        return

    if args.list_schedulers:
        print("Available scheduler presets:")
        for name in SimpleRunner.list_schedulers():
            print(f"  - {name}")
        return

    if args.show_preset:
        SimpleRunner.show_preset(args.show_preset, args.category)
        return

    # Validate required arguments for training
    if not args.model:
        print("Error: --model is required for training")
        print("Run with --list-models to see available options")
        return

    if not args.dataset:
        print("Error: --dataset is required for training")
        print("Run with --list-datasets to see available options")
        return

    # Print training configuration
    print(f"\n{'=' * 60}")
    print("Training Configuration")
    print(f"{'=' * 60}")
    print(f"Model:        {args.model}")
    print(f"Dataset:      {args.dataset}")
    print(f"Optimizer:    {args.optimizer}")
    print(f"Scheduler:    {args.scheduler or 'None'}")
    print(f"Epochs:       {args.epochs}")
    print(f"Work Dir:     {args.work_dir}")
    print(f"Val Interval: {args.val_interval}")
    if args.ddp_enabled:
        import torch

        gpu_count = torch.cuda.device_count()
        print(f"DDP Mode:     Enabled (detected {gpu_count} GPU(s))")
    else:
        print("DDP Mode:     Disabled")
    print(f"{'=' * 60}\n")

    # Define training function for DDP
    def create_and_train() -> None:
        """Create runner and start training."""
        runner = SimpleRunner(
            model=args.model,
            dataset=args.dataset,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            work_dir=args.work_dir,
            epochs=args.epochs,
            val_interval=args.val_interval,
        )
        runner.train()

    # Launch training with optional DDP
    if args.ddp_enabled:
        from visdet.ddp import auto_ddp_train

        auto_ddp_train(create_and_train)
    else:
        # Direct training without DDP
        create_and_train()

    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"{'=' * 60}")
    print(f"Results saved to: {args.work_dir}")


if __name__ == "__main__":
    main()
