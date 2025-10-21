#!/usr/bin/env python3
"""
Simple Training Script for visdet

A user-friendly wrapper around the main training script that provides:
- Single-GPU training with sensible defaults
- Minimal CLI for common use cases
- Clear progress indicators and error messages
- Easy configuration through both config files and CLI flags

Example usage:
    # Train with a config file
    python scripts/train_simple.py --config configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py

    # Override common settings
    python scripts/train_simple.py \\
        --config configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \\
        --data-root /path/to/coco \\
        --epochs 24 \\
        --batch-size 4 \\
        --lr 0.02

    # Resume training
    python scripts/train_simple.py \\
        --config configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \\
        --resume
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import mmdet
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Simple training script for visdet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config file",
    )

    # Common overrides
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory for dataset (overrides config)",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Directory to save logs and models (default: work_dirs/<config_name>_simple)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size per GPU (overrides config)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=None,
        dest="lr",
        help="Learning rate (overrides config)",
    )

    # Training control
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training (default: cuda)",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ID to use (default: 0)",
    )

    # Checkpointing
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Auto-resume from the latest checkpoint in work-dir",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained model weights",
    )

    # Validation
    parser.add_argument(
        "--no-val",
        action="store_true",
        help="Disable validation during training",
    )

    return parser.parse_args()


def build_cfg_options(args):
    """Build config options list from command line arguments."""
    cfg_options = []

    # Data settings
    if args.data_root:
        cfg_options.extend(["data_root=" + args.data_root])

    # Training settings
    if args.epochs:
        cfg_options.extend([f"runner.max_epochs={args.epochs}"])

    if args.batch_size:
        cfg_options.extend(
            [
                f"data.samples_per_gpu={args.batch_size}",
                f"data.train_dataloader.batch_size={args.batch_size}",
            ]
        )

    if args.lr:
        cfg_options.extend(
            [
                f"optimizer.lr={args.lr}",
            ]
        )

    # Seed and deterministic
    cfg_options.extend([f"seed={args.seed}"])

    if args.deterministic:
        cfg_options.extend(["deterministic=True"])

    # Validation
    if args.no_val:
        cfg_options.extend(["evaluation=None"])

    return cfg_options


def check_environment():
    """Check if the environment is properly set up."""
    import torch

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA is not available. Training will run on CPU.")
        print("This will be significantly slower. Consider using a GPU if available.\n")

    # Check for multiple GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"\n💡 INFO: {num_gpus} GPUs detected, but this script uses only 1 GPU for simplicity.")
        print("For multi-GPU training, use tools/train.py with --launcher pytorch\n")


def get_work_dir(args):
    """Determine the working directory."""
    if args.work_dir:
        return args.work_dir

    # Auto-generate work_dir from config name
    config_name = Path(args.config).stem
    return f"work_dirs/{config_name}_simple"


def print_header():
    """Print a nice header."""
    print("=" * 70)
    print("  visdet Simple Training Script")
    print("  Single-GPU training with sensible defaults")
    print("=" * 70)
    print()


def print_summary(args, work_dir, cfg_options):
    """Print training configuration summary."""
    print("📋 Training Configuration:")
    print(f"  Config file:    {args.config}")
    print(f"  Work directory: {work_dir}")
    print(f"  Device:         {args.device} (GPU {args.gpu_id if args.device == 'cuda' else 'N/A'})")
    print(f"  Random seed:    {args.seed}")
    print(f"  Deterministic:  {args.deterministic}")

    if cfg_options:
        print("\n🔧 Config Overrides:")
        for opt in cfg_options:
            key, value = opt.split("=", 1)
            print(f"  {key}: {value}")

    if args.resume:
        print(f"\n♻️  Resume: Enabled (will auto-resume from {work_dir})")

    if args.pretrained:
        print(f"\n📦 Pretrained: {args.pretrained}")

    print()


def main():
    """Main training function."""
    args = parse_args()

    # Print header
    print_header()

    # Check environment
    check_environment()

    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"❌ ERROR: Config file not found: {args.config}")
        print("\nPlease provide a valid config file path.")
        print("Example configs are located in: configs/")
        sys.exit(1)

    # Determine work directory
    work_dir = get_work_dir(args)

    # Build config options
    cfg_options = build_cfg_options(args)

    # Print summary
    print_summary(args, work_dir, cfg_options)

    # Build command to call tools/train.py
    tools_train = str(Path(__file__).parent.parent / "tools" / "train.py")

    cmd = [
        sys.executable,
        tools_train,
        args.config,
        "--work-dir",
        work_dir,
        "--launcher",
        "none",
        "--seed",
        str(args.seed),
        "--gpu-id",
        str(args.gpu_id),
    ]

    if args.deterministic:
        cmd.append("--deterministic")

    if args.resume:
        cmd.append("--auto-resume")

    if args.no_val:
        cmd.append("--no-validate")

    if cfg_options:
        cmd.extend(["--cfg-options"] + cfg_options)

    # Print command being executed
    print("🚀 Starting training...")
    print(f"\nExecuting: {' '.join(cmd)}\n")
    print("=" * 70)
    print()

    # Import and call the training function directly for better integration
    import subprocess

    result = subprocess.run(cmd)

    # Print final message
    print()
    print("=" * 70)
    if result.returncode == 0:
        print("✅ Training completed successfully!")
        print(f"\n📁 Results saved to: {work_dir}")
        print(f"   - Latest checkpoint: {work_dir}/latest.pth")
        print(f"   - Best checkpoint:   {work_dir}/best_*.pth")
        print(f"   - Training log:      {work_dir}/*.log")
    else:
        print("❌ Training failed!")
        print(f"\nCheck the logs in {work_dir} for more details.")
        sys.exit(result.returncode)

    print("=" * 70)


if __name__ == "__main__":
    main()
