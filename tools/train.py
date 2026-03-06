"""Train a detector with visdet-native runner config.

This entrypoint intentionally avoids mmcv/mmdet/mmengine package dependencies.
"""

import argparse
import os
import warnings
from typing import Any

import yaml

from visdet.engine import DefaultScope
from visdet.engine.config import Config
from visdet.engine.runner import Runner


def _parse_cfg_options(values: list[str] | None) -> dict[str, Any]:
    """Parse key=value CLI overrides into a dict for Config.merge_from_dict()."""
    parsed: dict[str, Any] = {}
    if not values:
        return parsed

    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid cfg option '{item}'. Expected key=value format.")
        key, value = item.split("=", 1)
        try:
            parsed[key] = yaml.safe_load(value)
        except Exception:
            parsed[key] = value
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a detector (visdet-native)")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument("--resume-from", help="checkpoint file to resume from")
    parser.add_argument("--load-from", help="checkpoint file to load model weights from")
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="resume from latest checkpoint in work_dir",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="disable validation loop during training",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="enable AMP optimizer wrapper for fp16 training",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--diff-seed",
        action="store_true",
        help="offset seed by LOCAL_RANK for distributed launchers",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="set deterministic options for reproducibility",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="deprecated alias for --cfg-options",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        help='override config in key=value format (e.g. optimizer.lr=0.001 data.train.batch_size=2)',
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--auto-scale-lr", action="store_true", help="enable auto_scale_lr in config if present")
    # Deprecated/ignored GPU flags retained for CLI compatibility.
    parser.add_argument("--gpus", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--gpu-ids", type=int, nargs="+", help=argparse.SUPPRESS)
    parser.add_argument("--gpu-id", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.options and args.cfg_options:
        raise ValueError("--options and --cfg-options cannot be both specified")
    if args.options:
        warnings.warn("--options is deprecated in favor of --cfg-options", stacklevel=2)
        args.cfg_options = args.options

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main() -> None:
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options:
        cfg.merge_from_dict(_parse_cfg_options(args.cfg_options))

    if args.work_dir:
        cfg.work_dir = args.work_dir

    if args.resume_from:
        cfg.load_from = args.resume_from
        cfg.resume = True
    elif args.load_from:
        cfg.load_from = args.load_from
        cfg.resume = False
    elif args.auto_resume:
        cfg.resume = True

    if args.no_validate:
        cfg.val_dataloader = None
        cfg.val_cfg = None
        cfg.val_evaluator = None

    if args.auto_scale_lr and isinstance(cfg.get("auto_scale_lr"), dict):
        cfg.auto_scale_lr["enable"] = True

    if args.seed is not None:
        seed = args.seed
        if args.diff_seed:
            seed += int(os.environ.get("LOCAL_RANK", "0"))
        cfg.randomness = {"seed": seed, "deterministic": args.deterministic}
    elif args.deterministic:
        cfg.randomness = {"seed": None, "deterministic": True}

    if args.fp16:
        optim_wrapper = cfg.get("optim_wrapper", {})
        if not isinstance(optim_wrapper, dict):
            optim_wrapper = {}
        optim_wrapper["type"] = "AmpOptimWrapper"
        cfg.optim_wrapper = optim_wrapper

    cfg.launcher = args.launcher

    # Ensure visdet registry scope is active before building the runner.
    DefaultScope.get_instance("visdet", scope_name="visdet")

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
