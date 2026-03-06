"""Test/evaluate a detector with visdet-native runner config.

This entrypoint intentionally avoids mmcv/mmdet/mmengine package dependencies.
"""

import argparse
import json
import os
import warnings
from typing import Any

import yaml

from visdet.engine import DefaultScope
from visdet.engine.config import Config
from visdet.engine.runner import Runner


def _parse_cfg_options(values: list[str] | None) -> dict[str, Any]:
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


def _apply_eval_metrics(cfg: Config, metrics: list[str]) -> None:
    metric_value: str | list[str] = metrics if len(metrics) > 1 else metrics[0]
    evaluator = cfg.get("test_evaluator", None)
    if evaluator is None:
        evaluator = cfg.get("val_evaluator", None)
        if evaluator is not None:
            cfg.test_evaluator = evaluator

    if evaluator is None:
        raise ValueError("No test_evaluator/val_evaluator found in config; cannot apply --eval metrics.")

    if isinstance(evaluator, dict):
        evaluator["metric"] = metric_value
    elif isinstance(evaluator, list):
        for item in evaluator:
            if isinstance(item, dict):
                item["metric"] = metric_value
    else:
        raise TypeError(f"Unsupported evaluator type: {type(evaluator)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test (and eval) a model (visdet-native)")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file path")
    parser.add_argument("--work-dir", help="directory to save evaluation metrics")
    parser.add_argument("--out", help="output JSON file for metrics")
    parser.add_argument("--eval", type=str, nargs="+", help='evaluation metrics, e.g., "bbox segm"')
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="format-only mode from legacy mmdet toolchain (not supported in visdet-native runner)",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        help="override config in key=value format (e.g. test_dataloader.batch_size=1)",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="deprecated alias for --eval-options",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        help="custom evaluation options in key=value format (merged into cfg)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)

    # Deprecated/ignored options retained for CLI compatibility.
    parser.add_argument("--fuse-conv-bn", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu-ids", type=int, nargs="+", help=argparse.SUPPRESS)
    parser.add_argument("--gpu-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--show", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--show-dir", help=argparse.SUPPRESS)
    parser.add_argument("--show-score-thr", type=float, default=0.3, help=argparse.SUPPRESS)
    parser.add_argument("--gpu-collect", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--tmpdir", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError("--options and --eval-options cannot be both specified")
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options", stacklevel=2)
        args.eval_options = args.options

    return args


def main() -> None:
    args = parse_args()

    if args.format_only:
        raise ValueError("--format-only is not supported by visdet-native tools/test.py")

    if args.fuse_conv_bn:
        warnings.warn("--fuse-conv-bn is ignored by visdet-native tools/test.py", stacklevel=2)

    cfg = Config.fromfile(args.config)

    if args.cfg_options:
        cfg.merge_from_dict(_parse_cfg_options(args.cfg_options))

    if args.eval_options:
        cfg.merge_from_dict(_parse_cfg_options(args.eval_options))

    if args.work_dir:
        cfg.work_dir = args.work_dir

    cfg.launcher = args.launcher
    cfg.load_from = args.checkpoint
    cfg.resume = False

    if args.eval:
        _apply_eval_metrics(cfg, args.eval)

    DefaultScope.get_instance("visdet", scope_name="visdet")
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()

    if metrics:
        print(metrics)

    output_path = args.out
    if output_path is None and args.work_dir:
        output_path = os.path.join(args.work_dir, "eval_metrics.json")

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"config": args.config, "checkpoint": args.checkpoint, "metrics": metrics}, f, indent=2)
        print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
