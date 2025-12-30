#!/usr/bin/env python3
"""Train with automatic single-node DDP (no torchrun).

This is a thin wrapper around `visdet.engine.runner.auto_train.auto_train`.

Usage:
    python scripts/train_auto_ddp.py path/to/config.py

Notes:
- The config must be compatible with `visdet.engine.runner.Runner.from_cfg()`.
- If multiple GPUs are available, one worker process is spawned per GPU.
"""

import argparse

from visdet.engine.config import Config
from visdet.engine.runner import auto_train

_CONFIG_PATH: str | None = None


def _config_builder(_rank: int, _world_size: int) -> tuple[Config, dict]:
    assert _CONFIG_PATH is not None
    cfg = Config.fromfile(_CONFIG_PATH)
    return cfg, {"config": _CONFIG_PATH}


def main() -> None:
    parser = argparse.ArgumentParser(description="visdet auto-DDP training")
    parser.add_argument("config", help="Path to a Runner.from_cfg config")
    args = parser.parse_args()

    global _CONFIG_PATH
    _CONFIG_PATH = args.config

    auto_train(_config_builder)


if __name__ == "__main__":
    main()
