"""Automatic distributed training with torch.multiprocessing.

This module is ported from BinItAI/core (PR #5414) to provide an
"auto-DDP" mode that keeps the Python entrypoint unchanged.

It detects the number of available GPUs and automatically configures
single-node DistributedDataParallel training when multiple GPUs are present.

Key points:
- Uses `torch.multiprocessing.spawn()` (no `torchrun` required)
- Rank 0 builds the config, then broadcasts it to other ranks
- Each worker sets its CUDA device before initializing distributed state
"""

import logging
import os
import socket
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)


def _find_free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def _get_gpu_count_safe() -> int:
    """Detect GPU count without initializing CUDA context in parent process."""

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        if cuda_visible == "":
            return 0
        devices = [d.strip() for d in cuda_visible.split(",") if d.strip()]
        return len(devices)

    # Prefer nvidia-smi if available; it doesn't initialize CUDA runtime.
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lines = [line for line in result.stdout.strip().split("\n") if line.strip()]
            return len(lines)
    except Exception:
        pass

    # Do not call torch.cuda.device_count() here.
    logger.warning(
        "Could not detect GPU count via nvidia-smi or CUDA_VISIBLE_DEVICES; "
        "defaulting to single-GPU mode. Set CUDA_VISIBLE_DEVICES to enable multi-GPU."
    )
    return 1


def _worker_fn(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: str,
    config_builder: Callable[[int, int], tuple[Any, dict]],
    rank_0_callback: Callable[[Any, dict], None] | None,
) -> None:
    try:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        if torch.cuda.is_available():
            torch.cuda.set_device(rank)

        backend = os.environ.get("DIST_BACKEND", "nccl" if torch.cuda.is_available() else "gloo")
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=3600),
        )

        if backend == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[rank])
        else:
            dist.barrier()

        from visdet.engine.runner import Runner

        if rank == 0:
            cfg, readable = config_builder(rank, world_size)
            cfg.launcher = "pytorch"

            if rank_0_callback is not None:
                rank_0_callback(cfg, readable)
        else:
            cfg = None
            readable = None

        object_list = [cfg, readable]
        dist.broadcast_object_list(object_list, src=0)
        cfg, readable = object_list

        if cfg is None:
            raise RuntimeError(f"[Rank {rank}] Config broadcast failed")

        runner = Runner.from_cfg(cfg)
        runner.readable_config = readable
        runner.train()

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _single_gpu_train(
    config_builder: Callable[[int, int], tuple[Any, dict]],
    rank_0_callback: Callable[[Any, dict], None] | None,
) -> None:
    from visdet.engine.runner import Runner

    cfg, readable = config_builder(0, 1)
    cfg.launcher = "none"

    if rank_0_callback is not None:
        rank_0_callback(cfg, readable)

    runner = Runner.from_cfg(cfg)
    runner.readable_config = readable
    runner.train()


def auto_train(
    config_builder: Callable[[int, int], tuple[Any, dict]],
    rank_0_callback: Callable[[Any, dict], None] | None = None,
) -> None:
    """Automatically run training in single or multi-GPU mode.

    If multiple GPUs are detected, this spawns one worker per GPU using
    `torch.multiprocessing.spawn()` and initializes a single-node process group.

    Args:
        config_builder: Function taking `(rank, world_size)` and returning
            `(cfg, readable_dict)` where `cfg` is compatible with `Runner.from_cfg`.
            Note: this function must be picklable (module-level) to work with spawn.
        rank_0_callback: Optional callback executed only on rank 0 after config
            is built but before training starts.
    """

    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    gpu_count = _get_gpu_count_safe()

    if gpu_count <= 1:
        logger.info("Detected %d GPU(s); running single-process training", gpu_count)
        return _single_gpu_train(config_builder, rank_0_callback)

    logger.info("Detected %d GPUs; enabling automatic DDP", gpu_count)

    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", _find_free_port())

    mp.spawn(
        _worker_fn,
        args=(gpu_count, master_addr, master_port, config_builder, rank_0_callback),
        nprocs=gpu_count,
        join=True,
    )
