"""Automatic DDP setup that detects GPUs and spawns worker processes.

This module provides automatic multi-GPU training without requiring torchrun.
It detects the number of available GPUs and automatically spawns worker processes
for distributed training when multiple GPUs are detected.

The system is designed to work seamlessly with SimpleRunner, allowing each
worker to independently create and configure its training runner.
"""

import logging
import os
import socket
from typing import Any, Dict, Optional, Union

import torch
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)


def setup_distributed_env(rank: int, world_size: int, port: int = 29500) -> None:
    """Set up environment variables for distributed training.

    Configures the environment variables needed for PyTorch distributed training,
    including rank, world size, and communication port.

    Args:
        rank: Process rank (0 to world_size-1).
        world_size: Total number of processes.
        port: Master port for communication. Defaults to 29500.
    """
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)  # Single node, so LOCAL_RANK == RANK
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    logger.info(f"[Rank {rank}] Distributed environment configured: rank={rank}, world_size={world_size}, port={port}")


def _find_free_port(start_port: int = 29500) -> int:
    """Find a free port for distributed training communication.

    Attempts to find an available port by binding to it temporarily.
    This avoids port conflicts when multiple training runs are launched.

    Args:
        start_port: Starting port to check. Defaults to 29500.

    Returns:
        int: A free port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
        return port


def _worker_fn(
    rank: int,
    world_size: int,
    port: int,
    train_fn: Any,
    train_args: tuple,
    train_kwargs: Dict[str, Any],
) -> None:
    """Worker process function for distributed training.

    This function is executed by each spawned process (one per GPU).
    Each worker sets up the distributed environment and calls the training function.

    Args:
        rank: Process rank (0 to world_size-1).
        world_size: Total number of processes.
        port: Master port for distributed communication.
        train_fn: Training function to execute (e.g., SimpleRunner.train).
        train_args: Positional arguments for train_fn.
        train_kwargs: Keyword arguments for train_fn.

    Raises:
        Exception: Re-raises any exception from the training function.
    """
    # Set up environment variables for this rank
    setup_distributed_env(rank, world_size, port)

    logger.info(f"[Rank {rank}] Starting worker process")

    try:
        # Initialize the distributed process group for NCCL communication
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
        logger.info(f"[Rank {rank}] Distributed process group initialized (NCCL backend)")

        # Call the training function
        logger.info(f"[Rank {rank}] Calling training function")
        train_fn(*train_args, **train_kwargs)
        logger.info(f"[Rank {rank}] Training complete")

    except Exception as e:
        logger.error(f"[Rank {rank}] Training failed with exception: {e}", exc_info=True)
        raise
    finally:
        # Clean up distributed process group
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            logger.info(f"[Rank {rank}] Distributed process group destroyed")


def auto_ddp_train(
    train_fn: Any,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Automatically configure and run DDP training based on GPU count.

    This is the main entry point for automatic DDP training. It detects the number
    of available GPUs and:
    - For 1 GPU: Runs training directly without distributed setup
    - For >1 GPU: Spawns worker processes for distributed training

    The training function is expected to handle the training logic and should
    be compatible with PyTorch's DDP (e.g., MMEngine Runner with launcher="pytorch").

    Args:
        train_fn: Training function to execute (e.g., lambda: SimpleRunner(...).train()).
        *args: Positional arguments to pass to train_fn.
        **kwargs: Keyword arguments to pass to train_fn.

    Example:
        >>> from visdet import SimpleRunner
        >>> from visdet.ddp import auto_ddp_train
        >>>
        >>> # Define training function that will be executed per-GPU
        >>> def create_and_train():
        ...     runner = SimpleRunner(
        ...         model='mask_rcnn_swin_s',
        ...         dataset='cmr_instance_segmentation',
        ...         epochs=12
        ...     )
        ...     runner.train()
        >>>
        >>> # Launch with automatic DDP detection
        >>> auto_ddp_train(create_and_train)

    Note:
        For multi-GPU training to work, the training function must be compatible
        with PyTorch's DDP. MMEngine Runner automatically handles this when
        launcher="pytorch" is set.
    """
    # Detect available GPUs
    gpu_count = torch.cuda.device_count()
    logger.info(f"Detected {gpu_count} GPU(s)")

    if gpu_count <= 1:
        # Single GPU: Run training directly without distributed setup
        logger.info("Running single-GPU training (no DDP)")
        train_fn(*args, **kwargs)

    else:
        # Multi-GPU: Spawn worker processes for distributed training
        logger.info(f"Detected {gpu_count} GPUs, enabling automatic DDP")

        # Find a free port for distributed communication
        port = _find_free_port()
        logger.info(f"Using port {port} for distributed communication")

        # Set multiprocessing start method to spawn to ensure clean worker processes
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")

        # Spawn worker processes
        try:
            mp.spawn(  # type: ignore
                _worker_fn,
                args=(
                    gpu_count,
                    port,
                    train_fn,
                    args,
                    kwargs,
                ),
                nprocs=gpu_count,
                join=True,
                daemon=False,
            )
            logger.info("All training processes completed successfully")
        except Exception as e:
            logger.error(f"Distributed training failed: {e}", exc_info=True)
            raise
