# Copyright (c) OpenMMLab. All rights reserved.
"""Distributed utilities for visdet."""

import functools
import os
import pickle
import warnings
from typing import Any, List, Optional

import torch
import torch.distributed as dist_lib


def _is_dist_available_and_initialized():
    """Check if distributed training is available and initialized."""
    return dist_lib.is_available() and dist_lib.is_initialized()


def get_dist_info():
    """Get distributed training info.

    Returns:
        tuple: rank, world_size
    """
    if _is_dist_available_and_initialized():
        rank = dist_lib.get_rank()
        world_size = dist_lib.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def get_rank():
    """Get rank of current process."""
    if _is_dist_available_and_initialized():
        return dist_lib.get_rank()
    return 0


def get_world_size():
    """Get world size."""
    if _is_dist_available_and_initialized():
        return dist_lib.get_world_size()
    return 1


def is_distributed():
    """Check if distributed training is initialized."""
    return _is_dist_available_and_initialized()


def is_main_process():
    """Check if current process is main process (rank 0)."""
    return get_rank() == 0


def master_only(func):
    """Decorator to make a function only execute on master process."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
    return wrapper


def barrier():
    """Synchronize all processes."""
    if _is_dist_available_and_initialized():
        dist_lib.barrier()


def broadcast(data, src=0, group=None):
    """Broadcast data from src rank to all ranks."""
    if not _is_dist_available_and_initialized():
        return data

    if isinstance(data, torch.Tensor):
        dist_lib.broadcast(data, src, group=group)
        return data
    else:
        # For non-tensor data, convert to tensor, broadcast, then convert back
        if get_rank() == src:
            data_tensor = torch.tensor(data, device='cuda' if torch.cuda.is_available() else 'cpu')
        else:
            data_tensor = torch.zeros_like(torch.tensor(data, device='cuda' if torch.cuda.is_available() else 'cpu'))
        dist_lib.broadcast(data_tensor, src, group=group)
        return data_tensor.item() if data_tensor.dim() == 0 else data_tensor


def broadcast_object_list(obj_list, src=0, group=None):
    """Broadcast a list of objects from src rank to all ranks."""
    if not _is_dist_available_and_initialized():
        return obj_list

    dist_lib.broadcast_object_list(obj_list, src, group=group)
    return obj_list


def all_reduce_params(model):
    """All reduce model parameters for synchronization."""
    if not _is_dist_available_and_initialized():
        return

    world_size = get_world_size()
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            dist_lib.all_reduce(param.grad.data)
            param.grad.data.div_(world_size)


def init_dist(launcher='pytorch', backend='nccl', **kwargs):
    """Initialize distributed environment."""
    if _is_dist_available_and_initialized():
        return get_dist_info()

    if launcher == 'pytorch':
        dist_lib.init_process_group(backend=backend, **kwargs)
    else:
        raise NotImplementedError(f'Launcher {launcher} is not supported')

    return get_dist_info()


def collect_results(result_part, size, tmpdir=None):
    """Collect results from all processes and merge them."""
    rank, world_size = get_dist_info()

    if tmpdir is None:
        tmpdir = '.'

    # Create result file
    result_file = os.path.join(tmpdir, f'result_rank_{rank}.pkl')
    with open(result_file, 'wb') as f:
        pickle.dump(result_part, f)

    dist_lib.barrier()

    if rank == 0:
        results = []
        for i in range(world_size):
            result_file = os.path.join(tmpdir, f'result_rank_{i}.pkl')
            with open(result_file, 'rb') as f:
                results.append(pickle.load(f))

        # Clean up
        for i in range(world_size):
            result_file = os.path.join(tmpdir, f'result_rank_{i}.pkl')
            if os.path.exists(result_file):
                os.remove(result_file)

        # Merge results (flatten list of lists)
        merged_results = []
        for result in results:
            if isinstance(result, list):
                merged_results.extend(result)
            else:
                merged_results.append(result)
        return merged_results

    return None


def sync_random_seed(seed=None, device='cuda'):
    """Make sure different ranks share the same seed.

    All workers must call this function, otherwise it will deadlock.
    """
    import numpy as np

    if seed is None:
        seed = np.random.randint(2**31)

    rank, world_size = get_dist_info()

    if world_size == 1:
        return seed

    if not _is_dist_available_and_initialized():
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)

    dist_lib.broadcast(random_num, src=0)
    return random_num.item()


def infer_launcher():
    """Infer launcher type from environment variables."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        return 'pytorch'
    else:
        return None


# Backward compatibility
utils = type('utils', (), {'master_only': master_only})()

__all__ = [
    'get_dist_info',
    'get_rank',
    'get_world_size',
    'is_distributed',
    'is_main_process',
    'master_only',
    'barrier',
    'broadcast',
    'broadcast_object_list',
    'all_reduce_params',
    'init_dist',
    'collect_results',
    'sync_random_seed',
    'infer_launcher',
]
