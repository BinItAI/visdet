# ruff: noqa
# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from visengine.device import is_cuda_available
from visengine.dist import get_rank, sync_random_seed
from visengine.logging import print_log
from visengine.utils import is_list_of


def calc_dynamic_intervals(
    start_interval: int, dynamic_interval_list: list[tuple[int, int]] | None = None
) -> tuple[list[int], list[int]]:
    """Calculate dynamic intervals.

    Args:
        start_interval (int): The interval used in the beginning.
        dynamic_interval_list (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.

    Returns:
        Tuple[List[int], List[int]]: a list of milestone and its corresponding
        intervals.
    """
    if dynamic_interval_list is None:
        return [0], [start_interval]

    assert is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend([dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend([dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals


def set_random_seed(seed: int | None = None, deterministic: bool = False, diff_rank_seed: bool = False) -> int:
    """Set random seed.

    Args:
        seed (int, optional): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Defaults to False.
        diff_rank_seed (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Defaults to False.
    """
    if seed is None:
        seed = sync_random_seed()

    if diff_rank_seed:
        rank = get_rank()
        seed += rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    if is_cuda_available():
        torch.cuda.manual_seed_all(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        if torch.backends.cudnn.benchmark:
            print_log(
                "torch.backends.cudnn.benchmark is going to be set as `False` to cause cuDNN to deterministically select an algorithm",
                logger="current",
                level=logging.WARNING,
            )
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    return seed


def _get_batch_size(dataloader: dict):
    if isinstance(dataloader, dict):
        if "batch_size" in dataloader:
            return dataloader["batch_size"]
        elif "batch_sampler" in dataloader and "batch_size" in dataloader["batch_sampler"]:
            return dataloader["batch_sampler"]["batch_size"]
        else:
            raise ValueError("Please set batch_size in `Dataloader` or `batch_sampler`")
    elif isinstance(dataloader, DataLoader):
        return dataloader.batch_sampler.batch_size
    else:
        raise ValueError(f"dataloader should be a dict or a Dataloader instance, but got {type(dataloader)}")
