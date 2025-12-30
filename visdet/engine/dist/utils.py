# ruff: noqa
# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.

import datetime
import functools
import os
import subprocess
from collections.abc import Callable, Iterable, Mapping

import numpy as np
import torch
import torch.multiprocessing as mp
from torch import Tensor
from torch import distributed as torch_dist
from torch.distributed import ProcessGroup

_LOCAL_PROCESS_GROUP: ProcessGroup | None = None


def is_distributed() -> bool:
    """Return True if distributed environment has been initialized."""

    return torch_dist.is_available() and torch_dist.is_initialized()


def get_local_group() -> ProcessGroup | None:
    """Return local process group."""

    if not is_distributed():
        return None

    if _LOCAL_PROCESS_GROUP is None:
        raise RuntimeError(
            "Local process group is not created, please use `init_local_group` to setup local process group."
        )

    return _LOCAL_PROCESS_GROUP


def get_default_group() -> ProcessGroup | None:
    """Return default process group."""

    return torch_dist.distributed_c10d._get_default_group()


def infer_launcher() -> str:
    if "WORLD_SIZE" in os.environ:
        return "pytorch"
    if "SLURM_NTASKS" in os.environ:
        return "slurm"
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        return "mpi"
    return "none"


def init_dist(launcher: str, backend: str = "nccl", init_backend: str = "torch", **kwargs) -> None:
    """Initialize distributed environment.

    Args:
        launcher: Way to launch multi-process. Supported launchers are
            'pytorch', 'mpi', 'slurm'.
        backend: torch.distributed backend. Typically 'nccl' or 'gloo'.
        init_backend: Initialization backend. Keep as 'torch' for visdet.
        **kwargs: Passed to torch.distributed.init_process_group.
    """

    timeout = kwargs.get("timeout", None)
    if timeout is not None:
        kwargs["timeout"] = datetime.timedelta(seconds=timeout)

    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    if launcher == "pytorch":
        _init_dist_pytorch(backend, init_backend=init_backend, **kwargs)
    elif launcher == "mpi":
        _init_dist_mpi(backend, **kwargs)
    elif launcher == "slurm":
        _init_dist_slurm(backend, init_backend=init_backend, **kwargs)
    else:
        raise ValueError(f"Invalid launcher type: {launcher}")


def _init_dist_pytorch(backend: str, init_backend: str = "torch", **kwargs) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if init_backend != "torch":
        raise ValueError(f'Only init_backend="torch" is supported, got {init_backend!r}')

    torch_dist.init_process_group(backend=backend, rank=rank, world_size=int(os.environ["WORLD_SIZE"]), **kwargs)


def _init_dist_mpi(backend: str, **kwargs) -> None:
    local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    if "MASTER_ADDR" not in os.environ:
        raise KeyError("The environment variable MASTER_ADDR is not set")

    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    os.environ["LOCAL_RANK"] = str(local_rank)

    torch_dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend: str, port: int | None = None, init_backend: str = "torch", **kwargs) -> None:
    proc_id = int(os.environ["SLURM_PROCID"])
    ntasks = int(os.environ["SLURM_NTASKS"])
    node_list = os.environ["SLURM_NODELIST"]

    local_rank_env = os.environ.get("SLURM_LOCALID", None)
    if local_rank_env is not None:
        local_rank = int(local_rank_env)
    else:
        local_rank = proc_id % max(torch.cuda.device_count(), 1)

    addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")

    if port is not None:
        os.environ["MASTER_PORT"] = str(port)
    elif "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = addr

    os.environ["WORLD_SIZE"] = str(ntasks)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(proc_id)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if init_backend != "torch":
        raise ValueError(f'Only init_backend="torch" is supported, got {init_backend!r}')

    torch_dist.init_process_group(backend=backend, **kwargs)


def init_local_group(node_rank: int, num_gpus_per_node: int) -> None:
    global _LOCAL_PROCESS_GROUP
    assert _LOCAL_PROCESS_GROUP is None

    ranks = list(range(node_rank * num_gpus_per_node, (node_rank + 1) * num_gpus_per_node))
    _LOCAL_PROCESS_GROUP = torch_dist.new_group(ranks)


def get_backend(group: ProcessGroup | None = None) -> str | None:
    if is_distributed():
        if group is None:
            group = get_default_group()
        return torch_dist.get_backend(group)
    return None


def get_world_size(group: ProcessGroup | None = None) -> int:
    if is_distributed():
        if group is None:
            group = get_default_group()
        return torch_dist.get_world_size(group)

    return int(os.environ.get("WORLD_SIZE", 1))


def get_rank(group: ProcessGroup | None = None) -> int:
    if is_distributed():
        if group is None:
            group = get_default_group()
        return torch_dist.get_rank(group)

    return int(os.environ.get("RANK", 0))


def get_local_size() -> int:
    if not is_distributed():
        return 1
    if _LOCAL_PROCESS_GROUP is None:
        raise RuntimeError(
            "Local process group is not created, please use `init_local_group` to setup local process group."
        )
    return torch_dist.get_world_size(_LOCAL_PROCESS_GROUP)


def get_local_rank() -> int:
    if not is_distributed():
        return 0
    if _LOCAL_PROCESS_GROUP is None:
        raise RuntimeError(
            "Local process group is not created, please use `init_local_group` to setup local process group."
        )
    return torch_dist.get_rank(_LOCAL_PROCESS_GROUP)


def get_dist_info(group: ProcessGroup | None = None) -> tuple[int, int]:
    world_size = get_world_size(group)
    rank = get_rank(group)
    return rank, world_size


def is_main_process(group: ProcessGroup | None = None) -> bool:
    return get_rank(group) == 0


def master_only(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)

    return wrapper


def barrier(group: ProcessGroup | None = None) -> None:
    if is_distributed():
        if group is None:
            group = get_default_group()
        torch_dist.barrier(group)


def get_data_device(data: Tensor | Mapping | Iterable) -> torch.device:
    if isinstance(data, Tensor):
        return data.device
    if isinstance(data, Mapping):
        pre = None
        for v in data.values():
            cur = get_data_device(v)
            if pre is None:
                pre = cur
            elif cur != pre:
                raise ValueError(f"device type in data should be consistent, but got {cur} and {pre}")
        if pre is None:
            raise ValueError("data should not be empty")
        return pre
    if isinstance(data, Iterable) and not isinstance(data, str) and not isinstance(data, np.ndarray):
        pre = None
        for item in data:
            cur = get_data_device(item)
            if pre is None:
                pre = cur
            elif cur != pre:
                raise ValueError(f"device type in data should be consistent, but got {cur} and {pre}")
        if pre is None:
            raise ValueError("data should not be empty")
        return pre

    raise TypeError(f"data should be a Tensor, sequence of tensor or dict, but got {type(data)}")


def get_comm_device(group: ProcessGroup | None = None) -> torch.device:
    backend = get_backend(group)
    if backend == torch_dist.Backend.NCCL:
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def cast_data_device(
    data: Tensor | Mapping | Iterable,
    device: torch.device,
    out: Tensor | Mapping | Iterable | None = None,
) -> Tensor | Mapping | Iterable:
    if out is not None and type(data) is not type(out):
        raise TypeError(f"out should be same type as data, got {type(data)} vs {type(out)}")

    if isinstance(data, Tensor):
        data_on_device = data if get_data_device(data) == device else data.to(device)
        if out is not None:
            out.copy_(data_on_device)  # type: ignore
        return data_on_device

    if isinstance(data, Mapping):
        data_on_device: dict = {}
        if out is not None:
            if len(data) != len(out):  # type: ignore
                raise ValueError("length of data and out should be same")
            for k, v in data.items():
                data_on_device[k] = cast_data_device(v, device, out[k])  # type: ignore
        else:
            for k, v in data.items():
                data_on_device[k] = cast_data_device(v, device)
        if not data_on_device:
            raise ValueError("data should not be empty")
        return type(data)(data_on_device)  # type: ignore

    if isinstance(data, Iterable) and not isinstance(data, str) and not isinstance(data, np.ndarray):
        data_on_device_list = []
        if out is not None:
            for v1, v2 in zip(data, out, strict=False):
                data_on_device_list.append(cast_data_device(v1, device, v2))
        else:
            for v in data:
                data_on_device_list.append(cast_data_device(v, device))
        if not data_on_device_list:
            raise ValueError("data should not be empty")
        return type(data)(data_on_device_list)  # type: ignore

    raise TypeError(f"data should be a Tensor, list of tensor or dict, but got {type(data)}")
