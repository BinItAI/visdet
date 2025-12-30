# ruff: noqa
# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.

import os.path as osp
import pickle
import shutil
import tempfile
from collections import OrderedDict
from collections.abc import Generator
from itertools import chain, zip_longest
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch import distributed as torch_dist
from torch._utils import _flatten_dense_tensors, _take_tensors, _unflatten_dense_tensors
from torch.distributed import ProcessGroup

from visdet.engine.utils import mkdir_or_exist

from visdet.engine.dist.utils import (
    barrier,
    cast_data_device,
    get_backend,
    get_comm_device,
    get_data_device,
    get_default_group,
    get_dist_info,
    get_rank,
    get_world_size,
)


def _get_reduce_op(name: str) -> torch_dist.ReduceOp:
    op_mappings = {
        "sum": torch_dist.ReduceOp.SUM,
        "product": torch_dist.ReduceOp.PRODUCT,
        "min": torch_dist.ReduceOp.MIN,
        "max": torch_dist.ReduceOp.MAX,
        "band": torch_dist.ReduceOp.BAND,
        "bor": torch_dist.ReduceOp.BOR,
        "bxor": torch_dist.ReduceOp.BXOR,
    }
    if name.lower() not in op_mappings:
        raise ValueError(f"reduce op should be one of {op_mappings.keys()}, but got {name}")
    return op_mappings[name.lower()]


def all_reduce(data: Tensor, op: str = "sum", group: ProcessGroup | None = None) -> None:
    world_size = get_world_size(group)
    if world_size > 1:
        if group is None:
            group = get_default_group()

        input_device = get_data_device(data)
        backend_device = get_comm_device(group)
        data_on_device = cast_data_device(data, backend_device)

        if op.lower() == "mean":
            torch_dist.all_reduce(data_on_device, _get_reduce_op("sum"), group)
            data_on_device = torch.true_divide(data_on_device, world_size)
        else:
            torch_dist.all_reduce(data_on_device, _get_reduce_op(op), group)

        cast_data_device(data_on_device, input_device, out=data)


def all_gather(data: Tensor, group: ProcessGroup | None = None) -> list[Tensor]:
    world_size = get_world_size(group)
    if world_size == 1:
        return [data]

    if group is None:
        group = get_default_group()

    input_device = get_data_device(data)
    backend_device = get_comm_device(group)
    data_on_device = cast_data_device(data, backend_device)

    gather_list = [torch.empty_like(data, device=backend_device) for _ in range(world_size)]
    torch_dist.all_gather(gather_list, data_on_device, group)

    return cast_data_device(gather_list, input_device)  # type: ignore


def gather(data: Tensor, dst: int = 0, group: ProcessGroup | None = None) -> list[Tensor | None]:
    world_size = get_world_size(group)
    if world_size == 1:
        return [data]

    if group is None:
        group = get_default_group()

    input_device = get_data_device(data)
    backend_device = get_comm_device(group)

    if get_rank(group) == dst:
        gather_list = [torch.empty_like(data, device=backend_device) for _ in range(world_size)]
    else:
        gather_list = []

    torch_dist.gather(data, gather_list, dst, group)

    if get_rank(group) == dst:
        return cast_data_device(gather_list, input_device)  # type: ignore
    return gather_list


def broadcast(data: Tensor, src: int = 0, group: ProcessGroup | None = None) -> None:
    if get_world_size(group) > 1:
        if group is None:
            group = get_default_group()

        input_device = get_data_device(data)
        backend_device = get_comm_device(group)
        data_on_device = cast_data_device(data, backend_device)
        data_on_device = data_on_device.contiguous()  # type: ignore
        torch_dist.broadcast(data_on_device, src, group)

        if get_rank(group) != src:
            cast_data_device(data_on_device, input_device, data)


def sync_random_seed(group: ProcessGroup | None = None) -> int:
    seed = np.random.randint(2**31)
    if get_world_size(group) == 1:
        return seed

    if group is None:
        group = get_default_group()

    backend_device = get_comm_device(group)

    if get_rank(group) == 0:
        random_num = torch.tensor(seed, dtype=torch.int32).to(backend_device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32).to(backend_device)

    torch_dist.broadcast(random_num, src=0, group=group)
    return int(random_num.item())


def broadcast_object_list(data: list[Any], src: int = 0, group: ProcessGroup | None = None) -> None:
    assert isinstance(data, list)

    if get_world_size(group) > 1:
        if group is None:
            group = get_default_group()

        torch_dist.broadcast_object_list(data, src, group)


def all_reduce_dict(data: dict[str, Tensor], op: str = "sum", group: ProcessGroup | None = None) -> None:
    assert isinstance(data, dict)

    world_size = get_world_size(group)
    if world_size > 1:
        if group is None:
            group = get_default_group()

        keys = sorted(data.keys())
        tensor_shapes = [data[k].shape for k in keys]
        tensor_sizes = [data[k].numel() for k in keys]

        flatten_tensor = torch.cat([data[k].flatten() for k in keys])
        all_reduce(flatten_tensor, op=op, group=group)

        split_tensors = [
            x.reshape(shape) for x, shape in zip(torch.split(flatten_tensor, tensor_sizes), tensor_shapes, strict=False)
        ]

        for k, v in zip(keys, split_tensors, strict=False):
            data[k] = v


def all_gather_object(data: Any, group: ProcessGroup | None = None) -> list[Any]:
    world_size = get_world_size(group)
    if world_size == 1:
        return [data]

    if group is None:
        group = get_default_group()

    object_list: list[Any] = [None] * world_size
    torch_dist.all_gather_object(object_list, data, group)
    return object_list


def gather_object(data: Any, dst: int = 0, group: ProcessGroup | None = None) -> list[Any] | None:
    world_size = get_world_size(group)
    if world_size == 1:
        return [data]

    if group is None:
        group = get_default_group()

    gather_list = [None] * world_size if get_rank(group) == dst else None
    torch_dist.gather_object(data, gather_list, dst, group)
    return gather_list


def collect_results(results: list, size: int, device: str = "cpu", tmpdir: str | None = None) -> list | None:
    if device not in ["gpu", "cpu"]:
        raise NotImplementedError(f"device must be 'cpu' or 'gpu', but got {device}")

    if device == "gpu":
        assert tmpdir is None, "tmpdir should be None when device is 'gpu'"
        return _collect_results_device(results, size)

    return collect_results_cpu(results, size, tmpdir)


def collect_results_cpu(result_part: list, size: int, tmpdir: str | None = None) -> list | None:
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part[:size]

    if tmpdir is None:
        max_len = 512
        dir_tensor = torch.full((max_len,), 32, dtype=torch.uint8)
        if rank == 0:
            mkdir_or_exist(".dist_test")
            tmpdir_path = tempfile.mkdtemp(dir=".dist_test")
            tmpdir_tensor = torch.tensor(bytearray(tmpdir_path.encode()), dtype=torch.uint8)
            dir_tensor[: len(tmpdir_tensor)] = tmpdir_tensor
        broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.numpy().tobytes().decode().rstrip()
    else:
        mkdir_or_exist(tmpdir)

    with open(osp.join(tmpdir, f"part_{rank}.pkl"), "wb") as f:
        pickle.dump(result_part, f, protocol=2)

    barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        path = osp.join(tmpdir, f"part_{i}.pkl")
        if not osp.exists(path):
            raise FileNotFoundError(
                f"{tmpdir} is not a shared directory for rank {i}. Ensure it is shared for all ranks."
            )
        with open(path, "rb") as f:
            part_list.append(pickle.load(f))

    zipped_results = zip_longest(*part_list)
    ordered_results = [i for i in chain.from_iterable(zipped_results) if i is not None]
    ordered_results = ordered_results[:size]

    shutil.rmtree(tmpdir)
    return ordered_results


def _collect_results_device(result_part: list, size: int) -> list | None:
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part[:size]

    part_list = all_gather_object(result_part)

    if rank == 0:
        zipped_results = zip_longest(*part_list)
        ordered_results = [i for i in chain.from_iterable(zipped_results) if i is not None]
        return ordered_results[:size]

    return None


def collect_results_gpu(result_part: list, size: int) -> list | None:
    return _collect_results_device(result_part, size)


def _all_reduce_coalesced(
    tensors: list[torch.Tensor],
    bucket_size_mb: int = -1,
    op: str = "sum",
    group: ProcessGroup | None = None,
) -> None:
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            buckets.setdefault(tp, []).append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        all_reduce(flat_tensors, op=op, group=group)
        for tensor, synced in zip(bucket, _unflatten_dense_tensors(flat_tensors, bucket), strict=False):
            tensor.copy_(synced)


def all_reduce_params(
    params: list | Generator[torch.Tensor, None, None],
    coalesce: bool = True,
    bucket_size_mb: int = -1,
    op: str = "sum",
    group: ProcessGroup | None = None,
) -> None:
    world_size = get_world_size(group)
    if world_size == 1:
        return

    params_data = [param.data for param in params]
    if coalesce:
        _all_reduce_coalesced(params_data, bucket_size_mb, op=op, group=group)
    else:
        for tensor in params_data:
            all_reduce(tensor, op=op, group=group)
