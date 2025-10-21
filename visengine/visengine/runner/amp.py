# ruff: noqa
# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.
import logging
from contextlib import contextmanager

import torch

from visengine.device import get_device, is_cuda_available
from visengine.logging import print_log
from visengine.utils import digit_version


@contextmanager
def autocast(
    device_type: str | None = None,
    dtype: torch.dtype | None = None,
    enabled: bool = True,
    cache_enabled: bool | None = None,
):
    """A wrapper of ``torch.autocast``.

    Provides a unified interface for PyTorch autocast functionality.
    Only supports PyTorch 2.0.0 and above.

    Args:
        device_type (str, required):  Whether to use 'cuda' or 'cpu' device.
        enabled(bool):  Whether autocasting should be enabled in the region.
            Defaults to True
        dtype (torch_dtype, optional):  Whether to use ``torch.float16`` or
            ``torch.bfloat16``.
        cache_enabled(bool, optional):  Whether the weight cache inside
            autocast should be enabled.
    """
    # Modified from https://github.com/pytorch/pytorch/blob/master/torch/amp/autocast_mode.py
    # This code should update with the `torch.autocast`.
    if cache_enabled is None:
        cache_enabled = torch.is_autocast_cache_enabled()
    device = get_device()
    device_type = device if device_type is None else device_type

    if device_type == "cuda":
        if dtype is None:
            dtype = torch.get_autocast_gpu_dtype()

        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Current CUDA Device does not support bfloat16. Please switch dtype to float16.")

    elif device_type == "cpu":
        if dtype is None:
            dtype = torch.bfloat16
        assert dtype == torch.bfloat16, "In CPU autocast, only support `torch.bfloat16` dtype"
    else:
        # Device like MPS does not support fp16 training or testing.
        # If an inappropriate device is set and fp16 is enabled, an error
        # will be thrown.
        if enabled is False:
            yield
            return
        else:
            raise ValueError(f"User specified autocast device_type must be cuda or cpu, but got {device_type}")

    with torch.autocast(
        device_type=device_type,
        enabled=enabled,
        dtype=dtype,
        cache_enabled=cache_enabled,
    ):
        yield
