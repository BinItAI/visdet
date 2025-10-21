# Copyright (c) OpenMMLab. All rights reserved.
import torch


def torch_meshgrid(*tensors):
    """A wrapper of torch.meshgrid to compat different PyTorch versions.

    Since PyTorch 1.10.0a0, torch.meshgrid supports the arguments ``indexing``.
    So we implement a wrapper here to avoid warning when using high-version
    PyTorch and avoid compatibility issues when using previous versions of
    PyTorch.

    Args:
        tensors (List[Tensor]): List of scalars or 1 dimensional tensors.

    Returns:
        Sequence[Tensor]: Sequence of meshgrid tensors.
    """
    return torch.meshgrid(*tensors, indexing="ij")
