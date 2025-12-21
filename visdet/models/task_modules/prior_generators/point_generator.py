# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor
from torch.nn.modules.utils import _pair

from visdet.registry import TASK_UTILS


@TASK_UTILS.register_module()
class MlvlPointGenerator:
    """Multi-level Point Generator for anchor-free detectors.

    Generates center points for each feature map level. Used by anchor-free
    detectors like FCOS, ATSS, etc.

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of each
            feature map level.
        offset (float): Offset of the center point. Defaults to 0.5.
    """

    def __init__(
        self,
        strides: list[int] | list[tuple[int, int]],
        offset: float = 0.5,
    ) -> None:
        self.strides = [_pair(stride) for stride in strides]
        self.offset = offset

    @property
    def num_levels(self) -> int:
        """int: Number of feature map levels."""
        return len(self.strides)

    @property
    def num_base_priors(self) -> list[int]:
        """list[int]: Number of base priors at each level (always 1 for points)."""
        return [1 for _ in range(len(self.strides))]

    def _meshgrid(
        self,
        x: Tensor,
        y: Tensor,
        row_major: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Generate meshgrid coordinates.

        Args:
            x (Tensor): x coordinates.
            y (Tensor): y coordinates.
            row_major (bool): If True, return (yy, xx), else (xx, yy).

        Returns:
            tuple[Tensor, Tensor]: Meshgrid coordinates.
        """
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_priors(
        self,
        featmap_sizes: list[tuple[int, int]],
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
        with_stride: bool = False,
    ) -> list[Tensor]:
        """Generate grid points for all feature map levels.

        Args:
            featmap_sizes (list[tuple[int, int]]): List of (H, W) for each level.
            dtype (torch.dtype): Data type of the output. Defaults to float32.
            device (str | torch.device): Device for the output.
            with_stride (bool): Whether to concatenate stride to points.
                If True, output is (N, 3) with (x, y, stride).
                If False, output is (N, 2) with (x, y).

        Returns:
            list[Tensor]: List of points for each level.
        """
        assert len(featmap_sizes) == self.num_levels
        multi_level_priors = []

        for i, (height, width) in enumerate(featmap_sizes):
            stride_h, stride_w = self.strides[i]

            shift_x = (torch.arange(0, width, device=device) + self.offset) * stride_w
            shift_y = (torch.arange(0, height, device=device) + self.offset) * stride_h

            shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
            shift_xx = shift_xx.to(dtype)
            shift_yy = shift_yy.to(dtype)

            if with_stride:
                stride_w_tensor = shift_xx.new_full((shift_xx.shape[0],), stride_w).to(dtype)
                stride_h_tensor = shift_yy.new_full((shift_yy.shape[0],), stride_h).to(dtype)
                # (x, y, stride_w, stride_h) format if stride_w != stride_h
                # For simplicity, just use one stride value if they're equal
                if stride_w == stride_h:
                    shifts = torch.stack([shift_xx, shift_yy, stride_w_tensor], dim=-1)
                else:
                    shifts = torch.stack([shift_xx, shift_yy, stride_w_tensor, stride_h_tensor], dim=-1)
            else:
                shifts = torch.stack([shift_xx, shift_yy], dim=-1)

            multi_level_priors.append(shifts)

        return multi_level_priors

    def valid_flags(
        self,
        featmap_sizes: list[tuple[int, int]],
        pad_shape: tuple[int, int],
        device: str | torch.device = "cuda",
    ) -> list[Tensor]:
        """Generate valid flags for all feature map levels.

        Args:
            featmap_sizes (list[tuple[int, int]]): List of (H, W) for each level.
            pad_shape (tuple[int, int]): Padded image shape (H, W).
            device (str | torch.device): Device for the output.

        Returns:
            list[Tensor]: List of valid flags for each level.
        """
        assert len(featmap_sizes) == self.num_levels
        multi_level_flags = []

        for i, (height, width) in enumerate(featmap_sizes):
            stride_h, stride_w = self.strides[i]
            h, w = pad_shape[:2]

            valid_h = torch.zeros(height, dtype=torch.bool, device=device)
            valid_w = torch.zeros(width, dtype=torch.bool, device=device)

            valid_h[: int(h / stride_h) + 1] = True
            valid_w[: int(w / stride_w) + 1] = True

            valid_xx, valid_yy = self._meshgrid(valid_w, valid_h)
            valid = valid_xx & valid_yy

            multi_level_flags.append(valid)

        return multi_level_flags

    def single_level_grid_priors(
        self,
        featmap_size: tuple[int, int],
        level_idx: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
        with_stride: bool = False,
    ) -> Tensor:
        """Generate grid points for a single feature map level.

        Args:
            featmap_size (tuple[int, int]): (H, W) of the feature map.
            level_idx (int): Index of the feature map level.
            dtype (torch.dtype): Data type of the output.
            device (str | torch.device): Device for the output.
            with_stride (bool): Whether to concatenate stride to points.

        Returns:
            Tensor: Points for this level with shape (N, 2) or (N, 3).
        """
        height, width = featmap_size
        stride_h, stride_w = self.strides[level_idx]

        shift_x = (torch.arange(0, width, device=device) + self.offset) * stride_w
        shift_y = (torch.arange(0, height, device=device) + self.offset) * stride_h

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shift_xx = shift_xx.to(dtype)
        shift_yy = shift_yy.to(dtype)

        if with_stride:
            stride_w_tensor = shift_xx.new_full((shift_xx.shape[0],), stride_w).to(dtype)
            shifts = torch.stack([shift_xx, shift_yy, stride_w_tensor], dim=-1)
        else:
            shifts = torch.stack([shift_xx, shift_yy], dim=-1)

        return shifts

    def sparse_priors(
        self,
        prior_idxs: Tensor,
        featmap_size: tuple[int, int],
        level_idx: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
    ) -> Tensor:
        """Generate sparse priors from given indices.

        Args:
            prior_idxs (Tensor): Indices of the priors.
            featmap_size (tuple[int, int]): (H, W) of the feature map.
            level_idx (int): Index of the feature map level.
            dtype (torch.dtype): Data type of the output.
            device (str | torch.device): Device for the output.

        Returns:
            Tensor: Selected points with shape (N, 2).
        """
        height, width = featmap_size
        stride_h, stride_w = self.strides[level_idx]

        x = ((prior_idxs % width).to(dtype) + self.offset) * stride_w
        y = ((prior_idxs // width).to(dtype) + self.offset) * stride_h

        return torch.stack([x, y], dim=-1)
