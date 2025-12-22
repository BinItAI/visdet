# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch.nn as nn
from torchvision.ops import RoIAlign as TVRoIAlign
from torchvision.ops import roi_align as tv_roi_align


class RoIAlign(nn.Module):
    """RoI align pooling layer using torchvision's implementation.

    Args:
        output_size (tuple): h, w
        spatial_scale (float): scale the input boxes by this number
        sampling_ratio (int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
        pool_mode (str): pooling mode in each bin.
        aligned (bool): if False, use the legacy implementation in
            MMDetection. If True, align the results more perfectly.
        use_torchvision (bool): whether to use torchvision's implementation.
            We set this to True by default for better performance.
    """

    output_size: Union[int, Tuple[int, int]]
    spatial_scale: float
    sampling_ratio: int
    pool_mode: str
    aligned: bool
    use_torchvision: bool
    roi_align: TVRoIAlign

    def __init__(
        self,
        output_size: Union[int, Tuple[int, int]],
        spatial_scale: float = 1.0,
        sampling_ratio: int = 0,
        pool_mode: str = "avg",
        aligned: bool = True,
        use_torchvision: bool = True,
    ) -> None:
        super().__init__()
        self.output_size = output_size  # type: ignore[misc]
        self.spatial_scale = spatial_scale  # type: ignore[misc]
        self.sampling_ratio = sampling_ratio  # type: ignore[misc]
        self.pool_mode = pool_mode  # type: ignore[misc]
        self.aligned = aligned  # type: ignore[misc]
        self.use_torchvision = use_torchvision  # type: ignore[misc]

        if isinstance(self.output_size, int):
            self.output_size = (self.output_size, self.output_size)  # type: ignore[misc]

        # We always use torchvision's implementation for simplicity
        self.roi_align = TVRoIAlign(
            output_size=self.output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio,
            aligned=self.aligned,
        )

    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N.
                The other 4 columns are xyxy.
        """
        return self.roi_align(input, rois)

    def __repr__(self):
        s = self.__class__.__name__
        s += f"(output_size={self.output_size}, "
        s += f"spatial_scale={self.spatial_scale}, "
        s += f"sampling_ratio={self.sampling_ratio}, "
        s += f"pool_mode={self.pool_mode}, "
        s += f"aligned={self.aligned}, "
        s += f"use_torchvision={self.use_torchvision})"
        return s


# Functional interface
def roi_align(
    input,
    rois,
    output_size,
    spatial_scale=1.0,
    sampling_ratio=0,
    pool_mode="avg",
    aligned=True,
):
    """RoI align pooling layer functional interface.

    Args:
        input (Tensor): input tensor.
        rois (Tensor): RoIs tensor.
        output_size (tuple): h, w
        spatial_scale (float): scale the input boxes by this number
        sampling_ratio (int): number of inputs samples to take for each
            output sample. 0 to take samples densely.
        pool_mode (str): pooling mode in each bin.
        aligned (bool): if False, use the legacy implementation in
            MMDetection. If True, align the results more perfectly.

    Returns:
        Tensor: RoI align pooling result.
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    # Use torchvision's roi_align directly
    return tv_roi_align(input, rois, output_size, spatial_scale, sampling_ratio, aligned)
