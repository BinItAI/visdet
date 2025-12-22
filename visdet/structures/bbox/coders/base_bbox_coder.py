# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Any

from torch import Tensor


class BaseBBoxCoder(metaclass=ABCMeta):
    """Base bounding box coder.

    Args:
        use_box_type (bool): Whether to warp decoded boxes with the
            box type data structure. Defaults to False.
    """

    # The size of the last of dimension of the encoded tensor.
    encode_size = 4

    def __init__(self, use_box_type: bool = False, **kwargs: Any) -> None:
        self.use_box_type = use_box_type

    @abstractmethod
    def encode(self, bboxes: Tensor, gt_bboxes: Tensor) -> Tensor:
        """Encode deltas between bboxes and ground truth boxes."""

    @abstractmethod
    def decode(self, bboxes: Tensor, bboxes_pred: Tensor) -> Tensor:
        """Decode the predicted bboxes according to prediction and base
        boxes."""
