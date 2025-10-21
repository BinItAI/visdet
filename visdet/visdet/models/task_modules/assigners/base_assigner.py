# ruff: noqa
# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from visengine.structures import InstanceData


class BaseAssigner(metaclass=ABCMeta):
    """Base assigner that assigns boxes to ground truth boxes."""

    @abstractmethod
    def assign(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        gt_instances_ignore: InstanceData | None = None,
        **kwargs,
    ):
        """Assign boxes to either a ground truth boxes or a negative boxes."""
