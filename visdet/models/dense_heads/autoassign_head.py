# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from visdet.cv.cnn import ConvModule, Scale
from visdet.models.utils.misc import multi_apply
from visdet.registry import MODELS, TASK_UTILS
from visdet.utils import reduce_mean

from .fcos_head import FCOSHead


@MODELS.register_module()
class AutoAssignHead(FCOSHead):
    """AutoAssignHead head used in AutoAssign.

    https://arxiv.org/abs/2007.03496
    """

    def __init__(
        self,
        *args,
        force_topk=False,
        topk=9,
        pos_loss_weight=0.25,
        neg_loss_weight=0.75,
        center_loss_weight=0.75,
        **kwargs,
    ):
        super().__init__(*args, conv_bias=True, **kwargs)
        self.pos_loss_weight = pos_loss_weight
        self.neg_loss_weight = neg_loss_weight
        self.center_loss_weight = center_loss_weight
        # Simplified implementation: Reuse FCOS logic where possible
        # but AutoAssign has specific assignment logic (CenterPrior)
        # For now, we will rely on FCOSHead behavior as a baseline
        # but with AutoAssign configuration compatibility.

    # TODO: Implement CenterPrior and full AutoAssign logic
