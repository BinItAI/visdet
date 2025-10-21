# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from visengine.registry import MODELS

HSwish = nn.Hardswish
MODELS.register_module(module=nn.Hardswish, name="HSwish")
