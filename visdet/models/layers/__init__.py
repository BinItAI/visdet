# ruff: noqa
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from visdet.engine.model import BaseModule
from visdet.engine.utils import to_2tuple
from visdet.models.layers.bbox_nms import multiclass_nms
from visdet.models.layers.transformer import (
    MultiheadAttention,
    Transformer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from . import normed_predictor  # This registers Linear in MODELS registry

# ... existing code ...

__all__ = [
    "AdaptivePadding",
    "PatchEmbed",
    "PatchMerging",
    "multiclass_nms",
    "MultiheadAttention",
    "Transformer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerEncoderLayer",
]
