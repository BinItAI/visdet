# ruff: noqa
from visdet.models.backbones.hrnet import HRNet
from visdet.models.backbones.regnet import RegNet
from visdet.models.backbones.res2net import Res2Net
from visdet.models.backbones.resnet import ResNet
from visdet.models.backbones.resnest import ResNeSt
from visdet.models.backbones.resnext import ResNeXt
from visdet.models.backbones.swin import SwinTransformer
from visdet.models.backbones.trident_resnet import TridentResNet

__all__ = [
    "HRNet",
    "RegNet",
    "Res2Net",
    "ResNet",
    "ResNeSt",
    "ResNeXt",
    "SwinTransformer",
    "TridentResNet",
]
