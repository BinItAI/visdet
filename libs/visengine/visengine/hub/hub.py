# ruff: noqa
# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os.path as osp

from visengine.config import Config
from visengine.config.utils import (
    _get_cfg_metainfo,
    _get_external_cfg_base_path,
    _get_package_and_cfg_path,
)
from visengine.registry import MODELS, DefaultScope
from visengine.runner import load_checkpoint
from visengine.utils import get_installed_path, install_package
from ml_env_config.env import env
from vision.tools.logger import logger
from pathlib import Path


def get_config(cfg_path: str, pretrained: bool = False) -> Config:
    """Get config from external package.

    Args:
        cfg_path (str): External relative config path.
        pretrained (bool): Whether to save pretrained model path. If
            ``pretrained==True``, the url of pretrained model can be accessed
            by ``cfg.model_path``. Defaults to False.

    Examples:
        >>> cfg = get_config('mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py', pretrained=True)
        >>> # Equivalent to
        >>> # cfg = Config.fromfile('/path/to/faster-rcnn_r50_fpn_1x_coco.py')
        >>> cfg.model_path
        https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

    Returns:
        Config: A `Config` parsed from external package.
    """
    # Get package name and relative config path.
    package, cfg_path = _get_package_and_cfg_path(cfg_path)
    # Handle the renaming from mmdet to visdet
    if package == "mmdet":
        package = "visdet"
    package_path = osp.join(osp.dirname(osp.abspath(importlib.import_module(package).__file__)))
    try:
        # Use `cfg_path` to search target config file.
        cfg_meta = _get_cfg_metainfo(package_path, cfg_path)
        cfg_basepath = Path(package_path).parent
        cfg_path = osp.join(cfg_basepath, cfg_meta["Config"])
        logger.info(f"Config path --> {cfg_path}")
        cfg = Config.fromfile(cfg_path)
        if pretrained:
            assert "Weights" in cfg_meta, "Cannot find `Weights` in cfg_file.metafile.yml, please check themetafile"
            cfg.model_path = cfg_meta["Weights"]
    except ValueError:
        # Since the base config does not contain a metafile, the absolute
        # config is `osp.join(package_path, cfg_path_prefix, cfg_name)`
        cfg_path = _get_external_cfg_base_path(package_path, cfg_path)
        cfg = Config.fromfile(cfg_path)
    except Exception as e:
        raise e
    return cfg


def get_model(cfg_path: str, pretrained: bool = False, **kwargs):
    """Get built model from external package.

    Args:
        cfg_path (str): External relative config path with prefix
            'package::' and without suffix.
        pretrained (bool): Whether to load pretrained model. Defaults to False.
        kwargs (dict): Default arguments to build model.

    Examples:
        >>> model = get_model('mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py', pretrained=True)
        >>> type(model)
        <class 'mmdet.models.detectors.faster_rcnn.FasterRCNN'>

    Returns:
        nn.Module: Built model.
    """
    package = cfg_path.split("::")[0]
    with DefaultScope.overwrite_default_scope(package):  # type: ignore
        cfg = get_config(cfg_path, pretrained)
        if "data_preprocessor" in cfg:
            cfg.model.data_preprocessor = cfg.data_preprocessor
        models_module = importlib.import_module(f"{package}.utils")
        models_module.register_all_modules()  # type: ignore
        model = MODELS.build(cfg.model, default_args=kwargs)
        if pretrained:
            load_checkpoint(model, cfg.model_path)
            # Hack to use pretrained weights.
            # If we do not set _is_init here, Runner will call
            # `model.init_weights()` to overwrite the pretrained model.
            model._is_init = True
        return model
