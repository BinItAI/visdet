# ruff: noqa
# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import json
import os.path as osp

from visdet.engine.config import Config
from visdet.engine.config.utils import (
    _get_cfg_metainfo,
    _get_external_cfg_base_path,
    _get_package_and_cfg_path,
)
from visdet.engine.registry import MODELS, DefaultScope
from visdet.engine.runner import load_checkpoint
from visdet.engine.utils import get_installed_path, install_package
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
    # visdet no longer mirrors the MMDetection python config zoo.
    # Keep `package` unchanged so external configs resolve from their own package.
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


def _load_huggingface_mapping() -> dict[str, str]:
    """Load the HuggingFace model URL mapping."""
    hub_dir = osp.dirname(osp.abspath(__file__))
    hf_json_path = osp.join(hub_dir, "huggingface.json")
    if osp.exists(hf_json_path):
        with open(hf_json_path) as f:
            return json.load(f)
    return {}


def get_huggingface_url(model_key: str) -> str | None:
    """Get the HuggingFace URL for a model key.

    Args:
        model_key: Key from huggingface.json (e.g., "resnet50", "swin_tiny")

    Returns:
        HuggingFace URL string (e.g., "hf://GeorgePearse/visdet-weights/...")
        or None if not found.

    Examples:
        >>> url = get_huggingface_url("resnet50")
        >>> print(url)
        hf://GeorgePearse/visdet-weights/mmcls/resnet50_8xb32_in1k_20210831-ea4938fc.pth
    """
    mapping = _load_huggingface_mapping()
    return mapping.get(model_key)


def list_huggingface_models() -> list[str]:
    """List all available model keys in the HuggingFace mapping.

    Returns:
        List of model key strings.
    """
    mapping = _load_huggingface_mapping()
    return list(mapping.keys())
