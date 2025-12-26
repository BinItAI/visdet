# Copyright (c) OpenMMLab. All rights reserved.

"""Multi-scale augmentation tests without Python configs.

This test file used to load full experiment configs from `configs/**/*.py`.
Those have been removed; we now build models from YAML presets.
"""

from __future__ import annotations

import copy
import os.path as osp

from visdet.cv import build_from_cfg
from visdet.datasets.builder import PIPELINES
from visdet.engine.fileio import load
from visdet.models import build_detector


def _build_model_from_preset(preset_yaml: str):
    cfg = copy.deepcopy(load(preset_yaml))
    cfg.pop("preset_meta", None)
    cfg.get("backbone", {}).pop("init_cfg", None)
    cfg.setdefault("train_cfg", None)
    return build_detector(cfg)


def _model_aug_test_template(preset_yaml: str):
    model = _build_model_from_preset(preset_yaml)

    load_cfg = dict(type="LoadImageFromFile")
    multi_scale_cfg = dict(
        type="MultiScaleFlipAug",
        transforms=[],
        img_scale=[(1333, 800), (800, 600), (640, 480)],
        flip=True,
        flip_direction=["horizontal", "vertical", "diagonal"],
    )

    load_transform = build_from_cfg(load_cfg, PIPELINES)
    multi_scale_transform = build_from_cfg(multi_scale_cfg, PIPELINES)

    results = dict(
        img_prefix=osp.join(osp.dirname(__file__), "../../../data"),
        img_info=dict(filename="color.jpg"),
    )

    results = load_transform(results)
    results = multi_scale_transform(results)

    assert len(results["img"]) == 12

    # Make sure the model actually builds.
    assert model is not None
    assert hasattr(model, "forward")


def test_aug_test_size():
    # validate MultiScaleFlipAug expansion
    load_cfg = dict(type="LoadImageFromFile")
    multi_scale_cfg = dict(
        type="MultiScaleFlipAug",
        transforms=[],
        img_scale=[(1333, 800), (800, 600), (640, 480)],
        flip=True,
        flip_direction=["horizontal", "vertical", "diagonal"],
    )

    load_transform = build_from_cfg(load_cfg, PIPELINES)
    multi_scale_transform = build_from_cfg(multi_scale_cfg, PIPELINES)

    results = dict(
        img_prefix=osp.join(osp.dirname(__file__), "../../../data"),
        img_info=dict(filename="color.jpg"),
    )

    results = load_transform(results)
    results = multi_scale_transform(results)

    assert len(results["img"]) == 12


def test_faster_rcnn_preset_builds():
    _model_aug_test_template("configs/models/faster_rcnn_r50.yaml")
