# Copyright (c) OpenMMLab. All rights reserved.

"""Runtime config/pipeline smoke tests for YAML presets."""

from __future__ import annotations

import numpy as np

from visdet.engine.fileio import load


def test_dataset_preset_has_pipelines():
    preset = load("configs/presets/datasets/coco_detection.yaml")
    assert preset["type"] == "CocoDataset"
    assert "train_pipeline" in preset
    assert "test_pipeline" in preset


def test_model_preset_has_expected_fields():
    preset = load("configs/presets/models/faster_rcnn_r50.yaml")
    assert preset["type"] == "FasterRCNN"
    assert "backbone" in preset
    assert "roi_head" in preset


def test_pipeline_compose_smoke():
    """Compose a minimal pipeline from dataset preset dicts."""

    from visdet.datasets.pipelines import Compose

    dset = load("configs/presets/datasets/coco_detection.yaml")
    pipeline = Compose(dset["train_pipeline"])

    img = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    results = {
        "img": img,
        "img_path": "dummy.png",
        "img_shape": img.shape,
        "ori_shape": img.shape,
        "instances": [],
    }

    out = pipeline(results)
    assert out is not None
