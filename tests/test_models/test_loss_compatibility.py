# Copyright (c) OpenMMLab. All rights reserved.
"""Loss API compatibility tests using YAML presets.

This replaces the old config-driven tests that modified `configs/_base_/*.py`.
"""

from __future__ import annotations

import copy

import pytest

from visdet.presets import MODEL_PRESETS


def _get_faster_rcnn_cfg() -> dict:
    cfg = copy.deepcopy(MODEL_PRESETS.get("faster_rcnn_r50"))
    cfg.pop("preset_meta", None)
    # Avoid downloading pretrained weights.
    cfg.get("backbone", {}).pop("init_cfg", None)
    return cfg


@pytest.mark.parametrize(
    "loss_bbox",
    [
        dict(type="L1Loss", loss_weight=1.0),
        dict(type="SmoothL1Loss", loss_weight=1.0),
        dict(type="GIoULoss", loss_weight=1.0),
        dict(type="DIoULoss", loss_weight=1.0),
        dict(type="CIoULoss", loss_weight=1.0),
        dict(type="MSELoss", loss_weight=1.0),
        dict(type="BalancedL1Loss", loss_weight=1.0),
    ],
)
def test_bbox_loss_compatibility(loss_bbox: dict):
    import torch

    from visdet.models import build_detector

    model_cfg = _get_faster_rcnn_cfg()
    model_cfg["roi_head"]["bbox_head"]["loss_bbox"] = loss_bbox

    detector = build_detector(model_cfg)
    detector.train()

    # Minimal dummy batch
    imgs = torch.randn(1, 3, 64, 64)
    img_metas = [
        {
            "img_shape": (64, 64, 3),
            "ori_shape": (64, 64, 3),
            "pad_shape": (64, 64, 3),
            "scale_factor": (1.0, 1.0, 1.0, 1.0),
            "flip": False,
            "flip_direction": None,
        }
    ]

    gt_bboxes = [torch.zeros((0, 4))]
    gt_labels = [torch.zeros((0,), dtype=torch.long)]

    losses = detector.forward(imgs, img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels, return_loss=True)
    assert isinstance(losses, dict)
