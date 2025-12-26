# Copyright (c) OpenMMLab. All rights reserved.
"""Forward-pass smoke tests using YAML presets.

The upstream test suite loaded detectors from `configs/**/*.py` experiment
configs. visdet has removed the Python config zoo, so these tests now build
models from `configs/presets/models/*.yaml` via `MODEL_PRESETS`.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from visdet.presets import MODEL_PRESETS


def _load_model_preset(preset_name: str) -> dict:
    cfg = copy.deepcopy(MODEL_PRESETS.get(preset_name))
    cfg.pop("preset_meta", None)

    # Avoid downloading pretrained weights during unit tests.
    backbone = cfg.get("backbone")
    if isinstance(backbone, dict):
        backbone.pop("init_cfg", None)

    return cfg


def _demo_mm_inputs(
    input_shape: tuple[int, int, int, int] = (1, 3, 128, 128),
    num_items: list[int] | None = None,
    num_classes: int = 10,
) -> dict:
    """Create a superset of inputs needed to run test or train batches."""

    from visdet.core import BitmapMasks

    n, c, h, w = input_shape
    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)

    img_metas = [
        {
            "img_shape": (h, w, c),
            "ori_shape": (h, w, c),
            "pad_shape": (h, w, c),
            "filename": "<demo>.png",
            "scale_factor": np.array([1.0, 1.0, 1.0, 1.0]),
            "flip": False,
            "flip_direction": None,
        }
        for _ in range(n)
    ]

    gt_bboxes = []
    gt_labels = []
    gt_masks = []

    for batch_idx in range(n):
        num_boxes = rng.randint(1, 5) if num_items is None else num_items[batch_idx]

        cx, cy, bw, bh = rng.rand(num_boxes, 4).T
        tl_x = ((cx * w) - (w * bw / 2)).clip(0, w)
        tl_y = ((cy * h) - (h * bh / 2)).clip(0, h)
        br_x = ((cx * w) + (w * bw / 2)).clip(0, w)
        br_y = ((cy * h) + (h * bh / 2)).clip(0, h)

        boxes = np.vstack([tl_x, tl_y, br_x, br_y]).T
        class_idxs = rng.randint(1, num_classes, size=num_boxes)

        gt_bboxes.append(torch.FloatTensor(boxes))
        gt_labels.append(torch.LongTensor(class_idxs))

        mask = rng.randint(0, 2, (num_boxes, h, w), dtype=np.uint8)
        gt_masks.append(BitmapMasks(mask, h, w))

    return {
        "imgs": torch.FloatTensor(imgs).requires_grad_(True),
        "img_metas": img_metas,
        "gt_bboxes": gt_bboxes,
        "gt_labels": gt_labels,
        "gt_masks": gt_masks,
        "gt_bboxes_ignore": None,
    }


@pytest.mark.parametrize(
    "preset_name",
    [
        "faster_rcnn_r50",
        "mask_rcnn_r50",
        "cascade_rcnn_r50",
        "rtmdet_s",
    ],
)
def test_forward_train_and_test(preset_name: str):
    from visdet.models import build_detector

    model_cfg = _load_model_preset(preset_name)
    detector = build_detector(model_cfg)

    mm_inputs = _demo_mm_inputs(input_shape=(1, 3, 128, 128), num_items=[3])

    imgs = mm_inputs.pop("imgs")
    img_metas = mm_inputs.pop("img_metas")

    detector.train()

    # Two-stage detectors need masks for Mask R-CNN; others ignore gt_masks.
    losses = detector.forward(imgs, img_metas, return_loss=True, **mm_inputs)
    assert isinstance(losses, dict)

    loss, _ = detector._parse_losses(losses)
    assert float(loss.item()) >= 0

    detector.eval()
    with torch.no_grad():
        result = detector.forward([imgs[0]], [[img_metas[0]]], return_loss=False)
    assert result is not None
