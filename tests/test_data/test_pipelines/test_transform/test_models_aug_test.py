# Copyright (c) OpenMMLab. All rights reserved.

"""Multi-scale augmentation tests without Python configs.

The previous version of this test module loaded full experiment configs from
`configs/**/*.py`. Those have been removed in favor of YAML presets.

This file now focuses on validating `MultiScaleFlipAug` behavior directly.
"""

from __future__ import annotations

import os.path as osp

from visdet.cv import build_from_cfg
from visdet.datasets.builder import PIPELINES


def test_aug_test_size():
    results = dict(
        img_prefix=osp.join(osp.dirname(__file__), "../../../data"),
        img_info=dict(filename="color.jpg"),
    )

    # Define simple load pipeline
    load = dict(type="LoadImageFromFile")
    load = build_from_cfg(load, PIPELINES)

    # Multi-scale + flip augmentation
    transform = dict(
        type="MultiScaleFlipAug",
        transforms=[],
        img_scale=[(1333, 800), (800, 600), (640, 480)],
        flip=True,
        flip_direction=["horizontal", "vertical", "diagonal"],
    )
    multi_aug_test_module = build_from_cfg(transform, PIPELINES)

    results = load(results)
    results = multi_aug_test_module(load(results))

    # len([original, h, v, d]) * len(scales)
    assert len(results["img"]) == 12
