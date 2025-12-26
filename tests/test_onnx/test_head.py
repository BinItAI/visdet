# Copyright (c) OpenMMLab. All rights reserved.

"""ONNX export smoke tests using YAML presets.

The original tests referenced `configs/**/*.py` experiment configs. Those have
been removed; we now load the model dicts from `configs/presets/models/*.yaml`.

These tests only check that `torch.onnx.export` can run.
"""

from __future__ import annotations

from functools import partial

import pytest
import torch

from visdet import digit_version
from visdet.engine.fileio import load
from visdet.models import build_detector


if digit_version(torch.__version__) <= digit_version("1.5.0"):
    pytest.skip("onnx export does not support torch<=1.5.0", allow_module_level=True)


def _model_from_preset_yaml(path: str) -> dict:
    cfg = load(path)
    cfg.pop("preset_meta", None)
    cfg.get("backbone", {}).pop("init_cfg", None)
    return cfg


def test_cascade_onnx_export():
    model_cfg = _model_from_preset_yaml("configs/presets/models/cascade_rcnn_r50.yaml")
    model = build_detector(model_cfg)

    with torch.no_grad():
        model.forward = partial(model.forward, img_metas=[[dict()]])
        torch.onnx.export(
            model,
            [torch.rand(1, 3, 128, 128)],
            "tmp.onnx",
            output_names=["dets", "labels"],
            input_names=["input_img"],
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11,
        )


def test_faster_onnx_export():
    model_cfg = _model_from_preset_yaml("configs/presets/models/faster_rcnn_r50.yaml")
    model = build_detector(model_cfg)

    with torch.no_grad():
        model.forward = partial(model.forward, img_metas=[[dict()]])
        torch.onnx.export(
            model,
            [torch.rand(1, 3, 128, 128)],
            "tmp.onnx",
            output_names=["dets", "labels"],
            input_names=["input_img"],
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11,
        )
