import os
from pathlib import Path

import pytest

from visdet.apis import init_detector

pytestmark = pytest.mark.skip(
    reason="Test requires ResNet backbone configs which are out of scope for minimal visdet (Swin + Mask R-CNN only)"
)


def test_init_detector():
    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    project_dir = os.path.join(project_dir, "..")

    config_file = os.path.join(project_dir, "configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py")

    # test init_detector with config_file: str and cfg_options
    cfg_options = dict(
        model=dict(
            backbone=dict(
                depth=18,
                init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet18"),
            )
        )
    )
    model = init_detector(config_file, device="cpu", cfg_options=cfg_options)

    # test init_detector with :obj:`Path`
    config_path_object = Path(config_file)
    model = init_detector(config_path_object, device="cpu")

    # test init_detector with undesirable type
    with pytest.raises(TypeError):
        config_list = [config_file]
        model = init_detector(config_list)  # noqa: F841
