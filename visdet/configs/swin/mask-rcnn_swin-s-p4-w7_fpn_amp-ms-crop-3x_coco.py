# ruff: noqa
# type: ignore
_base_ = "./mask-rcnn_swin-t-p4-w7_fpn_amp-ms-crop-3x_coco.py"
pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth"
model = {
    "backbone": {
        "depths": [2, 2, 18, 2],
        "init_cfg": {"type": "Pretrained", "checkpoint": pretrained},
    }
}
