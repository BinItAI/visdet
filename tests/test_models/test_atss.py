# Copyright (c) OpenMMLab. All rights reserved.
import torch

from visdet.registry import MODELS
from visdet.structures import DetDataSample, InstanceData


def test_atss_forward_backward():
    model_cfg = dict(
        type="ATSS",
        backbone=dict(
            type="ResNet",
            depth=18,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type="BN", requires_grad=True),
            norm_eval=True,
            style="pytorch",
        ),
        neck=dict(
            type="FPN",
            in_channels=[64, 128, 256, 512],
            out_channels=256,
            start_level=1,
            add_extra_convs="on_output",
            num_outs=5,
        ),
        bbox_head=dict(
            type="ATSSHead",
            num_classes=80,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            anchor_generator=dict(
                type="AnchorGenerator",
                octave_base_scale=8,
                scales_per_octave=1,
                ratios=[1.0],
                strides=[8, 16, 32, 64, 128],
            ),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
            loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
            loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        ),
        # training and testing settings
        train_cfg=dict(assigner=dict(type="ATSSAssigner", topk=9), allowed_border=-1, pos_weight=-1, debug=False),
        test_cfg=dict(
            nms_pre=1000, min_bbox_size=0, score_thr=0.05, nms=dict(type="nms", iou_threshold=0.6), max_per_img=100
        ),
    )

    detector = MODELS.build(model_cfg)

    # Test forward train
    imgs = torch.randn(2, 3, 224, 224, requires_grad=True)
    data_samples = []
    for i in range(2):
        data_sample = DetDataSample()
        data_sample.set_metainfo(
            dict(img_shape=(224, 224), ori_shape=(224, 224), pad_shape=(224, 224), scale_factor=(1.0, 1.0))
        )
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
        gt_instances.labels = torch.tensor([1], dtype=torch.long)
        data_sample.gt_instances = gt_instances
        data_samples.append(data_sample)

    losses = detector(imgs, data_samples, mode="loss")
    assert isinstance(losses, dict)
    assert "loss_cls" in losses
    assert "loss_bbox" in losses
    assert "loss_centerness" in losses

    # Backward pass
    total_loss = 0
    for loss_value in losses.values():
        if isinstance(loss_value, torch.Tensor):
            total_loss += loss_value.sum()
        elif isinstance(loss_value, list):
            total_loss += sum(l.sum() for l in loss_value)

    total_loss.backward()
    assert imgs.grad is not None
    print("Backward pass passed!")

    # Test forward predict
    detector.eval()
    with torch.no_grad():
        predictions = detector(imgs, data_samples, mode="predict")
        assert len(predictions) == 2
        assert isinstance(predictions[0], DetDataSample)
        assert hasattr(predictions[0], "pred_instances")
    print("Forward predict passed!")

    print("ATSS forward and backward tests passed!")


if __name__ == "__main__":
    test_atss_forward_backward()
