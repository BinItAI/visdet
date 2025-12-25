# Copyright (c) OpenMMLab. All rights reserved.
import torch
from visdet.structures import DetDataSample, InstanceData
from visdet.registry import MODELS


def test_yolov3_forward_backward():
    model_cfg = dict(
        type="YOLOV3",
        backbone=dict(
            type="ResNet",
            depth=18,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type="BN", requires_grad=True),
            style="pytorch",
        ),
        neck=dict(
            type="FPN",  # YOLOv3 usually uses custom neck, but can work with FPN outputs
            in_channels=[128, 256, 512],
            out_channels=256,
            num_outs=3,
        ),
        bbox_head=dict(
            type="YOLOV3Head",
            num_classes=80,
            in_channels=[256, 256, 256],
            out_channels=[512, 256, 128],
            anchor_generator=dict(
                type="YOLOAnchorGenerator",
                base_sizes=[
                    [(116, 90), (156, 198), (373, 326)],
                    [(30, 61), (62, 45), (59, 119)],
                    [(10, 13), (16, 30), (33, 23)],
                ],
                strides=[32, 16, 8],
            ),
            bbox_coder=dict(type="YOLOBBoxCoder"),
            featmap_strides=[32, 16, 8],
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
            loss_conf=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
            loss_xy=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
            loss_wh=dict(type="MSELoss", loss_weight=1.0),
        ),
        # training and testing settings
        train_cfg=dict(),
        test_cfg=dict(nms_pre=1000, score_thr=0.05, nms=dict(type="nms", iou_threshold=0.45), max_per_img=100),
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

    # Backward pass
    total_loss = 0
    for loss_value in losses.values():
        if isinstance(loss_value, torch.Tensor):
            total_loss += loss_value.sum()
        elif isinstance(loss_value, list):
            total_loss += sum(l.sum() for l in loss_value)

    total_loss.backward()
    assert imgs.grad is not None
    print("YOLOv3 forward/backward passed!")


def test_yolox_forward_backward():
    model_cfg = dict(
        type="YOLOX",
        backbone=dict(
            type="ResNet",
            depth=18,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type="BN", requires_grad=True),
            style="pytorch",
        ),
        neck=dict(type="FPN", in_channels=[128, 256, 512], out_channels=256, num_outs=3),
        bbox_head=dict(
            type="YOLOXHead",
            num_classes=80,
            in_channels=256,
            feat_channels=256,
            stacked_convs=2,
            strides=[8, 16, 32],
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type="IoULoss", mode="square", eps=1e-16, loss_weight=5.0),
            loss_obj=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
            loss_l1=dict(type="L1Loss", loss_weight=1.0),
        ),
        # training and testing settings
        train_cfg=dict(),
        test_cfg=dict(nms_pre=1000, score_thr=0.05, nms=dict(type="nms", iou_threshold=0.45), max_per_img=100),
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

    # Backward pass
    total_loss = 0
    for loss_value in losses.values():
        if isinstance(loss_value, torch.Tensor):
            total_loss += loss_value.sum()
        elif isinstance(loss_value, list):
            total_loss += sum(l.sum() for l in loss_value)

    total_loss.backward()
    assert imgs.grad is not None
    print("YOLOX forward/backward passed!")


if __name__ == "__main__":
    test_yolov3_forward_backward()
    test_yolox_forward_backward()
