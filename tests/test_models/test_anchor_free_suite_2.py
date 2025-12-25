# Copyright (c) OpenMMLab. All rights reserved.
import torch

from visdet.registry import MODELS
from visdet.structures import DetDataSample, InstanceData


def test_foveabox_forward_backward():
    model_cfg = dict(
        type="FoveaBox",
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
        neck=dict(type="FPN", in_channels=[64, 128, 256, 512], out_channels=256, start_level=1, num_outs=5),
        bbox_head=dict(
            type="FoveaHead",
            num_classes=80,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            strides=[8, 16, 32, 64, 128],
            base_edge_list=[16, 32, 64, 128, 256],
            scale_ranges=((1, 64), (32, 128), (64, 256), (128, 512), (256, 2048)),
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type="SmoothL1Loss", beta=0.11, loss_weight=1.0),
        ),
        # training and testing settings
        train_cfg=dict(),
        test_cfg=dict(nms_pre=1000, score_thr=0.05, nms=dict(type="nms", iou_threshold=0.5), max_per_img=100),
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

    # Backward pass
    total_loss = 0
    for loss_value in losses.values():
        if isinstance(loss_value, torch.Tensor):
            total_loss += loss_value.sum()
        elif isinstance(loss_value, list):
            total_loss += sum(l.sum() for l in loss_value)

    total_loss.backward()
    assert imgs.grad is not None
    print("FoveaBox forward/backward passed!")


def test_fsaf_forward_backward():
    model_cfg = dict(
        type="FSAF",
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
            add_extra_convs="on_input",
            num_outs=5,
        ),
        bbox_head=dict(
            type="FSAFHead",
            num_classes=80,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            anchor_generator=dict(
                type="AnchorGenerator",
                octave_base_scale=1,
                scales_per_octave=1,
                ratios=[1.0],
                strides=[8, 16, 32, 64, 128],
            ),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0]
            ),
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type="IoULoss", loss_weight=1.0),
        ),
        # training and testing settings
        train_cfg=dict(
            assigner=dict(type="MaxIoUAssigner", pos_iou_thr=0.5, neg_iou_thr=0.4, min_pos_iou=0, ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        test_cfg=dict(
            nms_pre=1000, min_bbox_size=0, score_thr=0.05, nms=dict(type="nms", iou_threshold=0.5), max_per_img=100
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

    # Backward pass
    total_loss = 0
    for loss_value in losses.values():
        if isinstance(loss_value, torch.Tensor):
            total_loss += loss_value.sum()
        elif isinstance(loss_value, list):
            total_loss += sum(l.sum() for l in loss_value)

    total_loss.backward()
    assert imgs.grad is not None
    print("FSAF forward/backward passed!")


def test_centernet_forward_backward():
    model_cfg = dict(
        type="CenterNet",
        backbone=dict(
            type="ResNet",
            depth=18,
            num_stages=4,
            out_indices=(3,),
            frozen_stages=1,
            norm_cfg=dict(type="BN", requires_grad=True),
            norm_eval=True,
            style="pytorch",
        ),
        neck=None,
        bbox_head=dict(
            type="CenterNetHead",
            in_channel=512,
            feat_channel=256,
            num_classes=80,
            loss_center_heatmap=dict(type="GaussianFocalLoss", loss_weight=1.0),
            loss_wh=dict(type="L1Loss", loss_weight=0.1),
            loss_offset=dict(type="L1Loss", loss_weight=1.0),
        ),
        train_cfg=None,
        test_cfg=dict(topk=100, local_maximum_kernel=3),
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
    assert "loss_center_heatmap" in losses

    # Backward pass
    total_loss = 0
    for loss_value in losses.values():
        if isinstance(loss_value, torch.Tensor):
            total_loss += loss_value.sum()
        elif isinstance(loss_value, list):
            total_loss += sum(l.sum() for l in loss_value)

    total_loss.backward()
    assert imgs.grad is not None
    print("CenterNet forward/backward passed!")


if __name__ == "__main__":
    test_foveabox_forward_backward()
    test_fsaf_forward_backward()
    test_centernet_forward_backward()
