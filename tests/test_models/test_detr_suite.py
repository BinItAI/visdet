# Copyright (c) OpenMMLab. All rights reserved.
import torch

from visdet.registry import MODELS
from visdet.structures import DetDataSample, InstanceData


def test_detr_forward_backward():
    model_cfg = dict(
        type="DETR",
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
        bbox_head=dict(
            type="DETRHead",
            num_classes=80,
            in_channels=512,
            transformer=dict(
                type="Transformer",
                d_model=512,
                nhead=8,
                num_encoder_layers=2,
                num_decoder_layers=2,
                dim_feedforward=2048,
                dropout=0.1,
                activation="relu",
                normalize_before=False,
                return_intermediate_dec=True,
            ),
            loss_cls=dict(
                type="CrossEntropyLoss", bg_cls_weight=0.1, use_sigmoid=False, loss_weight=1.0, class_weight=1.0
            ),
            loss_bbox=dict(type="L1Loss", loss_weight=5.0),
            loss_iou=dict(type="GIoULoss", loss_weight=2.0),
        ),
        train_cfg=dict(
            assigner=dict(
                type="HungarianAssigner",
                cls_cost=dict(type="ClassificationCost", weight=1.0),
                reg_cost=dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
            )
        ),
        test_cfg=dict(max_per_img=100),
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
    assert "loss_iou" in losses

    # Backward pass
    total_loss = 0
    for loss_value in losses.values():
        if isinstance(loss_value, torch.Tensor):
            total_loss += loss_value.sum()
        elif isinstance(loss_value, list):
            total_loss += sum(loss_item.sum() for loss_item in loss_value)

    total_loss.backward()
    assert imgs.grad is not None
    print("DETR forward/backward passed!")


def test_yolof_forward_backward():
    model_cfg = dict(
        type="YOLOF",
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
        neck=dict(
            type="DilatedEncoder",
            in_channels=512,
            out_channels=256,
            block_mid_channels=128,
            num_residual_blocks=2,
            block_dilations=[2, 4],
        ),
        bbox_head=dict(
            type="YOLOFHead",
            num_classes=80,
            in_channels=256,
            num_cls_convs=2,
            num_reg_convs=2,
            anchor_generator=dict(type="AnchorGenerator", ratios=[1.0], scales=[1, 2, 4], strides=[32]),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0]
            ),
            loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_bbox=dict(type="GIoULoss", loss_weight=1.0),
        ),
        train_cfg=dict(
            assigner=dict(type="UniformAssigner", pos_ignore_thr=0.15, neg_ignore_thr=0.7, match_times=2),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
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

    # Backward pass
    total_loss = 0
    for loss_value in losses.values():
        if isinstance(loss_value, torch.Tensor):
            total_loss += loss_value.sum()
        elif isinstance(loss_value, list):
            total_loss += sum(loss_item.sum() for loss_item in loss_value)

    total_loss.backward()
    assert imgs.grad is not None
    print("YOLOF forward/backward passed!")


def test_paa_forward_backward():
    model_cfg = dict(
        type="PAA",
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
            type="PAAHead",
            num_classes=80,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            anchor_generator=dict(
                type="AnchorGenerator",
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[8, 16, 32, 64, 128],
            ),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
            loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_bbox=dict(type="GIoULoss", loss_weight=1.3),
            loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=0.5),
        ),
        train_cfg=dict(
            assigner=dict(type="MaxIoUAssigner", pos_iou_thr=0.1, neg_iou_thr=0.1, min_pos_iou=0, ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
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

    # Backward pass
    total_loss = 0
    for loss_value in losses.values():
        if isinstance(loss_value, torch.Tensor):
            total_loss += loss_value.sum()
        elif isinstance(loss_value, list):
            total_loss += sum(loss_item.sum() for loss_item in loss_value)

    total_loss.backward()
    assert imgs.grad is not None
    print("PAA forward/backward passed!")


if __name__ == "__main__":
    test_detr_forward_backward()
    test_yolof_forward_backward()
    test_paa_forward_backward()
