# Copyright (c) OpenMMLab. All rights reserved.
import torch

from visdet.registry import MODELS
from visdet.structures import DetDataSample, InstanceData


def test_double_head_rcnn_forward_backward():
    model_cfg = dict(
        type="FasterRCNN",
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
        neck=dict(type="FPN", in_channels=[64, 128, 256, 512], out_channels=256, num_outs=5),
        rpn_head=dict(
            type="RPNHead",
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type="AnchorGenerator", scales=[8], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]
            ),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[1.0, 1.0, 1.0, 1.0]
            ),
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0),
        ),
        roi_head=dict(
            type="DoubleHeadRoIHead",
            reg_roi_scale_factor=1.3,
            bbox_roi_extractor=dict(
                type="SingleRoIExtractor",
                roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32],
            ),
            bbox_head=dict(
                type="DoubleConvFCBBoxHead",
                num_convs=2,
                num_fcs=2,
                in_channels=256,
                conv_out_channels=1024,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[0.1, 0.1, 0.2, 0.2]
                ),
                reg_class_agnostic=False,
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=2.0),
                loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=2.0),
            ),
        ),
        # model training and testing settings
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(type="RandomSampler", num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
            ),
            rpn_proposal=dict(nms_pre=2000, max_per_img=1000, nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(type="RandomSampler", num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False,
            ),
        ),
        test_cfg=dict(
            rpn=dict(nms_pre=1000, max_per_img=1000, nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0),
            roi=dict(score_thr=0.05, nms=dict(type="nms", iou_threshold=0.5), max_per_img=100),
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
    print("Double Head R-CNN forward/backward passed!")


if __name__ == "__main__":
    test_double_head_rcnn_forward_backward()
