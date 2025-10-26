import os

import pytest

from visdet.apis import init_detector


def test_init_detector():
    project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    project_dir = os.path.join(project_dir, "..")

    # Create a minimal config for testing (without data pipeline)
    from visdet.engine.config import Config

    config = Config(
        dict(
            model=dict(
                type="MaskRCNN",
                backbone=dict(
                    type="SwinTransformer",
                    embed_dims=96,
                    depths=[2, 2, 2, 2],
                    num_heads=[3, 6, 12, 24],
                    window_size=7,
                    out_indices=(0, 1, 2, 3),
                ),
                neck=dict(
                    type="FPN",
                    in_channels=[96, 192, 384, 768],
                    out_channels=256,
                    num_outs=5,
                ),
                rpn_head=dict(
                    type="RPNHead",
                    in_channels=256,
                    feat_channels=256,
                    anchor_generator=dict(
                        type="AnchorGenerator",
                        scales=[8],
                        ratios=[0.5, 1.0, 2.0],
                        strides=[4, 8, 16, 32, 64],
                    ),
                    bbox_coder=dict(
                        type="DeltaXYWHBBoxCoder",
                        target_means=[0.0, 0.0, 0.0, 0.0],
                        target_stds=[1.0, 1.0, 1.0, 1.0],
                    ),
                    loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
                    loss_bbox=dict(type="L1Loss", loss_weight=1.0),
                ),
                roi_head=dict(
                    type="StandardRoIHead",
                    bbox_roi_extractor=dict(
                        type="SingleRoIExtractor",
                        roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
                        out_channels=256,
                        featmap_strides=[4, 8, 16, 32],
                    ),
                    bbox_head=dict(
                        type="Shared2FCBBoxHead",
                        in_channels=256,
                        fc_out_channels=1024,
                        roi_feat_size=7,
                        num_classes=80,
                        bbox_coder=dict(
                            type="DeltaXYWHBBoxCoder",
                            target_means=[0.0, 0.0, 0.0, 0.0],
                            target_stds=[0.1, 0.1, 0.2, 0.2],
                        ),
                        reg_class_agnostic=False,
                        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
                        loss_bbox=dict(type="L1Loss", loss_weight=1.0),
                    ),
                    mask_roi_extractor=dict(
                        type="SingleRoIExtractor",
                        roi_layer=dict(type="RoIAlign", output_size=14, sampling_ratio=0),
                        out_channels=256,
                        featmap_strides=[4, 8, 16, 32],
                    ),
                    mask_head=dict(
                        type="FCNMaskHead",
                        num_convs=4,
                        in_channels=256,
                        conv_out_channels=256,
                        num_classes=80,
                        loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
                    ),
                ),
                train_cfg=dict(
                    rpn=dict(
                        assigner=dict(
                            type="MaxIoUAssigner",
                            pos_iou_thr=0.7,
                            neg_iou_thr=0.3,
                            min_pos_iou=0.3,
                            match_low_quality=True,
                        ),
                        sampler=dict(
                            type="RandomSampler",
                            num=256,
                            pos_fraction=0.5,
                        ),
                    ),
                    rpn_proposal=dict(
                        nms_pre=2000,
                        max_per_img=1000,
                        nms=dict(type="nms", iou_threshold=0.7),
                    ),
                    rcnn=dict(
                        assigner=dict(
                            type="MaxIoUAssigner",
                            pos_iou_thr=0.5,
                            neg_iou_thr=0.5,
                        ),
                        sampler=dict(
                            type="RandomSampler",
                            num=512,
                            pos_fraction=0.25,
                        ),
                    ),
                ),
                test_cfg=dict(
                    rpn=dict(
                        nms_pre=1000,
                        max_per_img=1000,
                        nms=dict(type="nms", iou_threshold=0.7),
                    ),
                    rcnn=dict(
                        score_thr=0.05,
                        nms=dict(type="nms", iou_threshold=0.5),
                        max_per_img=100,
                    ),
                ),
            ),
            default_scope="visdet",
        )
    )

    # test init_detector with config object and cfg_options
    cfg_options = dict(
        model=dict(
            backbone=dict(
                depths=[2, 2, 2, 2],  # Reduce depth for testing
            )
        )
    )
    model = init_detector(config, device="cpu", cfg_options=cfg_options, palette="coco")
    assert model is not None

    # test init_detector with Config object
    model = init_detector(config, device="cpu", palette="coco")
    assert model is not None

    # test init_detector with undesirable type
    with pytest.raises(TypeError):
        model = init_detector([config])  # noqa: F841
