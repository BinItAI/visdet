import torch
from visdet.structures import DetDataSample
from visengine.config import Config


def test_mask_rcnn_init():
    """Test MaskRCNN initialization."""
    # Set default scope to visdet to find our components
    from visengine.registry import DefaultScope

    with DefaultScope.overwrite_default_scope("visdet"):
        # Minimal configuration for MaskRCNN
        cfg = dict(
            type="MaskRCNN",
            data_preprocessor=dict(
                type="DetDataPreprocessor",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                bgr_to_rgb=True,
                pad_mask=True,
                pad_size_divisor=32,
            ),
            backbone=dict(
                type="SwinTransformer",
                embed_dims=96,
                depths=[2, 2, 6, 2],
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
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True),
                loss_bbox=dict(type="L1Loss"),
            ),
            roi_head=dict(
                type="StandardRoIHead",
                bbox_roi_extractor=dict(
                    type="SingleRoIExtractor",
                    roi_layer=dict(type="RoIAlign", output_size=7),
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
                    loss_cls=dict(type="CrossEntropyLoss"),
                    loss_bbox=dict(type="L1Loss"),
                ),
                mask_roi_extractor=dict(
                    type="SingleRoIExtractor",
                    roi_layer=dict(type="RoIAlign", output_size=14),
                    out_channels=256,
                    featmap_strides=[4, 8, 16, 32],
                ),
                mask_head=dict(
                    type="FCNMaskHead",
                    num_convs=4,
                    in_channels=256,
                    conv_out_channels=256,
                    num_classes=80,
                    loss_mask=dict(type="CrossEntropyLoss", use_mask=True),
                ),
            ),
            train_cfg=Config(
                dict(
                    rpn=dict(
                        assigner=dict(
                            type="MaxIoUAssigner",
                            pos_iou_thr=0.7,
                            neg_iou_thr=0.3,
                            min_pos_iou=0.3,
                        ),
                        sampler=dict(type="RandomSampler", num=256, pos_fraction=0.5),
                        allowed_border=-1,
                        pos_weight=-1,
                    ),
                    rpn_proposal=dict(
                        nms_pre=2000,
                        max_per_img=1000,
                        nms=dict(type="nms", iou_threshold=0.7),
                        min_bbox_size=0,
                    ),
                    rcnn=dict(
                        assigner=dict(
                            type="MaxIoUAssigner",
                            pos_iou_thr=0.5,
                            neg_iou_thr=0.5,
                            min_pos_iou=0.5,
                        ),
                        sampler=dict(type="RandomSampler", num=512, pos_fraction=0.25),
                        mask_size=28,
                        pos_weight=-1,
                    ),
                )
            ),
            test_cfg=Config(
                dict(
                    rpn=dict(
                        nms_pre=1000,
                        max_per_img=1000,
                        nms=dict(type="nms", iou_threshold=0.7),
                        min_bbox_size=0,
                    ),
                    rcnn=dict(
                        score_thr=0.05,
                        nms=dict(type="nms", iou_threshold=0.5),
                        max_per_img=100,
                        mask_thr_binary=0.5,
                    ),
                )
            ),
        )

        from visdet.registry import MODELS

        model = MODELS.build(cfg)

        # Check that all components are properly initialized
        assert hasattr(model, "backbone")
        assert hasattr(model, "neck")
        assert hasattr(model, "rpn_head")
        assert hasattr(model, "roi_head")
        assert hasattr(model, "data_preprocessor")


def test_mask_rcnn_forward():
    """Test MaskRCNN forward pass in test mode."""
    # Set default scope to visdet to find our components
    from visengine.registry import DefaultScope

    with DefaultScope.overwrite_default_scope("visdet"):
        # Create a simple model
        from visdet.registry import MODELS

        cfg = dict(
            type="MaskRCNN",
            data_preprocessor=dict(
                type="DetDataPreprocessor",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                bgr_to_rgb=True,
                pad_mask=True,
                pad_size_divisor=32,
            ),
            backbone=dict(
                type="SwinTransformer",
                embed_dims=96,
                depths=[2, 2, 6, 2],
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
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True),
                loss_bbox=dict(type="L1Loss"),
            ),
            roi_head=dict(
                type="StandardRoIHead",
                bbox_roi_extractor=dict(
                    type="SingleRoIExtractor",
                    roi_layer=dict(type="RoIAlign", output_size=7),
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
                    loss_cls=dict(type="CrossEntropyLoss"),
                    loss_bbox=dict(type="L1Loss"),
                ),
                mask_roi_extractor=dict(
                    type="SingleRoIExtractor",
                    roi_layer=dict(type="RoIAlign", output_size=14),
                    out_channels=256,
                    featmap_strides=[4, 8, 16, 32],
                ),
                mask_head=dict(
                    type="FCNMaskHead",
                    num_convs=4,
                    in_channels=256,
                    conv_out_channels=256,
                    num_classes=80,
                    loss_mask=dict(type="CrossEntropyLoss", use_mask=True),
                ),
            ),
            train_cfg=Config(
                dict(
                    rpn=dict(
                        assigner=dict(
                            type="MaxIoUAssigner",
                            pos_iou_thr=0.7,
                            neg_iou_thr=0.3,
                            min_pos_iou=0.3,
                        ),
                        sampler=dict(type="RandomSampler", num=256, pos_fraction=0.5),
                        allowed_border=-1,
                        pos_weight=-1,
                    ),
                    rpn_proposal=dict(
                        nms_pre=2000,
                        max_per_img=1000,
                        nms=dict(type="nms", iou_threshold=0.7),
                        min_bbox_size=0,
                    ),
                    rcnn=dict(
                        assigner=dict(
                            type="MaxIoUAssigner",
                            pos_iou_thr=0.5,
                            neg_iou_thr=0.5,
                            min_pos_iou=0.5,
                        ),
                        sampler=dict(type="RandomSampler", num=512, pos_fraction=0.25),
                        mask_size=28,
                        pos_weight=-1,
                    ),
                )
            ),
            test_cfg=Config(
                dict(
                    rpn=dict(
                        nms_pre=1000,
                        max_per_img=1000,
                        nms=dict(type="nms", iou_threshold=0.7),
                        min_bbox_size=0,
                    ),
                    rcnn=dict(
                        score_thr=0.05,
                        nms=dict(type="nms", iou_threshold=0.5),
                        max_per_img=100,
                        mask_thr_binary=0.5,
                    ),
                )
            ),
        )

        model = MODELS.build(cfg)
        model.eval()

        # Create dummy input
        batch_inputs = torch.randn(1, 3, 224, 224)
        batch_data_samples = [DetDataSample()]
        batch_data_samples[0].set_metainfo(
            {
                "img_shape": (224, 224),
                "ori_shape": (224, 224),
                "pad_shape": (224, 224),
                "batch_input_shape": (224, 224),
                "scale_factor": (1.0, 1.0),
            }
        )

        # Test forward
        with torch.no_grad():
            outputs = model(batch_inputs, batch_data_samples, mode="predict")

        assert len(outputs) == 1
        assert hasattr(outputs[0], "pred_instances")
