import torch
from visdet.structures import DetDataSample
from visengine.config import Config


def _build_stage_train_cfg(pos_iou_thr: float) -> dict:
    return dict(
        assigner=dict(
            type="MaxIoUAssigner",
            pos_iou_thr=pos_iou_thr,
            neg_iou_thr=pos_iou_thr,
            min_pos_iou=pos_iou_thr,
            match_low_quality=False,
            ignore_iof_thr=-1,
        ),
        sampler=dict(
            type="RandomSampler",
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True,
        ),
        mask_size=28,
        pos_weight=-1,
        debug=False,
    )


def _build_roi_heads(num_classes: int = 80) -> dict:
    bbox_head_cfg = dict(
        type="Shared2FCBBoxHead",
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=num_classes,
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2],
        ),
        reg_class_agnostic=True,
        loss_cls=dict(type="CrossEntropyLoss"),
        loss_bbox=dict(type="SmoothL1Loss", beta=1.0),
    )
    mask_head_cfg = dict(
        type="FCNMaskHead",
        num_convs=4,
        in_channels=256,
        conv_out_channels=256,
        num_classes=num_classes,
        loss_mask=dict(type="CrossEntropyLoss", use_mask=True),
    )
    return dict(
        type="CascadeRoIHead",
        num_stages=3,
        stage_loss_weights=[1.0, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=[
            bbox_head_cfg,
            bbox_head_cfg
            | {
                "bbox_coder": dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1],
                )
            },
            bbox_head_cfg
            | {
                "bbox_coder": dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067],
                )
            },
        ],
        mask_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=14),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        mask_head=[mask_head_cfg, mask_head_cfg, mask_head_cfg],
    )


def _build_training_cfg() -> Config:
    return Config(
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
            rcnn=[
                _build_stage_train_cfg(0.5),
                _build_stage_train_cfg(0.6),
                _build_stage_train_cfg(0.7),
            ],
        )
    )


def _build_testing_cfg() -> Config:
    return Config(
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
    )


def _build_model_cfg(model_type: str = "CascadeRCNN") -> dict:
    return dict(
        type=model_type,
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
        neck=dict(type="FPN", in_channels=[96, 192, 384, 768], out_channels=256, num_outs=5),
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
            loss_bbox=dict(type="SmoothL1Loss", beta=1.0),
        ),
        roi_head=_build_roi_heads(),
        train_cfg=_build_training_cfg(),
        test_cfg=_build_testing_cfg(),
    )


def test_cascade_rcnn_init():
    from visengine.registry import DefaultScope

    with DefaultScope.overwrite_default_scope("visdet"):
        from visdet.registry import MODELS

        cfg = _build_model_cfg()
        model = MODELS.build(cfg)
        assert hasattr(model, "roi_head")
        assert model.roi_head.num_stages == 3
        assert len(model.roi_head.bbox_head) == 3
        assert len(model.roi_head.mask_head) == 3


def test_cascade_rcnn_forward():
    from visengine.registry import DefaultScope

    with DefaultScope.overwrite_default_scope("visdet"):
        from visdet.registry import MODELS

        cfg = _build_model_cfg()
        model = MODELS.build(cfg)
        model.eval()

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

        with torch.no_grad():
            outputs = model(batch_inputs, batch_data_samples, mode="predict")

        assert len(outputs) == 1
        assert hasattr(outputs[0], "pred_instances")
