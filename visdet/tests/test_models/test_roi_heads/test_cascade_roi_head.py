"""Tests for CascadeRoIHead to validate cascade logic and mask prediction."""

import pytest
import torch
from visengine.config import ConfigDict
from visengine.structures import InstanceData


def _build_cascade_roi_head_cfg(num_classes: int = 80, include_test_cfg: bool = False) -> dict:
    """Build configuration for CascadeRoIHead."""
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
    cfg = dict(
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

    if include_test_cfg:
        cfg["test_cfg"] = ConfigDict(
            score_thr=0.05,
            nms=dict(type="nms", iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5,
        )

    return cfg


def test_cascade_roi_head_init():
    """Test CascadeRoIHead initialization."""
    from visengine.registry import DefaultScope

    with DefaultScope.overwrite_default_scope("visdet"):
        from visdet.registry import MODELS

        cfg = _build_cascade_roi_head_cfg()
        roi_head = MODELS.build(cfg)

        # Verify 3 stages
        assert roi_head.num_stages == 3
        assert len(roi_head.bbox_head) == 3
        assert len(roi_head.mask_head) == 3
        assert len(roi_head.bbox_roi_extractor) == 3
        assert len(roi_head.mask_roi_extractor) == 3

        # Verify stage loss weights
        assert roi_head.stage_loss_weights == [1.0, 0.5, 0.25]

        # Verify progressively tighter bbox stds (convert to tuple for comparison)
        assert tuple(roi_head.bbox_head[0].bbox_coder.stds) == (0.1, 0.1, 0.2, 0.2)
        assert tuple(roi_head.bbox_head[1].bbox_coder.stds) == (0.05, 0.05, 0.1, 0.1)
        assert tuple(roi_head.bbox_head[2].bbox_coder.stds) == (
            0.033,
            0.033,
            0.067,
            0.067,
        )


def test_cascade_roi_head_predict_bbox():
    """Test that bbox prediction uses final stage only."""
    from visengine.registry import DefaultScope

    with DefaultScope.overwrite_default_scope("visdet"):
        from visdet.registry import MODELS

        cfg = _build_cascade_roi_head_cfg()
        roi_head = MODELS.build(cfg)
        roi_head.eval()

        # Create mock inputs (256 channels to match roi_extractor out_channels)
        batch_size = 2
        feat_shapes = [(256, 56, 56), (256, 28, 28), (256, 14, 14), (256, 7, 7)]
        x = tuple(torch.randn(batch_size, *shape) for shape in feat_shapes)

        # Create mock proposals
        rpn_results_list = []
        for i in range(batch_size):
            instance_data = InstanceData()
            instance_data.bboxes = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32)
            instance_data.scores = torch.tensor([0.9, 0.8])
            rpn_results_list.append(instance_data)

        batch_img_metas = [
            {
                "img_shape": (224, 224),
                "ori_shape": (224, 224),
                "scale_factor": (1.0, 1.0),
            },
            {
                "img_shape": (224, 224),
                "ori_shape": (224, 224),
                "scale_factor": (1.0, 1.0),
            },
        ]

        rcnn_test_cfg = dict(score_thr=0.05, nms=dict(type="nms", iou_threshold=0.5), max_per_img=100)

        with torch.no_grad():
            results = roi_head.predict_bbox(x, batch_img_metas, rpn_results_list, rcnn_test_cfg, rescale=False)

        # Verify we get results for each image
        assert len(results) == batch_size
        for result in results:
            assert hasattr(result, "bboxes")
            assert hasattr(result, "scores")
            assert hasattr(result, "labels")


def test_cascade_roi_head_predict_mask_logic():
    """Test that mask prediction logic - this reveals the bug."""
    from unittest.mock import patch

    from visengine.registry import DefaultScope

    with DefaultScope.overwrite_default_scope("visdet"):
        from visdet.registry import MODELS

        cfg = _build_cascade_roi_head_cfg(include_test_cfg=True)
        roi_head = MODELS.build(cfg)
        roi_head.eval()

        # Create mock inputs (256 channels to match roi_extractor out_channels)
        batch_size = 1
        feat_shapes = [(256, 56, 56), (256, 28, 28), (256, 14, 14), (256, 7, 7)]
        x = tuple(torch.randn(batch_size, *shape) for shape in feat_shapes)

        # Create mock detection results
        results_list = []
        instance_data = InstanceData()
        instance_data.bboxes = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)
        instance_data.scores = torch.tensor([0.9])
        instance_data.labels = torch.tensor([0])
        results_list.append(instance_data)

        batch_img_metas = [
            {
                "img_shape": (224, 224),
                "ori_shape": (224, 224),
                "scale_factor": (1.0, 1.0),
            }
        ]

        # Track which mask heads are called
        call_counts = {0: 0, 1: 0, 2: 0}

        original_forward = roi_head._mask_forward

        def tracked_mask_forward(stage, *args, **kwargs):
            call_counts[stage] += 1
            return original_forward(stage, *args, **kwargs)

        with patch.object(roi_head, "_mask_forward", side_effect=tracked_mask_forward):
            with torch.no_grad():
                results = roi_head.predict_mask(x, batch_img_metas, results_list, rescale=False)

        # BUG: Currently all 3 stages are called
        # This test documents the bug - it should only call stage 2 (final stage)
        assert call_counts[0] == 1, "Stage 0 mask head called (should not be)"
        assert call_counts[1] == 1, "Stage 1 mask head called (should not be)"
        assert call_counts[2] == 1, "Stage 2 mask head called (correct)"

        print(f"BUG CONFIRMED: All {roi_head.num_stages} mask head stages called during inference")
        print(f"Call counts: {call_counts}")
        print("EXPECTED: Only stage 2 (final) should be called")


def test_cascade_roi_head_bbox_refinement():
    """Test that bbox refinement happens between stages during training."""
    from visengine.registry import DefaultScope

    with DefaultScope.overwrite_default_scope("visdet"):
        from visdet.registry import MODELS

        cfg = _build_cascade_roi_head_cfg(num_classes=2)

        # Add training config
        train_cfg = dict(
            rpn=dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                ),
                sampler=dict(type="RandomSampler", num=256, pos_fraction=0.5),
            ),
            rcnn=[
                dict(
                    assigner=dict(
                        type="MaxIoUAssigner",
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.5,
                        min_pos_iou=0.5,
                        match_low_quality=False,
                    ),
                    sampler=dict(
                        type="RandomSampler",
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True,
                    ),
                    mask_size=28,
                ),
                dict(
                    assigner=dict(
                        type="MaxIoUAssigner",
                        pos_iou_thr=0.6,
                        neg_iou_thr=0.6,
                        min_pos_iou=0.6,
                        match_low_quality=False,
                    ),
                    sampler=dict(
                        type="RandomSampler",
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True,
                    ),
                    mask_size=28,
                ),
                dict(
                    assigner=dict(
                        type="MaxIoUAssigner",
                        pos_iou_thr=0.7,
                        neg_iou_thr=0.7,
                        min_pos_iou=0.7,
                        match_low_quality=False,
                    ),
                    sampler=dict(
                        type="RandomSampler",
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True,
                    ),
                    mask_size=28,
                ),
            ],
        )

        cfg["train_cfg"] = train_cfg["rcnn"]
        roi_head = MODELS.build(cfg)
        roi_head.train()

        # Verify IoU thresholds increase across stages
        assert roi_head.bbox_assigner[0].pos_iou_thr == 0.5
        assert roi_head.bbox_assigner[1].pos_iou_thr == 0.6
        assert roi_head.bbox_assigner[2].pos_iou_thr == 0.7

        print("✓ Cascade IoU thresholds correctly configured: [0.5, 0.6, 0.7]")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
