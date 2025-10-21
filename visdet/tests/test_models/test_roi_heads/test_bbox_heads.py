"""Test cases for bbox heads."""

import pytest
import torch
from visdet.models.task_modules.samplers import SamplingResult
from visengine.config import Config


class TestShared2FCBBoxHead:
    """Test Shared2FCBBoxHead implementation."""

    @pytest.fixture
    def bbox_head_cfg(self):
        """Bbox head config."""
        return {
            "type": "Shared2FCBBoxHead",
            "in_channels": 256,
            "fc_out_channels": 1024,
            "roi_feat_size": 7,
            "num_classes": 80,
            "bbox_coder": {
                "type": "DeltaXYWHBBoxCoder",
                "target_means": [0.0, 0.0, 0.0, 0.0],
                "target_stds": [0.1, 0.1, 0.2, 0.2],
            },
            "reg_class_agnostic": False,
            "loss_cls": {
                "type": "CrossEntropyLoss",
                "use_sigmoid": False,
                "loss_weight": 1.0,
            },
            "loss_bbox": {"type": "L1Loss", "loss_weight": 1.0},
        }

    @pytest.fixture
    def roi_feats(self):
        """Create dummy RoI features."""
        # Features for 5 RoIs
        return torch.randn(5, 256, 7, 7)

    @pytest.fixture
    def sampling_results(self):
        """Create dummy sampling results."""
        from visdet.models.task_modules.assigners.assign_result import AssignResult

        results = []
        for _i in range(2):  # 2 images
            # Create assign result first
            gt_inds = torch.tensor([1, 2, 0], dtype=torch.long)  # pos, pos, neg
            max_overlaps = torch.tensor([0.9, 0.8, 0.1], dtype=torch.float)
            labels = torch.tensor([5, 10, -1], dtype=torch.long)
            assign_result = AssignResult(num_gts=2, gt_inds=gt_inds, max_overlaps=max_overlaps, labels=labels)

            # Create sampling result
            pos_inds = torch.tensor([0, 1], dtype=torch.long)
            neg_inds = torch.tensor([2], dtype=torch.long)
            priors = torch.tensor(
                [
                    [10.0, 10.0, 50.0, 50.0],
                    [100.0, 100.0, 200.0, 200.0],
                    [150.0, 150.0, 250.0, 250.0],
                ],
                dtype=torch.float32,
            )
            gt_bboxes = torch.tensor(
                [
                    [12.0, 12.0, 48.0, 48.0],
                    [102.0, 102.0, 198.0, 198.0],
                ],
                dtype=torch.float32,
            )
            gt_flags = torch.zeros(3, dtype=torch.uint8)

            result = SamplingResult(
                pos_inds=pos_inds,
                neg_inds=neg_inds,
                priors=priors,
                gt_bboxes=gt_bboxes,
                assign_result=assign_result,
                gt_flags=gt_flags,
                avg_factor_with_neg=True,
            )
            results.append(result)
        return results

    def test_init(self, bbox_head_cfg):
        """Test bbox head initialization."""
        from visdet.registry import MODELS

        bbox_head = MODELS.build(bbox_head_cfg)

        assert bbox_head.num_classes == 80
        assert bbox_head.in_channels == 256
        assert bbox_head.fc_out_channels == 1024
        assert hasattr(bbox_head, "bbox_coder")
        assert hasattr(bbox_head, "loss_cls")
        assert hasattr(bbox_head, "loss_bbox")
        assert hasattr(bbox_head, "shared_fcs")
        assert len(bbox_head.shared_fcs) == 2
        assert hasattr(bbox_head, "fc_cls")
        assert hasattr(bbox_head, "fc_reg")

    def test_forward(self, bbox_head_cfg, roi_feats):
        """Test forward pass."""
        from visdet.registry import MODELS

        bbox_head = MODELS.build(bbox_head_cfg)

        cls_score, bbox_pred = bbox_head(roi_feats)

        # Check output shapes
        num_rois = roi_feats.shape[0]
        assert cls_score.shape == (num_rois, 81)  # 80 classes + bg
        assert bbox_pred.shape == (
            num_rois,
            80 * 4,
        )  # class-specific regression (no bg)

    def test_loss(self, bbox_head_cfg):
        """Test loss calculation."""
        from visdet.registry import MODELS

        bbox_head = MODELS.build(bbox_head_cfg)

        # Create dummy inputs
        batch_size = 8
        cls_score = torch.randn(batch_size, 81)
        bbox_pred = torch.randn(batch_size, 80 * 4)  # No background for regression
        rois = torch.randn(batch_size, 5)  # [batch_idx, x1, y1, x2, y2]
        labels = torch.randint(0, 81, (batch_size,))
        label_weights = torch.ones(batch_size)
        bbox_targets = torch.randn(batch_size, 4)
        bbox_weights = torch.ones(batch_size, 4)

        losses = bbox_head.loss(
            cls_score,
            bbox_pred,
            rois,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
        )

        assert "loss_cls" in losses
        assert "loss_bbox" in losses
        assert losses["loss_cls"].sum() >= 0
        assert losses["loss_bbox"].sum() >= 0

    def test_loss_and_target(self, bbox_head_cfg, sampling_results):
        """Test loss_and_target method."""
        from visdet.registry import MODELS

        bbox_head = MODELS.build(bbox_head_cfg)

        # Create RoIs to match sampling results (3 per image, 2 images = 6 total)
        roi_feats = torch.randn(6, 256, 7, 7)

        # Forward pass
        cls_score, bbox_pred = bbox_head(roi_feats)

        # Create dummy RoIs
        rois = torch.tensor(
            [
                [0, 10, 10, 50, 50],  # img0, roi0
                [0, 100, 100, 200, 200],  # img0, roi1
                [0, 150, 150, 250, 250],  # img0, roi2
                [1, 20, 20, 60, 60],  # img1, roi0
                [1, 120, 120, 220, 220],  # img1, roi1
                [1, 150, 150, 250, 250],  # img1, roi2
            ],
            dtype=torch.float32,
        )

        rcnn_train_cfg = Config({"pos_weight": -1})

        result = bbox_head.loss_and_target(cls_score, bbox_pred, rois, sampling_results, rcnn_train_cfg)

        assert "loss_bbox" in result
        assert "loss_cls" in result["loss_bbox"]
        assert "loss_bbox" in result["loss_bbox"]
        assert result["loss_bbox"]["loss_cls"].sum() >= 0
        assert result["loss_bbox"]["loss_bbox"].sum() >= 0

    def test_get_targets(self, bbox_head_cfg, sampling_results):
        """Test get_targets method."""
        from visdet.registry import MODELS

        bbox_head = MODELS.build(bbox_head_cfg)

        rcnn_train_cfg = Config({"pos_weight": -1})

        labels, label_weights, bbox_targets, bbox_weights = bbox_head.get_targets(
            sampling_results, rcnn_train_cfg, concat=True
        )

        # Total samples = 2 images * 3 proposals each = 6
        assert labels.shape == (6,)
        assert label_weights.shape == (6,)
        assert bbox_targets.shape == (6, 4)
        assert bbox_weights.shape == (6, 4)

        # Check that positive samples have correct labels
        # First image: 2 pos, 1 neg
        assert labels[0] == 5  # First positive sample
        assert labels[1] == 10  # Second positive sample
        assert labels[2] == 80  # Negative sample (background)
        # Second image: similar pattern
        assert labels[3] == 5
        assert labels[4] == 10
        assert labels[5] == 80

    def test_predict_by_feat(self, bbox_head_cfg):
        """Test predict_by_feat method."""
        from visdet.registry import MODELS

        bbox_head = MODELS.build(bbox_head_cfg)

        # Create dummy inputs for 2 images
        rois = (
            torch.tensor([[0, 10, 10, 50, 50], [0, 20, 20, 80, 80]], dtype=torch.float32),
            torch.tensor([[1, 100, 100, 200, 200]], dtype=torch.float32),
        )
        cls_scores = (
            torch.randn(2, 81),
            torch.randn(1, 81),
        )
        bbox_preds = (
            torch.randn(2, 80 * 4),  # No background for regression
            torch.randn(1, 80 * 4),  # No background for regression
        )
        batch_img_metas = [
            {"img_shape": (640, 640, 3), "scale_factor": (1.0, 1.0)},
            {"img_shape": (640, 640, 3), "scale_factor": (1.0, 1.0)},
        ]
        rcnn_test_cfg = Config(
            {
                "score_thr": 0.05,
                "nms": {"type": "nms", "iou_threshold": 0.5},
                "max_per_img": 100,
            }
        )

        results = bbox_head.predict_by_feat(rois, cls_scores, bbox_preds, batch_img_metas, rcnn_test_cfg, rescale=False)

        assert len(results) == 2
        for result in results:
            assert hasattr(result, "bboxes")
            assert hasattr(result, "scores")
            assert hasattr(result, "labels")

    def test_reg_class_agnostic(self, bbox_head_cfg):
        """Test class-agnostic regression."""
        bbox_head_cfg["reg_class_agnostic"] = True

        from visdet.registry import MODELS

        bbox_head = MODELS.build(bbox_head_cfg)

        roi_feats = torch.randn(5, 256, 7, 7)
        cls_score, bbox_pred = bbox_head(roi_feats)

        # With class-agnostic regression, output should be 4 values per RoI
        assert bbox_pred.shape == (5, 4)
