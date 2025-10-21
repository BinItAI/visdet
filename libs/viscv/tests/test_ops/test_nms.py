# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
from viscv.ops import batched_nms, nms


class TestNMS:
    def test_nms_cpu(self):
        """Test NMS on CPU."""
        np_boxes = np.array(
            [
                [6.0, 3.0, 8.0, 7.0],
                [3.0, 6.0, 9.0, 11.0],
                [3.0, 7.0, 10.0, 12.0],
                [1.0, 4.0, 13.0, 7.0],
            ],
            dtype=np.float32,
        )
        np_scores = np.array([0.6, 0.9, 0.7, 0.2], dtype=np.float32)

        boxes = torch.from_numpy(np_boxes)
        scores = torch.from_numpy(np_scores)

        dets, inds = nms(boxes, scores, iou_threshold=0.3)

        # Check that highest scoring box is kept
        assert inds[0] == 1  # index of box with score 0.9

        # Check shape
        assert dets.shape[1] == 5  # x1, y1, x2, y2, score
        assert len(inds) <= len(boxes)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_nms_cuda(self):
        """Test NMS on CUDA."""
        np_boxes = np.array(
            [
                [6.0, 3.0, 8.0, 7.0],
                [3.0, 6.0, 9.0, 11.0],
                [3.0, 7.0, 10.0, 12.0],
                [1.0, 4.0, 13.0, 7.0],
            ],
            dtype=np.float32,
        )
        np_scores = np.array([0.6, 0.9, 0.7, 0.2], dtype=np.float32)

        boxes = torch.from_numpy(np_boxes).cuda()
        scores = torch.from_numpy(np_scores).cuda()

        dets, inds = nms(boxes, scores, iou_threshold=0.3)

        # Check that results are on cuda
        assert dets.is_cuda
        assert inds.is_cuda

        # Check that highest scoring box is kept
        assert inds[0].cpu() == 1  # index of box with score 0.9

    def test_batched_nms(self):
        """Test batched NMS."""
        boxes = torch.tensor(
            [
                [6.0, 3.0, 8.0, 7.0],
                [3.0, 6.0, 9.0, 11.0],
                [3.0, 7.0, 10.0, 12.0],
                [1.0, 4.0, 13.0, 7.0],
            ]
        )
        scores = torch.tensor([0.6, 0.9, 0.7, 0.2])
        idxs = torch.tensor([0, 0, 1, 1])  # Two different classes/batches

        # Test with class-based NMS
        dets, keep = batched_nms(boxes, scores, idxs, nms_cfg=dict(iou_threshold=0.3))

        # Should keep the top scoring box from each class
        assert len(keep) >= 2  # At least one from each class
        assert 1 in keep  # Highest score in class 0
        assert 2 in keep  # Highest score in class 1

    def test_batched_nms_split_thr(self):
        """Test batched NMS with split threshold."""
        # Create many boxes to trigger split
        n_boxes = 10000
        boxes = torch.rand(n_boxes, 4)
        boxes[:, 2:] = boxes[:, :2] + 0.1  # Ensure x2 > x1, y2 > y1
        scores = torch.rand(n_boxes)
        idxs = torch.zeros(n_boxes, dtype=torch.long)

        # Test with split threshold
        dets, keep = batched_nms(
            boxes,
            scores,
            idxs,
            nms_cfg=dict(type="nms", iou_threshold=0.5, split_thr=1000),
        )

        # Check output shapes
        assert dets.shape[1] == 5
        assert len(keep) > 0
        assert len(keep) <= n_boxes
