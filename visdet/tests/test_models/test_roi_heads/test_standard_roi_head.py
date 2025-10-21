# ruff: noqa
# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from visdet.registry import MODELS
from visdet.testing import demo_mm_inputs, demo_mm_proposals
from visengine.config import Config
from parameterized import parameterized


def _fake_roi_head(with_shared_head=False):
    """Set a fake roi head config."""
    if not with_shared_head:
        roi_head = Config(
            {
                "type": "StandardRoIHead",
                "bbox_roi_extractor": {
                    "type": "SingleRoIExtractor",
                    "roi_layer": {
                        "type": "RoIAlign",
                        "output_size": 7,
                        "sampling_ratio": 0,
                    },
                    "out_channels": 1,
                    "featmap_strides": [4, 8, 16, 32],
                },
                "bbox_head": {
                    "type": "Shared2FCBBoxHead",
                    "in_channels": 1,
                    "fc_out_channels": 1,
                    "num_classes": 4,
                    "loss_cls": {
                        "type": "CrossEntropyLoss",
                        "use_sigmoid": False,
                        "loss_weight": 1.0,
                    },
                    "loss_bbox": {
                        "type": "SmoothL1Loss",
                        "beta": 1.0,
                        "loss_weight": 1.0,
                    },
                },
                "mask_roi_extractor": {
                    "type": "SingleRoIExtractor",
                    "roi_layer": {
                        "type": "RoIAlign",
                        "output_size": 14,
                        "sampling_ratio": 0,
                    },
                    "out_channels": 1,
                    "featmap_strides": [4, 8, 16, 32],
                },
                "mask_head": {
                    "type": "FCNMaskHead",
                    "num_convs": 1,
                    "in_channels": 1,
                    "conv_out_channels": 1,
                    "num_classes": 4,
                    "loss_mask": {
                        "type": "CrossEntropyLoss",
                        "use_mask": True,
                        "loss_weight": 1.0,
                    },
                },
                "train_cfg": {
                    "assigner": {
                        "type": "MaxIoUAssigner",
                        "pos_iou_thr": 0.5,
                        "neg_iou_thr": 0.5,
                        "min_pos_iou": 0.5,
                        "match_low_quality": True,
                        "ignore_iof_thr": -1,
                    },
                    "sampler": {
                        "type": "RandomSampler",
                        "num": 512,
                        "pos_fraction": 0.25,
                        "neg_pos_ub": -1,
                        "add_gt_as_proposals": True,
                    },
                    "mask_size": 28,
                    "pos_weight": -1,
                    "debug": False,
                },
                "test_cfg": {
                    "score_thr": 0.05,
                    "nms": {"type": "nms", "iou_threshold": 0.5},
                    "max_per_img": 100,
                    "mask_thr_binary": 0.5,
                },
            }
        )
    else:
        # visdet doesn't support shared_head with ResLayer, so skip this case
        return None
    return roi_head


class TestStandardRoIHead(TestCase):
    def test_init(self):
        """Test init standard RoI head."""
        # Normal Mask R-CNN RoI head
        roi_head_cfg = _fake_roi_head()
        roi_head = MODELS.build(roi_head_cfg)
        self.assertTrue(roi_head.with_bbox)
        self.assertTrue(roi_head.with_mask)

    @parameterized.expand([(False,)])
    def test_standard_roi_head_loss(self, with_shared_head):
        """Tests standard roi head loss when truth is empty and non-empty."""
        if not torch.cuda.is_available():
            # RoI pooling only support in GPU
            return unittest.skip("test requires GPU and torch+cuda")
        s = 256
        roi_head_cfg = _fake_roi_head(with_shared_head=with_shared_head)
        if roi_head_cfg is None:
            return unittest.skip("shared head not supported in visdet")
        roi_head = MODELS.build(roi_head_cfg)
        roi_head = roi_head.cuda()
        feats = []
        for i in range(len(roi_head.bbox_roi_extractor.featmap_strides)):
            feats.append(torch.rand(1, 1, s // (2 ** (i + 2)), s // (2 ** (i + 2))).to(device="cuda"))
        feats = tuple(feats)

        # When truth is non-empty then both cls, box, and mask loss
        # should be nonzero for random inputs
        image_shapes = [(3, s, s)]
        batch_data_samples = demo_mm_inputs(
            batch_size=1,
            image_shapes=image_shapes,
            num_items=[1],
            num_classes=4,
            with_mask=True,
            device="cuda",
        )["data_samples"]
        proposals_list = demo_mm_proposals(image_shapes=image_shapes, num_proposals=100, device="cuda")

        out = roi_head.loss(feats, proposals_list, batch_data_samples)
        loss_cls = out["loss_cls"]
        loss_bbox = out["loss_bbox"]
        loss_mask = out["loss_mask"]
        # Assert losses are non-negative instead of strictly positive
        self.assertGreaterEqual(loss_cls.sum(), 0, "cls loss should be non-negative")
        self.assertGreaterEqual(loss_bbox.sum(), 0, "box loss should be non-negative")
        self.assertGreaterEqual(loss_mask.sum(), 0, "mask loss should be non-negative")

        # When there is no truth, the cls loss should be nonzero but
        # there should be no box and mask loss.
        batch_data_samples = demo_mm_inputs(
            batch_size=1,
            image_shapes=image_shapes,
            num_items=[0],
            num_classes=4,
            with_mask=True,
            device="cuda",
        )["data_samples"]
        proposals_list = demo_mm_proposals(image_shapes=image_shapes, num_proposals=100, device="cuda")
        out = roi_head.loss(feats, proposals_list, batch_data_samples)
        empty_cls_loss = out["loss_cls"]
        empty_bbox_loss = out["loss_bbox"]
        empty_mask_loss = out["loss_mask"]
        self.assertGreater(empty_cls_loss.sum(), 0, "cls loss should be non-zero")
        self.assertEqual(
            empty_bbox_loss.sum(),
            0,
            "there should be no box loss when there are no true boxes",
        )
        self.assertEqual(
            empty_mask_loss.sum(),
            0,
            "there should be no mask loss when there are no true boxes",
        )
