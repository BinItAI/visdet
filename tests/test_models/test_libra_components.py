# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from visdet.engine.structures import InstanceData
from visdet.models.losses.balanced_l1_loss import BalancedL1Loss
from visdet.models.necks.bfp import BFP
from visdet.models.task_modules.samplers import InstanceBalancedPosSampler, IoUBalancedNegSampler, SamplingResult


def test_bfp_forward():
    in_channels = 256
    num_levels = 4
    bfp = BFP(in_channels=in_channels, num_levels=num_levels, refine_level=2, refine_type="conv")

    inputs = [torch.randn(1, in_channels, 64 // (2**i), 64 // (2**i), requires_grad=True) for i in range(num_levels)]
    outputs = bfp(inputs)

    assert len(outputs) == num_levels
    for i in range(num_levels):
        assert outputs[i].shape == inputs[i].shape

    # Test backward
    loss = sum(o.sum() for o in outputs)
    loss.backward()
    for i in range(num_levels):
        assert inputs[i].grad is not None
    print("BFP forward/backward passed!")


def test_balanced_l1_loss():
    pred = torch.randn(10, 4, requires_grad=True)
    target = torch.randn(10, 4)
    loss_cfg = dict(alpha=0.5, gamma=1.5, beta=1.0, loss_weight=1.0)
    balanced_l1 = BalancedL1Loss(**loss_cfg)

    loss = balanced_l1(pred, target)
    assert loss.dim() == 0
    loss.backward()
    assert pred.grad is not None
    print("BalancedL1Loss forward/backward passed!")


def test_libra_samplers():
    # Mock data for samplers
    num_priors = 1000
    priors = torch.randn(num_priors, 4)
    gt_bboxes = torch.tensor([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=torch.float32)
    gt_labels = torch.tensor([1, 2], dtype=torch.long)

    class MockAssignResult:
        def __init__(self, gt_inds, max_overlaps, labels):
            self.gt_inds = gt_inds
            self.max_overlaps = max_overlaps
            self.labels = labels

        def add_gt_(self, labels):
            pass

    # Some positives, some negatives
    gt_inds = torch.zeros(num_priors, dtype=torch.long)
    gt_inds[:10] = 1
    gt_inds[10:20] = 2
    max_overlaps = torch.rand(num_priors)
    labels = torch.zeros(num_priors, dtype=torch.long)
    labels[:10] = 1
    labels[10:20] = 2

    assign_result = MockAssignResult(gt_inds, max_overlaps, labels)
    pred_instances = InstanceData(priors=priors)
    gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)

    # Test IoUBalancedNegSampler
    iou_sampler = IoUBalancedNegSampler(num=256, pos_fraction=0.25)
    res = iou_sampler.sample(assign_result, pred_instances, gt_instances)
    assert isinstance(res, SamplingResult)
    assert res.num_pos <= 64
    assert res.num_neg <= 256 - res.num_pos
    print("IoUBalancedNegSampler passed!")

    # Test InstanceBalancedPosSampler
    inst_sampler = InstanceBalancedPosSampler(num=256, pos_fraction=0.25)
    res = inst_sampler.sample(assign_result, pred_instances, gt_instances)
    assert isinstance(res, SamplingResult)
    assert res.num_pos <= 64
    print("InstanceBalancedPosSampler passed!")


if __name__ == "__main__":
    test_bfp_forward()
    test_balanced_l1_loss()
    test_libra_samplers()
