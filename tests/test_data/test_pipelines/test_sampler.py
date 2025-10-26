# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from visdet.core.bbox.assigners import MaxIoUAssigner
from visdet.core.bbox.samplers import OHEMSampler, RandomSampler, ScoreHLRSampler
from visdet.structures import InstanceData


def test_random_sampler():
    assigner = MaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        ignore_iof_thr=0.5,
        ignore_wrt_candidates=False,
    )
    bboxes = torch.FloatTensor(
        [
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [5, 5, 15, 15],
            [32, 32, 38, 42],
        ]
    )
    gt_bboxes = torch.FloatTensor(
        [
            [0, 0, 10, 9],
            [0, 10, 10, 19],
        ]
    )
    gt_labels = torch.LongTensor([1, 2])
    gt_bboxes_ignore = torch.Tensor(
        [
            [30, 30, 40, 40],
        ]
    )

    # Wrap in InstanceData for new API
    pred_instances = InstanceData(priors=bboxes)
    gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
    gt_instances_ignore = InstanceData(bboxes=gt_bboxes_ignore)

    assign_result = assigner.assign(pred_instances, gt_instances, gt_instances_ignore=gt_instances_ignore)

    sampler = RandomSampler(num=10, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=True)

    sample_result = sampler.sample(assign_result, pred_instances, gt_instances)

    assert len(sample_result.pos_bboxes) == len(sample_result.pos_inds)
    assert len(sample_result.neg_bboxes) == len(sample_result.neg_inds)


def test_random_sampler_empty_gt():
    assigner = MaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        ignore_iof_thr=0.5,
        ignore_wrt_candidates=False,
    )
    bboxes = torch.FloatTensor(
        [
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [5, 5, 15, 15],
            [32, 32, 38, 42],
        ]
    )
    gt_bboxes = torch.empty(0, 4)
    gt_labels = torch.empty(
        0,
    ).long()

    # Wrap in InstanceData for new API
    pred_instances = InstanceData(priors=bboxes)
    gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)

    assign_result = assigner.assign(pred_instances, gt_instances)

    sampler = RandomSampler(num=10, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=True)

    sample_result = sampler.sample(assign_result, pred_instances, gt_instances)

    assert len(sample_result.pos_bboxes) == len(sample_result.pos_inds)
    assert len(sample_result.neg_bboxes) == len(sample_result.neg_inds)


def test_random_sampler_empty_pred():
    assigner = MaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        ignore_iof_thr=0.5,
        ignore_wrt_candidates=False,
    )
    bboxes = torch.empty(0, 4)
    gt_bboxes = torch.FloatTensor(
        [
            [0, 0, 10, 9],
            [0, 10, 10, 19],
        ]
    )
    gt_labels = torch.LongTensor([1, 2])

    # Wrap in InstanceData for new API
    pred_instances = InstanceData(priors=bboxes)
    gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)

    assign_result = assigner.assign(pred_instances, gt_instances)

    sampler = RandomSampler(num=10, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=True)

    sample_result = sampler.sample(assign_result, pred_instances, gt_instances)

    assert len(sample_result.pos_bboxes) == len(sample_result.pos_inds)
    assert len(sample_result.neg_bboxes) == len(sample_result.neg_inds)


def _context_for_ohem():
    import sys
    from os.path import dirname

    sys.path.insert(0, dirname(dirname(dirname(__file__))))
    from test_models.test_forward import _get_detector_cfg

    model = _get_detector_cfg("faster_rcnn/faster_rcnn_r50_fpn_ohem_1x_coco.py")
    model["pretrained"] = None

    from visdet.models import build_detector

    context = build_detector(model).roi_head
    return context


@pytest.mark.skip(
    reason="Requires FasterRCNN configs which are out of scope for minimal visdet (Swin + Mask R-CNN only)"
)
def test_ohem_sampler():
    assigner = MaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        ignore_iof_thr=0.5,
        ignore_wrt_candidates=False,
    )
    bboxes = torch.FloatTensor(
        [
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [5, 5, 15, 15],
            [32, 32, 38, 42],
        ]
    )
    gt_bboxes = torch.FloatTensor(
        [
            [0, 0, 10, 9],
            [0, 10, 10, 19],
        ]
    )
    gt_labels = torch.LongTensor([1, 2])
    gt_bboxes_ignore = torch.Tensor(
        [
            [30, 30, 40, 40],
        ]
    )

    # Wrap in InstanceData for new API
    pred_instances = InstanceData(priors=bboxes)
    gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
    gt_instances_ignore = InstanceData(bboxes=gt_bboxes_ignore)

    assign_result = assigner.assign(pred_instances, gt_instances, gt_instances_ignore=gt_instances_ignore)

    context = _context_for_ohem()

    sampler = OHEMSampler(
        num=10,
        pos_fraction=0.5,
        context=context,
        neg_pos_ub=-1,
        add_gt_as_proposals=True,
    )

    feats = [torch.rand(1, 256, int(2**i), int(2**i)) for i in [6, 5, 4, 3, 2]]
    sample_result = sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels, feats=feats)

    assert len(sample_result.pos_bboxes) == len(sample_result.pos_inds)
    assert len(sample_result.neg_bboxes) == len(sample_result.neg_inds)


@pytest.mark.skip(
    reason="Requires FasterRCNN configs which are out of scope for minimal visdet (Swin + Mask R-CNN only)"
)
def test_ohem_sampler_empty_gt():
    assigner = MaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        ignore_iof_thr=0.5,
        ignore_wrt_candidates=False,
    )
    bboxes = torch.FloatTensor(
        [
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [5, 5, 15, 15],
            [32, 32, 38, 42],
        ]
    )
    gt_bboxes = torch.empty(0, 4)
    gt_labels = torch.LongTensor([])
    gt_bboxes_ignore = torch.Tensor([])

    # Wrap in InstanceData for new API
    pred_instances = InstanceData(priors=bboxes)
    gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
    gt_instances_ignore = InstanceData(bboxes=gt_bboxes_ignore)

    assign_result = assigner.assign(pred_instances, gt_instances, gt_instances_ignore=gt_instances_ignore)

    context = _context_for_ohem()

    sampler = OHEMSampler(
        num=10,
        pos_fraction=0.5,
        context=context,
        neg_pos_ub=-1,
        add_gt_as_proposals=True,
    )

    feats = [torch.rand(1, 256, int(2**i), int(2**i)) for i in [6, 5, 4, 3, 2]]

    sample_result = sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels, feats=feats)

    assert len(sample_result.pos_bboxes) == len(sample_result.pos_inds)
    assert len(sample_result.neg_bboxes) == len(sample_result.neg_inds)


@pytest.mark.skip(
    reason="Requires FasterRCNN configs which are out of scope for minimal visdet (Swin + Mask R-CNN only)"
)
def test_ohem_sampler_empty_pred():
    assigner = MaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        ignore_iof_thr=0.5,
        ignore_wrt_candidates=False,
    )
    bboxes = torch.empty(0, 4)
    gt_bboxes = torch.FloatTensor(
        [
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [5, 5, 15, 15],
            [32, 32, 38, 42],
        ]
    )
    gt_labels = torch.LongTensor([1, 2, 2, 3])
    gt_bboxes_ignore = torch.Tensor([])

    # Wrap in InstanceData for new API
    pred_instances = InstanceData(priors=bboxes)
    gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
    gt_instances_ignore = InstanceData(bboxes=gt_bboxes_ignore)

    assign_result = assigner.assign(pred_instances, gt_instances, gt_instances_ignore=gt_instances_ignore)

    context = _context_for_ohem()

    sampler = OHEMSampler(
        num=10,
        pos_fraction=0.5,
        context=context,
        neg_pos_ub=-1,
        add_gt_as_proposals=True,
    )

    feats = [torch.rand(1, 256, int(2**i), int(2**i)) for i in [6, 5, 4, 3, 2]]

    sample_result = sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels, feats=feats)

    assert len(sample_result.pos_bboxes) == len(sample_result.pos_inds)
    assert len(sample_result.neg_bboxes) == len(sample_result.neg_inds)


@pytest.mark.skip(reason="SamplingResult.random() method does not exist in this version")
def test_random_sample_result():
    from visdet.core.bbox.samplers.sampling_result import SamplingResult

    SamplingResult.random(num_gts=0, num_preds=0)
    SamplingResult.random(num_gts=0, num_preds=3)
    SamplingResult.random(num_gts=3, num_preds=3)
    SamplingResult.random(num_gts=0, num_preds=3)
    SamplingResult.random(num_gts=7, num_preds=7)
    SamplingResult.random(num_gts=7, num_preds=64)
    SamplingResult.random(num_gts=24, num_preds=3)

    for i in range(3):
        SamplingResult.random(rng=i)


@pytest.mark.skip(
    reason="Requires FasterRCNN configs which are out of scope for minimal visdet (Swin + Mask R-CNN only)"
)
def test_score_hlr_sampler_empty_pred():
    assigner = MaxIoUAssigner(
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        ignore_iof_thr=0.5,
        ignore_wrt_candidates=False,
    )
    context = _context_for_ohem()
    sampler = ScoreHLRSampler(
        num=10,
        pos_fraction=0.5,
        context=context,
        neg_pos_ub=-1,
        add_gt_as_proposals=True,
    )
    gt_bboxes_ignore = torch.Tensor([])
    feats = [torch.rand(1, 256, int(2**i), int(2**i)) for i in [6, 5, 4, 3, 2]]

    # empty bbox
    bboxes = torch.empty(0, 4)
    gt_bboxes = torch.FloatTensor(
        [
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [5, 5, 15, 15],
            [32, 32, 38, 42],
        ]
    )
    gt_labels = torch.LongTensor([1, 2, 2, 3])

    # Wrap in InstanceData for new API
    pred_instances = InstanceData(priors=bboxes)
    gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
    gt_instances_ignore = InstanceData(bboxes=gt_bboxes_ignore)

    assign_result = assigner.assign(pred_instances, gt_instances, gt_instances_ignore=gt_instances_ignore)
    sample_result, _ = sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels, feats=feats)
    assert len(sample_result.neg_inds) == 0
    assert len(sample_result.pos_bboxes) == len(sample_result.pos_inds)
    assert len(sample_result.neg_bboxes) == len(sample_result.neg_inds)

    # empty gt
    bboxes = torch.FloatTensor(
        [
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [5, 5, 15, 15],
            [32, 32, 38, 42],
        ]
    )
    gt_bboxes = torch.empty(0, 4)
    gt_labels = torch.LongTensor([])

    # Wrap in InstanceData for new API
    pred_instances = InstanceData(priors=bboxes)
    gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
    gt_instances_ignore = InstanceData(bboxes=gt_bboxes_ignore)

    assign_result = assigner.assign(pred_instances, gt_instances, gt_instances_ignore=gt_instances_ignore)
    sample_result, _ = sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels, feats=feats)
    assert len(sample_result.pos_inds) == 0
    assert len(sample_result.pos_bboxes) == len(sample_result.pos_inds)
    assert len(sample_result.neg_bboxes) == len(sample_result.neg_inds)

    # non-empty input
    bboxes = torch.FloatTensor(
        [
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [5, 5, 15, 15],
            [32, 32, 38, 42],
        ]
    )
    gt_bboxes = torch.FloatTensor(
        [
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [5, 5, 15, 15],
            [32, 32, 38, 42],
        ]
    )
    gt_labels = torch.LongTensor([1, 2, 2, 3])

    # Wrap in InstanceData for new API
    pred_instances = InstanceData(priors=bboxes)
    gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
    gt_instances_ignore = InstanceData(bboxes=gt_bboxes_ignore)

    assign_result = assigner.assign(pred_instances, gt_instances, gt_instances_ignore=gt_instances_ignore)
    sample_result, _ = sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels, feats=feats)
    assert len(sample_result.pos_bboxes) == len(sample_result.pos_inds)
    assert len(sample_result.neg_bboxes) == len(sample_result.neg_inds)
