# Copyright (c) OpenMMLab. All rights reserved.
import torch

from visdet.core import build_assigner, build_sampler
from visdet.structures import InstanceData


def _dummy_bbox_sampling(proposal_list, gt_bboxes, gt_labels):
    """Create sample results that can be passed to BBoxHead.get_targets."""
    num_imgs = 1
    feat = torch.rand(1, 1, 3, 3)
    assign_config = dict(
        type="MaxIoUAssigner",
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.5,
        ignore_iof_thr=-1,
    )
    sampler_config = dict(
        type="RandomSampler",
        num=512,
        pos_fraction=0.25,
        neg_pos_ub=-1,
        add_gt_as_proposals=True,
    )
    bbox_assigner = build_assigner(assign_config)
    bbox_sampler = build_sampler(sampler_config)
    gt_bboxes_ignore = [None for _ in range(num_imgs)]
    sampling_results = []
    for i in range(num_imgs):
        # Use InstanceData wrapper for new API
        pred_instances = InstanceData(priors=proposal_list[i])
        gt_instances = InstanceData(bboxes=gt_bboxes[i], labels=gt_labels[i])
        gt_instances_ignore = (
            InstanceData(bboxes=gt_bboxes_ignore[i])
            if gt_bboxes_ignore[i] is not None
            else InstanceData(bboxes=torch.empty(0, 4))
        )
        assign_result = bbox_assigner.assign(pred_instances, gt_instances, gt_instances_ignore=gt_instances_ignore)
        sampling_result = bbox_sampler.sample(assign_result, pred_instances, gt_instances, feats=feat)
        sampling_results.append(sampling_result)

    return sampling_results
