# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from visdet.registry import MODELS

from .standard_roi_head import StandardRoIHead

EPS = 1e-15


@MODELS.register_module()
class DynamicRoIHead(StandardRoIHead):
    """RoI head for `Dynamic R-CNN <https://arxiv.org/abs/2004.06002>`_."""

    def __init__(self, **kwargs) -> None:
        super(DynamicRoIHead, self).__init__(**kwargs)
        # the IoU history of the past `update_iter_interval` iterations
        self.iou_history = []
        # the beta history of the past `update_iter_interval` iterations
        self.beta_history = []

    def loss(self, x, rpn_results_list, batch_data_samples):
        """Calculate losses from a batch of inputs and data samples."""
        num_imgs = len(batch_data_samples)
        batch_gt_instances = [data_samples.gt_instances for data_samples in batch_data_samples]
        batch_gt_instances_ignore = [
            getattr(data_samples, "ignored_instances", None) for data_samples in batch_data_samples
        ]

        sampling_results = []
        cur_iou = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            if "bboxes" in rpn_results:
                rpn_results.priors = rpn_results.pop("bboxes")

            assign_result = self.bbox_assigner.assign(rpn_results, batch_gt_instances[i], batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(assign_result, rpn_results, batch_gt_instances[i])
            sampling_results.append(sampling_result)

            # record the `iou_topk`-th largest IoU in an image
            iou_topk = min(
                self.train_cfg.dynamic_rcnn.iou_topk,
                len(assign_result.max_overlaps),
            )
            if iou_topk > 0:
                ious, _ = torch.topk(assign_result.max_overlaps, iou_topk)
                cur_iou.append(ious[-1].item())
            else:
                cur_iou.append(self.train_cfg.dynamic_rcnn.initial_iou)

        # average the current IoUs over images
        cur_iou = np.mean(cur_iou)
        self.iou_history.append(cur_iou)

        # bbox head forward and loss
        bbox_results = self.bbox_loss(x, sampling_results)

        # update IoU threshold and SmoothL1 beta
        update_iter_interval = self.train_cfg.dynamic_rcnn.update_iter_interval
        if len(self.iou_history) % update_iter_interval == 0:
            self.update_hyperparameters()

        return bbox_results

    def bbox_loss(self, x, sampling_results):
        from visdet.models.task_modules.assigners import get_box_tensor
        from visdet.structures.bbox import bbox2roi

        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, self.train_cfg)

        # record the `beta_topk`-th smallest target
        # bbox_targets[2] is label_weights, bbox_targets[1] is labels, bbox_targets[2] is bbox_targets...
        # Wait, check Shared2FCBBoxHead.get_targets return value
        # In visdet: (labels, label_weights, bbox_targets, bbox_weights)
        labels, label_weights, bbox_targets_vals, bbox_weights_vals = bbox_targets

        pos_inds = bbox_weights_vals[:, 0].nonzero().squeeze(1)
        num_pos = len(pos_inds)
        if num_pos > 0:
            cur_target = bbox_targets_vals[pos_inds, :2].abs().mean(dim=1)
            beta_topk = min(self.train_cfg.dynamic_rcnn.beta_topk * len(sampling_results), num_pos)
            cur_target = torch.kthvalue(cur_target, beta_topk)[0].item()
            self.beta_history.append(cur_target)

        loss_bbox = self.bbox_head.loss(bbox_results["cls_score"], bbox_results["bbox_pred"], rois, *bbox_targets)
        return loss_bbox

    def update_hyperparameters(self):
        """Update hyperparameters."""
        new_iou_thr = max(self.train_cfg.dynamic_rcnn.initial_iou, np.mean(self.iou_history))
        self.iou_history = []
        self.bbox_assigner.pos_iou_thr = new_iou_thr
        self.bbox_assigner.neg_iou_thr = new_iou_thr
        self.bbox_assigner.min_pos_iou = new_iou_thr

        if len(self.beta_history) > 0:
            if np.median(self.beta_history) < EPS:
                new_beta = self.bbox_head.loss_bbox.beta
            else:
                new_beta = min(self.train_cfg.dynamic_rcnn.initial_beta, np.median(self.beta_history))
            self.beta_history = []
            if hasattr(self.bbox_head.loss_bbox, "beta"):
                self.bbox_head.loss_bbox.beta = new_beta
