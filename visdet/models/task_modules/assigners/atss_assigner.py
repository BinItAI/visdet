# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor

from visdet.models.task_modules.assigners.assign_result import AssignResult
from visdet.models.task_modules.assigners.base_assigner import BaseAssigner
from visdet.registry import TASK_UTILS


@TASK_UTILS.register_module()
class ATSSAssigner(BaseAssigner):
    """Adaptive Training Sample Selection (ATSS) Assigner.

    ATSS automatically selects positive and negative samples based on
    statistical characteristics of the object. It selects the top-k anchors
    per level and uses IoU mean + std as the threshold.

    Paper: https://arxiv.org/abs/1912.02424

    Args:
        topk (int): Number of top candidates per level. Defaults to 9.
        iou_calculator (dict): Config for IoU calculator. Defaults to
            BboxOverlaps2D.
    """

    def __init__(
        self,
        topk: int = 9,
        iou_calculator: dict | None = None,
    ) -> None:
        self.topk = topk
        if iou_calculator is None:
            from visdet.models.task_modules.assigners.iou2d_calculator import BboxOverlaps2D

            self.iou_calculator = BboxOverlaps2D()
        else:
            self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def assign(
        self,
        pred_instances,
        gt_instances,
        gt_instances_ignore=None,
        **kwargs,
    ) -> AssignResult:
        """Assign gt to anchors.

        Args:
            pred_instances: Predictions containing 'priors' and 'num_level_priors'.
            gt_instances: Ground truth containing 'bboxes' and 'labels'.
            gt_instances_ignore: Ignored ground truth (unused).

        Returns:
            AssignResult: Assignment result.
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        priors = pred_instances.priors
        num_level_priors = pred_instances.num_level_priors

        num_gt = gt_bboxes.size(0)
        num_priors = priors.size(0)

        # Initialize assignment
        assigned_gt_inds = priors.new_full((num_priors,), 0, dtype=torch.long)
        assigned_labels = priors.new_full((num_priors,), -1, dtype=torch.long)

        if num_gt == 0 or num_priors == 0:
            if num_gt == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gt,
                gt_inds=assigned_gt_inds,
                max_overlaps=priors.new_zeros(num_priors),
                labels=assigned_labels,
            )

        # Compute IoU between gt and priors
        overlaps = self.iou_calculator(gt_bboxes, priors)

        # Compute center distance between gt and priors
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = torch.stack([gt_cx, gt_cy], dim=1)

        # Get prior centers (first 2 dims of priors are x, y)
        prior_cx = (priors[:, 0] + priors[:, 2]) / 2.0 if priors.size(-1) == 4 else priors[:, 0]
        prior_cy = (priors[:, 1] + priors[:, 3]) / 2.0 if priors.size(-1) == 4 else priors[:, 1]
        prior_points = torch.stack([prior_cx, prior_cy], dim=1)

        # Distance between gt and priors: (num_gt, num_priors)
        distances = (prior_points[None, :, :] - gt_points[:, None, :]).pow(2).sum(dim=-1).sqrt()

        # Select top-k candidates per level
        candidate_idxs = []
        start_idx = 0
        for level, num_priors_level in enumerate(num_level_priors):
            end_idx = start_idx + num_priors_level
            distances_per_level = distances[:, start_idx:end_idx]
            selectable_k = min(self.topk, num_priors_level)

            _, topk_idxs_per_level = distances_per_level.topk(selectable_k, dim=1, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx

        # Concatenate candidates from all levels: (num_gt, topk * num_levels)
        candidate_idxs = torch.cat(candidate_idxs, dim=1)

        # Get overlaps of candidates
        candidate_overlaps = overlaps.gather(1, candidate_idxs)

        # Compute adaptive threshold per gt using mean + std
        overlaps_mean = candidate_overlaps.mean(dim=1)
        overlaps_std = candidate_overlaps.std(dim=1)
        overlaps_thr = overlaps_mean + overlaps_std

        # Select positive samples
        is_pos = candidate_overlaps >= overlaps_thr[:, None]

        # Filter out candidates whose center is outside the gt box
        for gt_idx in range(num_gt):
            candidate_idxs_for_gt = candidate_idxs[gt_idx]
            ep_prior_cx = prior_cx[candidate_idxs_for_gt]
            ep_prior_cy = prior_cy[candidate_idxs_for_gt]

            l_ = ep_prior_cx - gt_bboxes[gt_idx, 0]
            t_ = ep_prior_cy - gt_bboxes[gt_idx, 1]
            r_ = gt_bboxes[gt_idx, 2] - ep_prior_cx
            b_ = gt_bboxes[gt_idx, 3] - ep_prior_cy

            is_in_gt = (l_ > 0) & (t_ > 0) & (r_ > 0) & (b_ > 0)
            is_pos[gt_idx] = is_pos[gt_idx] & is_in_gt

        # Get all positive indices
        for gt_idx in range(num_gt):
            candidate_idxs_for_gt = candidate_idxs[gt_idx][is_pos[gt_idx]]
            if len(candidate_idxs_for_gt) > 0:
                ious_for_candidates = overlaps[gt_idx, candidate_idxs_for_gt]
                # If a prior is assigned to multiple gts, select the one with max IoU
                for cand_idx, iou in zip(candidate_idxs_for_gt, ious_for_candidates):
                    if assigned_gt_inds[cand_idx] == 0:
                        assigned_gt_inds[cand_idx] = gt_idx + 1
                    else:
                        # Already assigned, check if this gt has higher IoU
                        prev_gt_idx = assigned_gt_inds[cand_idx] - 1
                        if iou > overlaps[prev_gt_idx, cand_idx]:
                            assigned_gt_inds[cand_idx] = gt_idx + 1

        # Assign labels
        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1)
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]

        # Get max overlaps
        max_overlaps = overlaps.max(dim=0)[0]

        return AssignResult(
            num_gts=num_gt,
            gt_inds=assigned_gt_inds,
            max_overlaps=max_overlaps,
            labels=assigned_labels,
        )
