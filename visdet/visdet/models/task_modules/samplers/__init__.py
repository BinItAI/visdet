# ruff: noqa
import torch
from visdet.registry import TASK_UTILS
from visdet.utils import util_mixins
from visengine.structures import InstanceData


class SamplingResult(util_mixins.NiceRepr):
    """Bbox sampling result."""

    def __init__(
        self,
        pos_inds: torch.Tensor,
        neg_inds: torch.Tensor,
        priors: torch.Tensor,
        gt_bboxes: torch.Tensor,
        assign_result,
        gt_flags: torch.Tensor,
        avg_factor_with_neg: bool = True,
    ) -> None:
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.num_pos = max(pos_inds.numel(), 1)
        self.num_neg = max(neg_inds.numel(), 1)
        self.avg_factor_with_neg = avg_factor_with_neg
        self.avg_factor = self.num_pos + self.num_neg if avg_factor_with_neg else self.num_pos
        self.pos_priors = priors[pos_inds]
        self.neg_priors = priors[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        self.pos_gt_labels = assign_result.labels[pos_inds]

        box_dim = 4
        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_bboxes = gt_bboxes.view(-1, box_dim)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, box_dim)
            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds.long()]

    @property
    def priors(self):
        """torch.Tensor: concatenated positive and negative priors"""
        return torch.cat([self.pos_priors, self.neg_priors])

    def __nice__(self):
        parts = []
        parts.append(f"num_gts={self.num_gts}")
        parts.append(f"num_pos={self.num_pos}")
        parts.append(f"num_neg={self.num_neg}")
        parts.append(f"avg_factor={self.avg_factor}")
        return ", ".join(parts)


@TASK_UTILS.register_module()
class PseudoSampler:
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        self.context = kwargs.get("context", None)

    def sample(self, assign_result, pred_instances: InstanceData, gt_instances: InstanceData, **kwargs):
        """Directly returns the positive and negative indices."""
        priors = pred_instances.priors
        gt_bboxes = gt_instances.bboxes

        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1)
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1)

        gt_flags = priors.new_zeros(priors.shape[0], dtype=torch.uint8)

        return SamplingResult(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            priors=priors,
            gt_bboxes=gt_bboxes,
            assign_result=assign_result,
            gt_flags=gt_flags,
            avg_factor_with_neg=True,
        )


@TASK_UTILS.register_module()
class RandomSampler:
    """Random sampler."""

    def __init__(self, num, pos_fraction, neg_pos_ub=-1, **kwargs):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.context = kwargs.get("context", None)

    def sample(self, assign_result, pred_instances: InstanceData, gt_instances: InstanceData, **kwargs):
        """Sample positive and negative bboxes."""
        priors = pred_instances.priors
        gt_bboxes = gt_instances.bboxes

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1)
        if pos_inds.numel() > num_expected_pos:
            perm = torch.randperm(pos_inds.numel())[:num_expected_pos]
            pos_inds = pos_inds[perm]

        num_expected_neg = self.num - pos_inds.numel()
        if self.neg_pos_ub > 0:
            num_expected_neg = min(num_expected_neg, pos_inds.numel() * self.neg_pos_ub)
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1)
        if neg_inds.numel() > num_expected_neg:
            perm = torch.randperm(neg_inds.numel())[:num_expected_neg]
            neg_inds = neg_inds[perm]

        gt_flags = priors.new_zeros(priors.shape[0], dtype=torch.uint8)

        return SamplingResult(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            priors=priors,
            gt_bboxes=gt_bboxes,
            assign_result=assign_result,
            gt_flags=gt_flags,
            avg_factor_with_neg=True,
        )


__all__ = ["PseudoSampler", "RandomSampler"]
