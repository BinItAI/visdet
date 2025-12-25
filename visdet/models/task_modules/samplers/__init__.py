# ruff: noqa
from abc import ABCMeta, abstractmethod
import torch
import numpy as np
from visdet.registry import TASK_UTILS
from visdet.utils import util_mixins
from visdet.engine.structures import InstanceData


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

    @property
    def bboxes(self):
        """torch.Tensor: concatenated positive and negative priors"""
        return self.priors

    def __nice__(self):
        parts = []
        parts.append(f"num_gts={self.num_gts}")
        parts.append(f"num_pos={self.num_pos}")
        parts.append(f"num_neg={self.num_neg}")
        parts.append(f"avg_factor={self.avg_factor}")
        return ", ".join(parts)


class BaseSampler(metaclass=ABCMeta):
    """Base class of samplers."""

    def __init__(self, num, pos_fraction, neg_pos_ub=-1, add_gt_as_proposals=True, **kwargs):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_sampler = self
        self.neg_sampler = self
        self.context = kwargs.pop("context", None)

    @abstractmethod
    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Sample positive samples."""
        pass

    @abstractmethod
    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Sample negative samples."""
        pass

    def sample(self, assign_result, pred_instances: InstanceData, gt_instances: InstanceData, **kwargs):
        """Sample positive and negative bboxes."""
        bboxes = pred_instances.priors
        gt_bboxes = gt_instances.bboxes
        gt_labels = getattr(gt_instances, "labels", None)

        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]

        bboxes = bboxes[:, :4]

        gt_flags = bboxes.new_zeros((bboxes.shape[0],), dtype=torch.uint8)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError("gt_labels must be given when add_gt_as_proposals is True")
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        neg_inds = neg_inds.unique()

        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags)
        return sampling_result


@TASK_UTILS.register_module()
class PseudoSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        super().__init__(0, 0, **kwargs)

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        return torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1)

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        return torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1)

    def sample(self, assign_result, pred_instances: InstanceData, gt_instances: InstanceData, **kwargs):
        """Directly returns the positive and negative indices."""
        priors = pred_instances.priors
        gt_bboxes = gt_instances.bboxes

        pos_inds = self._sample_pos(assign_result, 0)
        neg_inds = self._sample_neg(assign_result, 0)

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
class RandomSampler(BaseSampler):
    """Random sampler."""

    def __init__(self, num, pos_fraction, neg_pos_ub=-1, add_gt_as_proposals=True, **kwargs):
        super().__init__(num, pos_fraction, neg_pos_ub, add_gt_as_proposals, **kwargs)

    def random_choice(self, gallery, num):
        """Random select some elements from the gallery."""
        assert len(gallery) >= num

        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            gallery = torch.tensor(gallery, dtype=torch.long, device=device)

        perm = torch.randperm(gallery.numel(), device=gallery.device)[:num]
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Randomly sample some negative samples."""
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)


@TASK_UTILS.register_module()
class IoUBalancedNegSampler(RandomSampler):
    """IoU Balanced Sampling."""

    def __init__(self, num, pos_fraction, floor_thr=-1, floor_fraction=0, num_bins=3, **kwargs):
        super().__init__(num, pos_fraction, **kwargs)
        self.floor_thr = floor_thr
        self.floor_fraction = floor_fraction
        self.num_bins = num_bins

    def sample_via_interval(self, max_overlaps, full_set, num_expected):
        max_iou = max_overlaps.max()
        iou_interval = (max_iou - self.floor_thr) / self.num_bins
        per_num_expected = int(num_expected / self.num_bins)

        sampled_inds = []
        for i in range(self.num_bins):
            start_iou = self.floor_thr + i * iou_interval
            end_iou = self.floor_thr + (i + 1) * iou_interval
            tmp_set = set(np.where(np.logical_and(max_overlaps >= start_iou, max_overlaps < end_iou))[0])
            tmp_inds = list(tmp_set & full_set)
            if len(tmp_inds) > per_num_expected:
                tmp_sampled_set = self.random_choice(tmp_inds, per_num_expected)
            else:
                tmp_sampled_set = np.array(tmp_inds, dtype=int)
            sampled_inds.append(tmp_sampled_set)

        sampled_inds = np.concatenate(sampled_inds)
        if len(sampled_inds) < num_expected:
            num_extra = num_expected - len(sampled_inds)
            extra_inds = np.array(list(full_set - set(sampled_inds)))
            if len(extra_inds) > num_extra:
                extra_inds = self.random_choice(extra_inds, num_extra)
            sampled_inds = np.concatenate([sampled_inds, extra_inds])

        return sampled_inds

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            max_overlaps = assign_result.max_overlaps.cpu().numpy()
            neg_set = set(neg_inds.cpu().numpy())

            if self.floor_thr > 0:
                floor_set = set(np.where(np.logical_and(max_overlaps >= 0, max_overlaps < self.floor_thr))[0])
                iou_sampling_set = set(np.where(max_overlaps >= self.floor_thr)[0])
            elif self.floor_thr == 0:
                floor_set = set(np.where(max_overlaps == 0)[0])
                iou_sampling_set = set(np.where(max_overlaps > self.floor_thr)[0])
            else:
                floor_set = set()
                iou_sampling_set = set(np.where(max_overlaps > self.floor_thr)[0])
                self.floor_thr = 0

            floor_neg_inds = list(floor_set & neg_set)
            iou_sampling_neg_inds = list(iou_sampling_set & neg_set)
            num_expected_iou_sampling = int(num_expected * (1 - self.floor_fraction))
            if len(iou_sampling_neg_inds) > num_expected_iou_sampling:
                if self.num_bins >= 2:
                    iou_sampled_inds = self.sample_via_interval(
                        max_overlaps,
                        set(iou_sampling_neg_inds),
                        num_expected_iou_sampling,
                    )
                else:
                    iou_sampled_inds = self.random_choice(iou_sampling_neg_inds, num_expected_iou_sampling)
            else:
                iou_sampled_inds = np.array(iou_sampling_neg_inds, dtype=int)
            num_expected_floor = num_expected - len(iou_sampled_inds)
            if len(floor_neg_inds) > num_expected_floor:
                sampled_floor_inds = self.random_choice(floor_neg_inds, num_expected_floor)
            else:
                sampled_floor_inds = np.array(floor_neg_inds, dtype=int)
            sampled_inds = np.concatenate((sampled_floor_inds, iou_sampled_inds))
            if len(sampled_inds) < num_expected:
                num_extra = num_expected - len(sampled_inds)
                extra_inds = np.array(list(neg_set - set(sampled_inds)))
                if len(extra_inds) > num_extra:
                    extra_inds = self.random_choice(extra_inds, num_extra)
                sampled_inds = np.concatenate((sampled_inds, extra_inds))
            sampled_inds = torch.from_numpy(sampled_inds).long().to(assign_result.gt_inds.device)
            return sampled_inds


@TASK_UTILS.register_module()
class InstanceBalancedPosSampler(RandomSampler):
    """Instance Balanced Positive Sampling."""

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            # Each instance should have the same probability to be sampled.
            # Thus, we first sample index of instance and then sample index of
            # positive bboxes of this instance.
            gt_inds = assign_result.gt_inds[pos_inds]
            unique_gt_inds = gt_inds.unique()
            num_gts = unique_gt_inds.numel()

            pos_inds_from_instance = []
            if num_expected >= num_gts:
                # at least one sample per instance
                avg_num_per_instance = int(num_expected / num_gts)
                for i in range(num_gts):
                    instance_pos_inds = pos_inds[gt_inds == unique_gt_inds[i]]
                    if instance_pos_inds.numel() > avg_num_per_instance:
                        sampled_instance_pos_inds = self.random_choice(instance_pos_inds, avg_num_per_instance)
                    else:
                        sampled_instance_pos_inds = instance_pos_inds
                    pos_inds_from_instance.append(sampled_instance_pos_inds)

                remaining_num = num_expected - torch.cat(pos_inds_from_instance).numel()
                if remaining_num > 0:
                    all_pos_inds = torch.cat(pos_inds_from_instance)
                    ignore_inds = set(all_pos_inds.cpu().numpy())
                    gallery_inds = list(set(pos_inds.cpu().numpy()) - ignore_inds)
                    if len(gallery_inds) > remaining_num:
                        extra_inds = self.random_choice(gallery_inds, remaining_num)
                    else:
                        extra_inds = np.array(gallery_inds, dtype=int)
                    pos_inds_from_instance.append(torch.from_numpy(extra_inds).to(pos_inds.device))
            else:
                # sample instances and then sample one bbox per instance
                sampled_instance_indices = self.random_choice(range(num_gts), num_expected)
                for i in sampled_instance_indices:
                    instance_pos_inds = pos_inds[gt_inds == unique_gt_inds[i]]
                    pos_inds_from_instance.append(self.random_choice(instance_pos_inds, 1))

            return torch.cat(pos_inds_from_instance)


__all__ = [
    "SamplingResult",
    "BaseSampler",
    "PseudoSampler",
    "RandomSampler",
    "IoUBalancedNegSampler",
    "InstanceBalancedPosSampler",
]
