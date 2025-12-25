# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor

from visdet.models.utils.misc import multi_apply, unmap
from visdet.registry import MODELS

from .retina_head import RetinaHead


@MODELS.register_module()
class FSAFHead(RetinaHead):
    """Anchor-free head used in `FSAF <https://arxiv.org/abs/1903.00621>`_."""

    def __init__(
        self,
        *args,
        score_threshold: float | None = None,
        init_cfg: dict | list[dict] | None = None,
        **kwargs,
    ) -> None:
        if init_cfg is None:
            init_cfg = dict(
                type="Normal",
                layer="Conv2d",
                std=0.01,
                override=[
                    dict(type="Normal", name="retina_cls", std=0.01, bias_prob=0.01),
                    dict(type="Normal", name="retina_reg", std=0.01, bias=0.25),
                ],
            )
        super().__init__(*args, init_cfg=init_cfg, **kwargs)
        self.score_threshold = score_threshold
        self.relu = nn.ReLU(inplace=True)
        self.sampling = False

    def forward_single(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward feature map of a single scale level."""
        cls_score, bbox_pred = super().forward_single(x)
        # relu: TBLR encoder only accepts positive bbox_pred
        return cls_score, self.relu(bbox_pred)

    def _get_targets_single(
        self,
        flat_anchors: Tensor,
        valid_flags: Tensor,
        gt_bboxes: Tensor,
        gt_bboxes_ignore: Tensor | None,
        gt_labels: Tensor,
        img_meta: dict,
        label_channels: int = 1,
        unmap_outputs: bool = True,
    ) -> tuple:
        from visdet.models.task_modules.prior_generators import anchor_inside_flags

        inside_flags = anchor_inside_flags(
            flat_anchors,
            valid_flags,
            img_meta["img_shape"][:2],
            self.train_cfg.get("allowed_border", 0),
        )
        if not inside_flags.any():
            return (None,) * 8

        anchors = flat_anchors[inside_flags.type(torch.bool), :]

        from visdet.engine.structures import InstanceData

        pred_instances = InstanceData(priors=anchors)
        gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
        if gt_bboxes_ignore is not None:
            gt_instances_ignore = InstanceData(bboxes=gt_bboxes_ignore)
        else:
            gt_instances_ignore = None

        assign_result = self.assigner.assign(pred_instances, gt_instances, gt_instances_ignore)

        sampling_result = self.sampler.sample(assign_result, pred_instances, gt_instances)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros((num_valid_anchors,), dtype=torch.float)
        pos_gt_inds = anchors.new_full((num_valid_anchors,), -1, dtype=torch.long)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_priors, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            # The assigned gt_index for each anchor. (0-based)
            pos_gt_inds[pos_inds] = sampling_result.pos_assigned_gt_inds
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            label_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            pos_gt_inds = unmap(pos_gt_inds, num_total_anchors, inside_flags, fill=-1)

        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            pos_inds,
            neg_inds,
            sampling_result,
            pos_gt_inds,
        )

    def loss_by_feat(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        batch_gt_instances: list,
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: list | None = None,
    ) -> dict:
        for i in range(len(bbox_preds)):  # loop over fpn level
            # avoid 0 area of the predicted bbox
            bbox_preds[i] = bbox_preds[i].clamp(min=1e-4)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, batch_img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            label_channels=label_channels,
        )
        if cls_reg_targets is None:
            return None
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
            pos_assigned_gt_inds_list,
        ) = cls_reg_targets

        num_total_samples = num_total_pos + num_total_neg if self.sampling else num_total_pos
        num_total_samples = max(1.0, num_total_samples)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        from visdet.models.utils.misc import images_to_levels
        from visdet.structures.bbox import cat_boxes

        # Re-organize anchors
        all_anchor_list = []
        for i in range(len(batch_img_metas)):
            # anchor_list[i] is a list of Tensors
            all_anchor_list.append(torch.cat(tuple(anchor_list[i])))
        all_anchor_list_by_level = images_to_levels(all_anchor_list, num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            bbox_preds,
            all_anchor_list_by_level,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            avg_factor=num_total_samples,
        )

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def get_targets(
        self,
        anchor_list: list[list[Tensor]],
        valid_flag_list: list[list[Tensor]],
        batch_gt_instances: list,
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: list | None = None,
        label_channels: int = 1,
        unmap_outputs: bool = True,
    ) -> tuple:
        """Get targets for FSAF head."""
        num_imgs = len(batch_img_metas)
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        for i in range(num_imgs):
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None for _ in range(num_imgs)]

        gt_bboxes_list = [gt.bboxes for gt in batch_gt_instances]
        gt_labels_list = [gt.labels for gt in batch_gt_instances]

        (
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            pos_inds_list,
            neg_inds_list,
            sampling_results_list,
            pos_assigned_gt_inds_list,
        ) = multi_apply(
            self._get_targets_single,
            anchor_list,
            valid_flag_list,
            gt_bboxes_list,
            batch_gt_instances_ignore,
            gt_labels_list,
            batch_img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs,
        )

        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])

        from visdet.models.utils.misc import images_to_levels

        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
        pos_assigned_gt_inds_list = images_to_levels(pos_assigned_gt_inds_list, num_level_anchors)

        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
            pos_assigned_gt_inds_list,
        )
