# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor

from visdet.cv.cnn import ConvModule, Scale
from visdet.models.task_modules.samplers import PseudoSampler
from visdet.models.utils import images_to_levels, multi_apply, unmap
from visdet.utils import reduce_mean
from visdet.registry import MODELS, TASK_UTILS
from .anchor_head import AnchorHead


@MODELS.register_module()
class ATSSHead(AnchorHead):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection via
    Adaptive Training Sample Selection.

    https://arxiv.org/abs/1912.02424
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        pred_kernel_size: int = 3,
        stacked_convs: int = 4,
        conv_cfg: dict | None = None,
        norm_cfg: dict = dict(type="GN", num_groups=32, requires_grad=True),
        reg_decoded_bbox: bool = True,
        loss_centerness: dict = dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        init_cfg: dict | list[dict] = dict(
            type="Normal",
            layer="Conv2d",
            std=0.01,
            override=dict(type="Normal", name="atss_cls", std=0.01, bias_prob=0.01),
        ),
        **kwargs,
    ) -> None:
        self.pred_kernel_size = pred_kernel_size
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super().__init__(
            num_classes,
            in_channels,
            reg_decoded_bbox=reg_decoded_bbox,
            init_cfg=init_cfg,
            **kwargs,
        )

        self.sampling = False
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg["assigner"])
            self.sampler = PseudoSampler(context=self)
        self.loss_centerness = MODELS.build(loss_centerness)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                )
            )
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                )
            )
        pred_pad_size = self.pred_kernel_size // 2
        self.atss_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            self.pred_kernel_size,
            padding=pred_pad_size,
        )
        self.atss_reg = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 4,
            self.pred_kernel_size,
            padding=pred_pad_size,
        )
        self.atss_centerness = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * 1,
            self.pred_kernel_size,
            padding=pred_pad_size,
        )
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.prior_generator.strides])

    def forward(self, feats: tuple[Tensor]) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """Forward features from the upstream network."""
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x: Tensor, scale: Scale) -> tuple[Tensor, Tensor, Tensor]:
        """Forward feature of a single scale level."""
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.atss_cls(cls_feat)
        bbox_pred = scale(self.atss_reg(reg_feat)).float()
        centerness = self.atss_centerness(reg_feat)
        return cls_score, bbox_pred, centerness

    def loss_by_feat_single(
        self,
        anchors: Tensor,
        cls_score: Tensor,
        bbox_pred: Tensor,
        centerness: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        num_total_samples: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute loss of a single scale level."""
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # classification loss
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]

            centerness_targets = self.centerness_target(pos_anchors, pos_bbox_targets)
            pos_decode_bbox_pred = self.bbox_coder.decode(pos_anchors, pos_bbox_pred)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_bbox_targets,
                weight=centerness_targets,
                avg_factor=1.0,
            )

            # centerness loss
            loss_centerness = self.loss_centerness(pos_centerness, centerness_targets, avg_factor=num_total_samples)

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.0)

        return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum()

    def loss_by_feat(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        centernesses: list[Tensor],
        batch_gt_instances: list,
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: list | None = None,
    ) -> dict:
        """Calculate the loss based on the features extracted by the detection head."""
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
        )
        (
            anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets

        num_total_samples = reduce_mean(torch.tensor(num_total_pos, dtype=torch.float, device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, loss_centerness, bbox_avg_factor = multi_apply(
            self.loss_by_feat_single,
            anchor_list,
            cls_scores,
            bbox_preds,
            centernesses,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            num_total_samples=num_total_samples,
        )

        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(torch.tensor(bbox_avg_factor, device=device)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_centerness=loss_centerness)

    def centerness_target(self, anchors: Tensor, gts: Tensor) -> Tensor:
        """Compute centerness targets."""
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = torch.stack([l_, r_], dim=1)
        top_bottom = torch.stack([t_, b_], dim=1)
        centerness = torch.sqrt(
            (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]).clamp(min=0)
            * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]).clamp(min=0)
        )
        return centerness

    def get_targets(
        self,
        anchor_list: list[list[Tensor]],
        valid_flag_list: list[list[Tensor]],
        batch_gt_instances: list,
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: list | None = None,
    ) -> tuple:
        """Get targets for ATSS head."""
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
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            anchor_list,
            valid_flag_list,
            num_level_anchors_list,
            gt_bboxes_list,
            batch_gt_instances_ignore,
            gt_labels_list,
            batch_img_metas,
        )

        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])

        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
        return (
            anchors_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def _get_target_single(
        self,
        flat_anchors: Tensor,
        valid_flags: Tensor,
        num_level_anchors: list[int],
        gt_bboxes: Tensor,
        gt_bboxes_ignore: Tensor | None,
        gt_labels: Tensor,
        img_meta: dict,
    ) -> tuple:
        """Compute regression, classification targets for anchors in a single image."""
        from visdet.models.task_modules.prior_generators import anchor_inside_flags

        inside_flags = anchor_inside_flags(
            flat_anchors,
            valid_flags,
            img_meta["img_shape"][:2],
            self.train_cfg.get("allowed_border", 0),
        )
        if not inside_flags.any():
            return (None,) * 7

        anchors = flat_anchors[inside_flags, :]

        # Get number of anchors per level that are inside the image
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [int(flags.sum()) for flags in split_inside_flags]

        assign_result = self.assigner.assign(anchors, num_level_anchors_inside, gt_bboxes, gt_bboxes_ignore, gt_labels)

        from visdet.engine.structures import InstanceData

        pred_instances = InstanceData(priors=anchors)
        gt_instances = InstanceData(bboxes=gt_bboxes)
        sampling_result = self.sampler.sample(assign_result, pred_instances, gt_instances)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if self.reg_decoded_bbox:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            else:
                pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_priors, sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            label_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        num_total_anchors = flat_anchors.size(0)
        anchors_out = unmap(anchors, num_total_anchors, inside_flags)
        labels_out = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)
        label_weights_out = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets_out = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights_out = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (
            anchors_out,
            labels_out,
            label_weights_out,
            bbox_targets_out,
            bbox_weights_out,
            pos_inds,
            neg_inds,
        )
