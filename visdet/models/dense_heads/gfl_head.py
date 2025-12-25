# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from visdet.cv.cnn import ConvModule, Scale
from visdet.models.task_modules.samplers import PseudoSampler
from visdet.models.utils import (
    filter_scores_and_topk,
    images_to_levels,
    multi_apply,
    unmap,
)
from visdet.structures.bbox import bbox_overlaps
from visdet.utils import reduce_mean
from visdet.registry import MODELS, TASK_UTILS
from .anchor_head import AnchorHead


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution."""

    def __init__(self, reg_max: int = 16) -> None:
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer("project", torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x: Tensor) -> Tensor:
        """Forward feature from the regression head to get integral result of
        bounding box location.
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x


@MODELS.register_module()
class GFLHead(AnchorHead):
    """Generalized Focal Loss: Learning Qualified and Distributed Bounding
    Boxes for Dense Object Detection.

    https://arxiv.org/abs/2006.04388
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        stacked_convs: int = 4,
        conv_cfg: dict | None = None,
        norm_cfg: dict = dict(type="GN", num_groups=32, requires_grad=True),
        loss_qfl: dict = dict(type="QualityFocalLoss", use_sigmoid=True, beta=2.0, loss_weight=1.0),
        loss_dfl: dict = dict(type="DistributionFocalLoss", loss_weight=0.25),
        bbox_coder: dict = dict(type="DistancePointBBoxCoder"),
        reg_max: int = 16,
        init_cfg: dict | list[dict] = dict(
            type="Normal",
            layer="Conv2d",
            std=0.01,
            override=dict(type="Normal", name="gfl_cls", std=0.01, bias_prob=0.01),
        ),
        **kwargs,
    ) -> None:
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.reg_max = reg_max
        super(GFLHead, self).__init__(num_classes, in_channels, bbox_coder=bbox_coder, init_cfg=init_cfg, **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg["assigner"])
            self.sampler = PseudoSampler(context=self)

        self.integral = Integral(self.reg_max)
        self.loss_qfl = MODELS.build(loss_qfl)
        self.loss_dfl = MODELS.build(loss_dfl)

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
        self.gfl_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.gfl_reg = nn.Conv2d(self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.prior_generator.strides])

    def forward(self, feats: tuple[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Forward features from the upstream network."""
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x: Tensor, scale: Scale) -> tuple[Tensor, Tensor]:
        """Forward feature of a single scale level."""
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.gfl_cls(cls_feat)
        bbox_pred = scale(self.gfl_reg(reg_feat)).float()
        return cls_score, bbox_pred

    def anchor_center(self, anchors: Tensor) -> Tensor:
        """Get anchor centers from anchors."""
        anchors_cx = (anchors[..., 2] + anchors[..., 0]) / 2
        anchors_cy = (anchors[..., 3] + anchors[..., 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    def loss_by_feat_single(
        self,
        anchors: Tensor,
        cls_score: Tensor,
        bbox_pred: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        stride: tuple[int, int],
        num_total_samples: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute loss of a single scale level."""
        assert stride[0] == stride[1], "h stride is not equal to w stride!"
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = self.bbox_coder.decode(pos_anchor_centers, pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            score[pos_inds] = bbox_overlaps(pos_decode_bbox_pred.detach(), pos_decode_bbox_targets, is_aligned=True)
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = self.bbox_coder.encode(pos_anchor_centers, pos_decode_bbox_targets, self.reg_max).reshape(
                -1
            )

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0,
            )

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0,
            )
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = bbox_pred.new_tensor(0)

        # cls (qfl) loss
        loss_cls = self.loss_qfl(
            cls_score,
            (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples,
        )

        return loss_cls, loss_bbox, loss_dfl, weight_targets.sum()

    def loss_by_feat(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
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

        losses_cls, losses_bbox, losses_dfl, avg_factor = multi_apply(
            self.loss_by_feat_single,
            anchor_list,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            self.prior_generator.strides,
            num_total_samples=num_total_samples,
        )

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(torch.tensor(avg_factor, device=device)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl)

    def predict_by_feat(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        score_factors: list[Tensor] | None = None,
        batch_img_metas: list[dict] | None = None,
        cfg: dict | None = None,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> list:
        """Transform a batch of output features extracted from the head into
        bbox results.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device
        )

        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms,
            )
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(
        self,
        cls_score_list: list[Tensor],
        bbox_pred_list: list[Tensor],
        mlvl_priors: list[Tensor],
        img_meta: dict,
        cfg: dict,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> list:
        """Transform a single image's features extracted from the head into
        bbox results.
        """
        from visdet.engine.structures import InstanceData

        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta["img_shape"]
        nms_pre = cfg.get("nms_pre", -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for level_idx, (cls_score, bbox_pred, stride, priors) in enumerate(
            zip(
                cls_score_list,
                bbox_pred_list,
                self.prior_generator.strides,
                mlvl_priors,
            )
        ):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0)
            bbox_pred = self.integral(bbox_pred) * stride[0]

            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()

            results = filter_scores_and_topk(
                scores, cfg.get("score_thr", 0.0), nms_pre, dict(bbox_pred=bbox_pred, priors=priors)
            )
            scores, labels, _, filtered_results = results

            bbox_pred = filtered_results["bbox_pred"]
            priors = filtered_results["priors"]

            bboxes = self.bbox_coder.decode(self.anchor_center(priors), bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        results = InstanceData()
        results.bboxes = torch.cat(mlvl_bboxes)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta,
        )

    def get_targets(
        self,
        anchor_list: list[list[Tensor]],
        valid_flag_list: list[list[Tensor]],
        batch_gt_instances: list,
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: list | None = None,
    ) -> tuple:
        """Get targets for GFL head."""
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
            pos_bbox_targets = sampling_result.pos_gt_bboxes
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
