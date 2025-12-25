# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor

from visdet.cv.cnn import ConvModule
from visdet.models.utils.misc import multi_apply
from visdet.registry import MODELS
from visdet.utils import reduce_mean

from .anchor_head import AnchorHead

INF = 1e8


def levels_to_images(mlvl_tensor: list[Tensor]) -> list[Tensor]:
    """Concat multi-level feature maps by image."""
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    channels = mlvl_tensor[0].size(1)
    for t in mlvl_tensor:
        t = t.permute(0, 2, 3, 1)
        t = t.view(batch_size, -1, channels).contiguous()
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]


@MODELS.register_module()
class YOLOFHead(AnchorHead):
    """YOLOFHead Paper link: https://arxiv.org/abs/2103.09460."""

    def __init__(
        self,
        num_classes,
        in_channels,
        num_cls_convs=2,
        num_reg_convs=4,
        norm_cfg=dict(type="BN", requires_grad=True),
        **kwargs,
    ):
        self.num_cls_convs = num_cls_convs
        self.num_reg_convs = num_reg_convs
        self.norm_cfg = norm_cfg
        super(YOLOFHead, self).__init__(num_classes, in_channels, **kwargs)

    def _init_layers(self):
        cls_subnet = []
        bbox_subnet = []
        for i in range(self.num_cls_convs):
            cls_subnet.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                )
            )
        for i in range(self.num_reg_convs):
            bbox_subnet.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                )
            )
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(
            self.in_channels,
            self.num_base_priors * self.num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bbox_pred = nn.Conv2d(
            self.in_channels,
            self.num_base_priors * 4,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.object_pred = nn.Conv2d(self.in_channels, self.num_base_priors, kernel_size=3, stride=1, padding=1)

    def forward_single(self, feature: Tensor) -> tuple[Tensor, Tensor]:
        cls_score = self.cls_score(self.cls_subnet(feature))
        N, _, H, W = cls_score.shape
        cls_score = cls_score.view(N, -1, self.num_classes, H, W)

        reg_feat = self.bbox_subnet(feature)
        bbox_reg = self.bbox_pred(reg_feat)
        objectness = self.object_pred(reg_feat)

        # implicit objectness
        objectness = objectness.view(N, -1, 1, H, W)
        normalized_cls_score = (
            cls_score
            + objectness
            - torch.log(1.0 + torch.clamp(cls_score.exp(), max=INF) + torch.clamp(objectness.exp(), max=INF))
        )
        normalized_cls_score = normalized_cls_score.view(N, -1, H, W)
        return normalized_cls_score, bbox_reg

    def loss(
        self,
        x: tuple[Tensor],
        batch_data_samples: list,
    ) -> dict:
        """Compute losses of the head."""
        outs = self(x)
        cls_scores = outs[0]
        bbox_preds = outs[1]

        batch_gt_instances = [data_samples.gt_instances for data_samples in batch_data_samples]
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]
        batch_gt_instances_ignore = [
            getattr(data_samples, "ignored_instances", None) for data_samples in batch_data_samples
        ]

        assert len(cls_scores) == 1
        assert self.prior_generator.num_levels == 1

        device = cls_scores[0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, batch_img_metas, device=device)

        # The output level is always 1
        anchor_list = [anchors[0] for anchors in anchor_list]
        valid_flag_list = [valid_flags[0] for valid_flags in valid_flag_list]

        cls_scores_list = levels_to_images(cls_scores)
        bbox_preds_list = levels_to_images(bbox_preds)

        gt_bboxes = [gt.bboxes for gt in batch_gt_instances]
        gt_labels = [gt.labels for gt in batch_gt_instances]
        gt_bboxes_ignore = batch_gt_instances_ignore

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            batch_img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
        )
        if cls_reg_targets is None:
            return None
        (
            batch_labels,
            batch_label_weights,
            num_total_pos,
            num_total_neg,
            batch_bbox_weights,
            batch_pos_predicted_boxes,
            batch_target_boxes,
        ) = cls_reg_targets

        flatten_labels = batch_labels.reshape(-1)
        batch_label_weights = batch_label_weights.reshape(-1)
        cls_score = cls_scores[0].permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)

        num_total_samples = (num_total_pos + num_total_neg) if self.sampling else num_total_pos
        num_total_samples = reduce_mean(cls_score.new_tensor(num_total_samples)).clamp_(1.0).item()

        # classification loss
        loss_cls = self.loss_cls(cls_score, flatten_labels, batch_label_weights, avg_factor=num_total_samples)

        # regression loss
        if batch_pos_predicted_boxes.shape[0] == 0:
            # no pos sample
            loss_bbox = batch_pos_predicted_boxes.sum() * 0
        else:
            loss_bbox = self.loss_bbox(
                batch_pos_predicted_boxes,
                batch_target_boxes,
                batch_bbox_weights.float(),
                avg_factor=num_total_samples,
            )

        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)

    def get_targets(
        self,
        cls_scores_list,
        bbox_preds_list,
        anchor_list,
        valid_flag_list,
        gt_bboxes_list,
        img_metas,
        gt_bboxes_ignore_list=None,
        gt_labels_list=None,
        label_channels=1,
        unmap_outputs=True,
    ):
        """Compute regression and classification targets for anchors in
        multiple images.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            bbox_preds_list,
            anchor_list,
            valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs,
        )
        (
            all_labels,
            all_label_weights,
            pos_inds_list,
            neg_inds_list,
            sampling_results_list,
        ) = results[:5]
        rest_results = list(results[5:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])

        batch_labels = torch.stack(all_labels, 0)
        batch_label_weights = torch.stack(all_label_weights, 0)

        res = (batch_labels, batch_label_weights, num_total_pos, num_total_neg)
        for i, rests in enumerate(rest_results):  # user-added return values
            rest_results[i] = torch.cat(rests, 0)

        return res + tuple(rest_results)

    def _get_targets_single(
        self,
        bbox_preds,
        flat_anchors,
        valid_flags,
        gt_bboxes,
        gt_bboxes_ignore,
        gt_labels,
        img_meta,
        label_channels=1,
        unmap_outputs=True,
    ):
        from visdet.models.task_modules.prior_generators import anchor_inside_flags

        inside_flags = anchor_inside_flags(
            flat_anchors,
            valid_flags,
            img_meta["img_shape"][:2],
            self.train_cfg.get("allowed_border", 0),
        )
        if not inside_flags.any():
            return (None,) * 8
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        bbox_preds = bbox_preds.reshape(-1, 4)
        bbox_preds = bbox_preds[inside_flags, :]

        # decoded bbox
        decoder_bbox_preds = self.bbox_coder.decode(anchors, bbox_preds)
        assign_result = self.assigner.assign(
            decoder_bbox_preds,
            anchors,
            gt_bboxes,
            gt_bboxes_ignore,
            None if self.sampling else gt_labels,
        )

        # In YOLOF, the assigner returns extra properties
        pos_bbox_weights = assign_result.get_extra_property("pos_idx")
        pos_predicted_boxes = assign_result.get_extra_property("pos_predicted_boxes")
        pos_target_boxes = assign_result.get_extra_property("target_boxes")

        from visdet.engine.structures import InstanceData

        pred_instances = InstanceData(priors=anchors)
        gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)

        sampling_result = self.sampler.sample(assign_result, pred_instances, gt_instances)

        num_valid_anchors = anchors.shape[0]
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if gt_labels is None:
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.get("pos_weight", -1) <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg["pos_weight"]
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            from visdet.models.utils.misc import unmap

            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)

        return (
            labels,
            label_weights,
            pos_inds,
            neg_inds,
            sampling_result,
            pos_bbox_weights,
            pos_predicted_boxes,
            pos_target_boxes,
        )
