# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor

from visdet.models.utils.misc import filter_scores_and_topk, multi_apply
from visdet.registry import MODELS

from .anchor_free_head import AnchorFreeHead

INF = 1e8


@MODELS.register_module()
class FoveaHead(AnchorFreeHead):
    """FoveaBox: Beyond Anchor-based Object Detector
    https://arxiv.org/abs/1904.03797
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        base_edge_list: tuple[int, ...] = (16, 32, 64, 128, 256),
        scale_ranges: tuple[tuple[int, int], ...] = ((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
        sigma: float = 0.4,
        with_deform: bool = False,
        deform_groups: int = 4,
        init_cfg: dict | list[dict] = dict(
            type="Normal",
            layer="Conv2d",
            std=0.01,
            override=dict(type="Normal", name="conv_cls", std=0.01, bias_prob=0.01),
        ),
        **kwargs,
    ) -> None:
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.sigma = sigma
        self.with_deform = with_deform
        self.deform_groups = deform_groups
        super().__init__(num_classes, in_channels, init_cfg=init_cfg, **kwargs)

    def _init_layers(self) -> None:
        # box branch
        super()._init_reg_convs()
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

        # cls branch
        if not self.with_deform:
            super()._init_cls_convs()
            self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        else:
            # TODO: implement feature adaption with deform conv if needed
            raise NotImplementedError("with_deform=True is not supported in visdet yet")

    def forward_single(self, x: Tensor) -> tuple[Tensor, Tensor]:
        cls_feat = x
        reg_feat = x
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)
        return cls_score, bbox_pred

    def loss_by_feat(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        batch_gt_instances: list,
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: list | None = None,
    ) -> dict:
        assert len(cls_scores) == len(bbox_preds)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        points = self.prior_generator.grid_priors(featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device)
        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cls_score in cls_scores
        ]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4) for bbox_pred in bbox_preds]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)

        gt_bboxes = [gt.bboxes for gt in batch_gt_instances]
        gt_labels = [gt.labels for gt in batch_gt_instances]

        flatten_labels, flatten_bbox_targets = self.get_targets(gt_bboxes, gt_labels, featmap_sizes, points)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((flatten_labels >= 0) & (flatten_labels < self.num_classes)).nonzero().view(-1)
        num_pos = len(pos_inds)

        loss_cls = self.loss_cls(flatten_cls_scores, flatten_labels, avg_factor=num_pos + num_imgs)
        if num_pos > 0:
            pos_bbox_preds = flatten_bbox_preds[pos_inds]
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_weights = pos_bbox_targets.new_zeros(pos_bbox_targets.size()) + 1.0
            loss_bbox = self.loss_bbox(pos_bbox_preds, pos_bbox_targets, pos_weights, avg_factor=num_pos)
        else:
            loss_bbox = flatten_bbox_preds.sum() * 0
        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)

    def get_targets(
        self,
        gt_bbox_list: list[Tensor],
        gt_label_list: list[Tensor],
        featmap_sizes: list[tuple[int, int]],
        points: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        label_list, bbox_target_list = multi_apply(
            self._get_target_single,
            gt_bbox_list,
            gt_label_list,
            featmap_size_list=featmap_sizes,
            point_list=points,
        )
        flatten_labels = [
            torch.cat([labels_level_img.flatten() for labels_level_img in labels_level])
            for labels_level in zip(*label_list)
        ]
        flatten_bbox_targets = [
            torch.cat([bbox_targets_level_img.reshape(-1, 4) for bbox_targets_level_img in bbox_targets_level])
            for bbox_targets_level in zip(*bbox_target_list)
        ]
        flatten_labels = torch.cat(flatten_labels)
        flatten_bbox_targets = torch.cat(flatten_bbox_targets)
        return flatten_labels, flatten_bbox_targets

    def _get_target_single(
        self,
        gt_bboxes_raw: Tensor,
        gt_labels_raw: Tensor,
        featmap_size_list: list[tuple[int, int]],
        point_list: list[Tensor],
    ) -> tuple[list[Tensor], list[Tensor]]:
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))
        label_list = []
        bbox_target_list = []
        # for each pyramid, find the cls and box target
        for base_len, (lower_bound, upper_bound), stride, featmap_size, points in zip(
            self.base_edge_list,
            self.scale_ranges,
            self.strides,
            featmap_size_list,
            point_list,
        ):
            # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
            points = points.view(*featmap_size, 2)
            x, y = points[..., 0], points[..., 1]
            labels = gt_labels_raw.new_zeros(featmap_size) + self.num_classes
            bbox_targets = gt_bboxes_raw.new_full((featmap_size[0], featmap_size[1], 4), 1.0)
            # scale assignment
            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            if len(hit_indices) == 0:
                label_list.append(labels)
                bbox_target_list.append(torch.log(bbox_targets))
                continue
            _, hit_index_order = torch.sort(-gt_areas[hit_indices])
            hit_indices = hit_indices[hit_index_order]
            gt_bboxes = gt_bboxes_raw[hit_indices, :] / stride
            gt_labels = gt_labels_raw[hit_indices]
            half_w = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0])
            half_h = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
            # valid fovea area: left, right, top, down
            pos_left = (
                torch.ceil(gt_bboxes[:, 0] + (1 - self.sigma) * half_w - 0.5).long().clamp(0, featmap_size[1] - 1)
            )
            pos_right = (
                torch.floor(gt_bboxes[:, 0] + (1 + self.sigma) * half_w - 0.5).long().clamp(0, featmap_size[1] - 1)
            )
            pos_top = torch.ceil(gt_bboxes[:, 1] + (1 - self.sigma) * half_h - 0.5).long().clamp(0, featmap_size[0] - 1)
            pos_down = (
                torch.floor(gt_bboxes[:, 1] + (1 + self.sigma) * half_h - 0.5).long().clamp(0, featmap_size[0] - 1)
            )
            for px1, py1, px2, py2, label, (gt_x1, gt_y1, gt_x2, gt_y2) in zip(
                pos_left,
                pos_top,
                pos_right,
                pos_down,
                gt_labels,
                gt_bboxes_raw[hit_indices, :],
            ):
                if px1 > px2 or py1 > py2:
                    continue
                labels[py1 : py2 + 1, px1 : px2 + 1] = label
                bbox_targets[py1 : py2 + 1, px1 : px2 + 1, 0] = (x[py1 : py2 + 1, px1 : px2 + 1] - gt_x1) / base_len
                bbox_targets[py1 : py2 + 1, px1 : px2 + 1, 1] = (y[py1 : py2 + 1, px1 : px2 + 1] - gt_y1) / base_len
                bbox_targets[py1 : py2 + 1, px1 : px2 + 1, 2] = (gt_x2 - x[py1 : py2 + 1, px1 : px2 + 1]) / base_len
                bbox_targets[py1 : py2 + 1, px1 : px2 + 1, 3] = (gt_y2 - y[py1 : py2 + 1, px1 : px2 + 1]) / base_len
            bbox_targets = bbox_targets.clamp(min=1.0 / 16, max=16.0)
            label_list.append(labels)
            bbox_target_list.append(torch.log(bbox_targets))
        return label_list, bbox_target_list

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
        cfg = self.test_cfg if cfg is None else cfg
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
        from visdet.engine.structures import InstanceData

        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta["img_shape"]
        nms_pre = getattr(cfg, "nms_pre", -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for level_idx, (cls_score, bbox_pred, stride, base_len, priors) in enumerate(
            zip(
                cls_score_list,
                bbox_pred_list,
                self.strides,
                self.base_edge_list,
                mlvl_priors,
            )
        ):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()

            results = filter_scores_and_topk(
                scores, getattr(cfg, "score_thr", 0.0), nms_pre, dict(bbox_pred=bbox_pred, priors=priors)
            )
            scores, labels, _, filtered_results = results

            bbox_pred = filtered_results["bbox_pred"]
            priors = filtered_results["priors"]

            bboxes = self._bbox_decode(priors, bbox_pred, base_len, img_shape)
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

    def _bbox_decode(self, priors: Tensor, bbox_pred: Tensor, base_len: int, max_shape: tuple[int, int]) -> Tensor:
        bbox_pred = bbox_pred.exp()

        y = priors[:, 1]
        x = priors[:, 0]
        x1 = (x - base_len * bbox_pred[:, 0]).clamp(min=0, max=max_shape[1] - 1)
        y1 = (y - base_len * bbox_pred[:, 1]).clamp(min=0, max=max_shape[0] - 1)
        x2 = (x + base_len * bbox_pred[:, 2]).clamp(min=0, max=max_shape[1] - 1)
        y2 = (y + base_len * bbox_pred[:, 3]).clamp(min=0, max=max_shape[0] - 1)
        decoded_bboxes = torch.stack([x1, y1, x2, y2], -1)
        return decoded_bboxes
