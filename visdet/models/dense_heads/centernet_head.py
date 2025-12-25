# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch import Tensor

from visdet.models.utils import (
    gaussian_radius,
    gen_gaussian_target,
    get_local_maximum,
    get_topk_from_heatmap,
    transpose_and_gather_feat,
)
from visdet.models.utils.misc import multi_apply
from visdet.registry import MODELS

from .base_dense_head import BaseDenseHead


@MODELS.register_module()
class CenterNetHead(BaseDenseHead):
    """Objects as Points Head. CenterHead use center_point to indicate object's
    position. Paper link <https://arxiv.org/abs/1904.07850>
    """

    def __init__(
        self,
        in_channel: int,
        feat_channel: int,
        num_classes: int,
        loss_center_heatmap: dict = dict(type="GaussianFocalLoss", loss_weight=1.0),
        loss_wh: dict = dict(type="L1Loss", loss_weight=0.1),
        loss_offset: dict = dict(type="L1Loss", loss_weight=1.0),
        train_cfg: dict | None = None,
        test_cfg: dict | None = None,
        init_cfg: dict | list[dict] | None = None,
    ) -> None:
        super(CenterNetHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.heatmap_head = self._build_head(in_channel, feat_channel, num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)

        self.loss_center_heatmap = MODELS.build(loss_center_heatmap)
        self.loss_wh = MODELS.build(loss_wh)
        self.loss_offset = MODELS.build(loss_offset)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _build_head(self, in_channel: int, feat_channel: int, out_channel: int) -> nn.Sequential:
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1),
        )
        return layer

    def forward(self, feats: tuple[Tensor]) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """Forward features."""
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward feature of a single level."""
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)
        return center_heatmap_pred, wh_pred, offset_pred

    def loss_by_feat(
        self,
        center_heatmap_preds: list[Tensor],
        wh_preds: list[Tensor],
        offset_preds: list[Tensor],
        batch_gt_instances: list,
        batch_img_metas: list[dict],
        batch_gt_instances_ignore: list | None = None,
    ) -> dict:
        """Compute losses of the head."""
        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == 1
        center_heatmap_pred = center_heatmap_preds[0]
        wh_pred = wh_preds[0]
        offset_pred = offset_preds[0]

        target_result, avg_factor = self.get_targets(
            batch_gt_instances, center_heatmap_pred.shape, batch_img_metas[0]["pad_shape"]
        )

        center_heatmap_target = target_result["center_heatmap_target"]
        wh_target = target_result["wh_target"]
        offset_target = target_result["offset_target"]
        wh_offset_target_weight = target_result["wh_offset_target_weight"]

        loss_center_heatmap = self.loss_center_heatmap(
            center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor
        )
        loss_wh = self.loss_wh(wh_pred, wh_target, wh_offset_target_weight, avg_factor=avg_factor * 2)
        loss_offset = self.loss_offset(
            offset_pred,
            offset_target,
            wh_offset_target_weight,
            avg_factor=avg_factor * 2,
        )
        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset,
        )

    def get_targets(
        self, batch_gt_instances: list, feat_shape: torch.Size, img_shape: tuple[int, int]
    ) -> tuple[dict, float]:
        """Compute regression and classification targets in multiple images."""
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        device = batch_gt_instances[0].bboxes.device
        center_heatmap_target = batch_gt_instances[0].bboxes.new_zeros([bs, self.num_classes, feat_h, feat_w])
        wh_target = batch_gt_instances[0].bboxes.new_zeros([bs, 2, feat_h, feat_w])
        offset_target = batch_gt_instances[0].bboxes.new_zeros([bs, 2, feat_h, feat_w])
        wh_offset_target_weight = batch_gt_instances[0].bboxes.new_zeros([bs, 2, feat_h, feat_w])

        for batch_id in range(bs):
            gt_bbox = batch_gt_instances[batch_id].bboxes
            gt_label = batch_gt_instances[batch_id].labels
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                radius = gaussian_radius([scale_box_h, scale_box_w], min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind], [ctx_int, cty_int], radius)

                wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h

                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int

                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            offset_target=offset_target,
            wh_offset_target_weight=wh_offset_target_weight,
        )
        return target_result, avg_factor

    def predict_by_feat(
        self,
        center_heatmap_preds: list[Tensor],
        wh_preds: list[Tensor],
        offset_preds: list[Tensor],
        batch_img_metas: list[dict] | None = None,
        rescale: bool = True,
        with_nms: bool = False,
    ) -> list:
        """Transform network output for a batch into bbox predictions."""
        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == 1
        result_list = []
        for img_id in range(len(batch_img_metas)):
            result_list.append(
                self._predict_by_feat_single(
                    center_heatmap_preds[0][img_id : img_id + 1, ...],
                    wh_preds[0][img_id : img_id + 1, ...],
                    offset_preds[0][img_id : img_id + 1, ...],
                    batch_img_metas[img_id],
                    rescale=rescale,
                    with_nms=with_nms,
                )
            )
        return result_list

    def _predict_by_feat_single(
        self,
        center_heatmap_pred: Tensor,
        wh_pred: Tensor,
        offset_pred: Tensor,
        img_meta: dict,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> list:
        """Transform outputs of a single image into bbox results."""
        from visdet.engine.structures import InstanceData
        from visdet.models.utils.misc import batched_nms

        batch_det_bboxes, batch_labels = self.decode_heatmap(
            center_heatmap_pred,
            wh_pred,
            offset_pred,
            img_meta["batch_input_shape"],
            k=getattr(self.test_cfg, "topk", 100),
            kernel=getattr(self.test_cfg, "local_maximum_kernel", 3),
        )

        det_bboxes = batch_det_bboxes.view([-1, 5])
        det_labels = batch_labels.view(-1)

        if rescale:
            det_bboxes[..., :4] /= det_bboxes.new_tensor(img_meta["scale_factor"])

        if with_nms:
            det_bboxes, det_labels = batched_nms(det_bboxes[:, :4], det_bboxes[:, 4], det_labels, self.test_cfg.nms)

        results = InstanceData()
        results.bboxes = det_bboxes[:, :4]
        results.scores = det_bboxes[:, 4]
        results.labels = det_labels

        return results

    def decode_heatmap(self, center_heatmap_pred, wh_pred, offset_pred, img_shape, k=100, kernel=3):
        """Transform outputs into detections raw bbox prediction."""
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]
        tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
        tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
        br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
        br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]), dim=-1)
        return batch_bboxes, batch_topk_labels
