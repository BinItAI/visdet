# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from visdet.registry import MODELS


@MODELS.register_module()
class MaskIoUHead(nn.Module):
    """Mask IoU Head."""

    def __init__(
        self,
        num_convs: int = 4,
        num_fcs: int = 2,
        roi_feat_size: int = 14,
        in_channels: int = 256,
        conv_out_channels: int = 256,
        fc_out_channels: int = 1024,
        num_classes: int = 80,
        loss_iou: dict = dict(type="MSELoss", loss_weight=0.5),
    ) -> None:
        super(MaskIoUHead, self).__init__()
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.num_classes = num_classes

        self.convs = nn.ModuleList()
        for i in range(num_convs):
            if i == 0:
                # concatenation of mask feature and mask prediction
                ch = self.in_channels + 1
            else:
                ch = self.conv_out_channels
            stride = 2 if i == num_convs - 1 else 1
            self.convs.append(nn.Conv2d(ch, self.conv_out_channels, 3, stride=stride, padding=1))

        if isinstance(roi_feat_size, int):
            roi_feat_size = (roi_feat_size, roi_feat_size)
        pooled_area = (roi_feat_size[0] // 2) * (roi_feat_size[1] // 2)
        self.fcs = nn.ModuleList()
        for i in range(num_fcs):
            ch = self.conv_out_channels * pooled_area if i == 0 else self.fc_out_channels
            self.fcs.append(nn.Linear(ch, self.fc_out_channels))

        self.fc_mask_iou = nn.Linear(self.fc_out_channels, self.num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.loss_iou = MODELS.build(loss_iou)

    def forward(self, mask_feat: Tensor, mask_pred: Tensor) -> Tensor:
        mask_pred = mask_pred.sigmoid()
        mask_pred_pooled = self.max_pool(mask_pred.unsqueeze(1))

        x = torch.cat((mask_feat, mask_pred_pooled), 1)

        for conv in self.convs:
            x = self.relu(conv(x))
        x = x.flatten(1)
        for fc in self.fcs:
            x = self.relu(fc(x))
        mask_iou = self.fc_mask_iou(x)
        return mask_iou

    def loss(self, mask_iou_pred: Tensor, mask_iou_targets: Tensor) -> dict:
        pos_inds = mask_iou_targets > 0
        if pos_inds.sum() > 0:
            loss_mask_iou = self.loss_iou(mask_iou_pred[pos_inds], mask_iou_targets[pos_inds])
        else:
            loss_mask_iou = mask_iou_pred.sum() * 0
        return dict(loss_mask_iou=loss_mask_iou)

    def get_targets(
        self, sampling_results: list, gt_masks: list, mask_pred: Tensor, mask_targets: Tensor, rcnn_train_cfg: dict
    ) -> Tensor:
        """Compute target of mask IoU."""
        pos_proposals = [res.pos_priors for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]

        area_ratios = map(self._get_area_ratio, pos_proposals, pos_assigned_gt_inds, gt_masks)
        area_ratios = torch.cat(list(area_ratios))
        assert mask_targets.size(0) == area_ratios.size(0)

        mask_thr_binary = getattr(rcnn_train_cfg, "mask_thr_binary", 0.5)
        mask_pred = (mask_pred > mask_thr_binary).float()
        mask_pred_areas = mask_pred.sum((-1, -2))

        # mask_pred and mask_targets are binary maps
        overlap_areas = (mask_pred * mask_targets).sum((-1, -2))

        # compute the mask area of the whole instance
        gt_full_areas = mask_targets.sum((-1, -2)) / (area_ratios + 1e-7)

        mask_iou_targets = overlap_areas / (mask_pred_areas + gt_full_areas - overlap_areas + 1e-7)
        return mask_iou_targets

    def _get_area_ratio(self, pos_proposals: Tensor, pos_assigned_gt_inds: Tensor, gt_masks) -> Tensor:
        """Compute area ratio of the gt mask inside the proposal and the gt
        mask of the corresponding instance."""
        num_pos = pos_proposals.size(0)
        if num_pos > 0:
            area_ratios = []
            proposals_np = pos_proposals.cpu().numpy()
            pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
            # compute mask areas of gt instances
            gt_instance_mask_area = gt_masks.areas
            for i in range(num_pos):
                gt_mask = gt_masks[pos_assigned_gt_inds[i]]

                # crop the gt mask inside the proposal
                bbox = proposals_np[i, :].astype(np.int32)
                gt_mask_in_proposal = gt_mask.crop(bbox)

                ratio = gt_mask_in_proposal.areas[0] / (gt_instance_mask_area[pos_assigned_gt_inds[i]] + 1e-7)
                area_ratios.append(ratio)
            area_ratios = torch.from_numpy(np.stack(area_ratios)).float().to(pos_proposals.device)
        else:
            area_ratios = pos_proposals.new_zeros((0,))
        return area_ratios

    def get_mask_scores(self, mask_iou_pred: Tensor, det_bboxes: Tensor, det_labels: Tensor) -> list:
        """Get the mask scores."""
        inds = range(det_labels.size(0))
        mask_scores = mask_iou_pred[inds, det_labels] * det_bboxes[inds, -1]
        mask_scores = mask_scores.cpu().numpy()
        det_labels = det_labels.cpu().numpy()
        return [mask_scores[det_labels == i] for i in range(self.num_classes)]
