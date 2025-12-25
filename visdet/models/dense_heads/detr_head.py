# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from visdet.models.utils.misc import multi_apply
from visdet.registry import MODELS, TASK_UTILS
from visdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from .anchor_free_head import AnchorFreeHead


@MODELS.register_module()
class DETRHead(AnchorFreeHead):
    """Implements the DETR transformer head.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/abs/2005.12872>`_ for details.

    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): The number of fully-connected layers in
            the regression branch. Default: 2.
        transformer (dict, optional): Config for transformer.
        sync_cls_avg_factor (bool, optional): Whether to sync the avg_factor
            of the classification loss across all ranks. Default: False.
        bg_cls_weight (float, optional): The weight of background class in the
            classification loss. Default: 0.1.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        num_query: int = 100,
        num_reg_fcs: int = 2,
        transformer: dict = None,
        sync_cls_avg_factor: bool = False,
        bg_cls_weight: float = 0.1,
        loss_cls: dict = dict(
            type="CrossEntropyLoss",
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0,
        ),
        loss_bbox: dict = dict(type="L1Loss", loss_weight=5.0),
        loss_iou: dict = dict(type="GIoULoss", loss_weight=2.0),
        train_cfg: dict | None = None,
        test_cfg: dict | None = None,
        init_cfg: dict | list[dict] | None = None,
        **kwargs,
    ) -> None:
        super(DETRHead, self).__init__(num_classes, in_channels, init_cfg=init_cfg, **kwargs)
        self.bg_cls_weight = bg_cls_weight
        self.sync_cls_avg_factor = sync_cls_avg_factor

        # In DETR, we don't use the standard loss_cls of AnchorFreeHead
        # because we need to handle the background class weight specifically
        # inside the cross entropy loss or manually.

        self.num_query = num_query
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_iou = MODELS.build(loss_iou)

        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg["assigner"])
            # DETR doesn't use sampler, but we can set a pseudo one
            self.sampler = TASK_UTILS.build(dict(type="PseudoSampler"), default_args=dict(context=self))

        self._init_layers()
        self.transformer = MODELS.build(transformer)
        self.embed_dims = self.transformer.d_model
        self._init_transformer_weights()

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.input_proj = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes + 1)
        self.reg_ffn = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_channels, self.in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_channels, 4),
        )
        self.query_embedding = nn.Embedding(self.num_query, self.in_channels)

    def _init_transformer_weights(self) -> None:
        """Initialize transformer weights."""
        # The transformer weights are initialized in the transformer itself
        pass

    def forward(self, feats: tuple[Tensor], img_metas: list[dict]) -> tuple[Tensor, Tensor]:
        """Forward function.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: Outputs from the transformer head.
                - all_cls_scores (Tensor): (num_decoder_layers, batch_size,
                  num_query, cls_out_channels)
                - all_bbox_preds (Tensor): (num_decoder_layers, batch_size,
                  num_query, 4)
        """
        # We only use the last feature map
        x = feats[-1]
        batch_size, _, height, width = x.shape

        # construct binary masks which is the input to transformer
        masks = x.new_zeros((batch_size, height, width)).to(torch.bool)
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]["img_shape"]
            pad_h, pad_w = img_metas[img_id]["pad_shape"][:2]
            # Since input images are padded, we want to mask out the padded area
            # We need to calculate the valid area in feature map
            valid_h = int(img_h / pad_h * height)
            valid_w = int(img_w / pad_w * width)
            masks[img_id, :valid_h, :valid_w] = False
            # The padding part is set to True (masked)
            # Wait, PyTorch transformer mask: True means ignored.
            # So valid area should be False.
            if valid_h < height:
                masks[img_id, valid_h:, :] = True
            if valid_w < width:
                masks[img_id, :, valid_w:] = True

        # position embedding
        # For simplicity, we use a fixed sine position embedding generated on the fly or learned.
        # Here we assume it's passed or we implement a simple one.
        # Let's implement a simple learnable one or sine one.
        # Actually DETR uses sine positional encoding.
        # I'll implement a simple version here.
        pos_embed = self.positional_encoding(masks)

        # projector
        x = self.input_proj(x)

        # transformer
        hs, _ = self.transformer(x, masks, self.query_embedding.weight, pos_embed)

        # prediction
        outputs_class = self.fc_cls(hs)
        outputs_coord = self.reg_ffn(hs).sigmoid()

        return outputs_class, outputs_coord

    def positional_encoding(self, masks):
        # Implement sine positional encoding
        # This is a simplified version
        not_mask = ~masks
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.training:
            # During training, we use the real mask
            pass

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * 3.14159
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * 3.14159

        dim_t = torch.arange(self.embed_dims // 2, dtype=torch.float32, device=masks.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / (self.embed_dims // 2))

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def loss(
        self,
        outputs_class,
        outputs_coord,
        batch_gt_instances,
        batch_img_metas,
        gt_bboxes_ignore=None,
    ) -> dict:
        # Get the last layer outputs
        cls_scores = outputs_class[-1]
        bbox_preds = outputs_coord[-1]

        # Convert to list of tensors for each image
        batch_size = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(batch_size)]
        bbox_preds_list = [bbox_preds[i] for i in range(batch_size)]

        gt_bboxes_list = [gt.bboxes for gt in batch_gt_instances]
        gt_labels_list = [gt.labels for gt in batch_gt_instances]

        # Assign targets
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = multi_apply(
            self.get_targets_single,
            cls_scores_list,
            bbox_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            batch_img_metas,
        )

        # Calculate loss
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # Classification loss
        cls_scores = cls_scores.reshape(-1, self.num_classes + 1)
        # Construct weighted loss for background
        # Usually DETR uses CrossEntropy with a weight for the background class

        # For simplicity, let's assume loss_cls handles it (e.g. CrossEntropyLoss)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=num_total_pos + num_total_neg)

        # Regression loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        pos_inds = labels < self.num_classes
        if pos_inds.sum() > 0:
            pos_bbox_preds = bbox_preds[pos_inds]
            pos_bbox_targets = bbox_targets[pos_inds]
            loss_bbox = self.loss_bbox(pos_bbox_preds, pos_bbox_targets, avg_factor=num_total_pos)

            # IoU loss
            pos_bbox_preds_xyxy = bbox_cxcywh_to_xyxy(pos_bbox_preds)
            pos_bbox_targets_xyxy = bbox_cxcywh_to_xyxy(pos_bbox_targets)
            loss_iou = self.loss_iou(pos_bbox_preds_xyxy, pos_bbox_targets_xyxy, avg_factor=num_total_pos)
        else:
            loss_bbox = bbox_preds.sum() * 0
            loss_iou = bbox_preds.sum() * 0

        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_iou=loss_iou)

    def get_targets_single(self, cls_score, bbox_pred, gt_bboxes, gt_labels, img_meta):
        # Assign
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes, gt_labels, img_meta)

        # Get targets
        num_bboxes = bbox_pred.size(0)
        labels = cls_score.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        label_weights = cls_score.new_ones(num_bboxes)
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)

        pos_inds = assign_result.gt_inds > 0
        if pos_inds.sum() > 0:
            assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
            labels[pos_inds] = gt_labels[assigned_gt_inds]

            # Normalize GT bboxes to [0, 1] (cx, cy, w, h)
            img_h, img_w = img_meta["img_shape"][:2]
            factor = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h])

            pos_gt_bboxes = gt_bboxes[assigned_gt_inds] / factor
            pos_gt_bboxes_cxcywh = bbox_xyxy_to_cxcywh(pos_gt_bboxes)
            bbox_targets[pos_inds] = pos_gt_bboxes_cxcywh
            bbox_weights[pos_inds] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights, pos_inds.sum(), (1 - pos_inds.float()).sum()

    def predict(self, feats, batch_data_samples, rescale=True):
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]
        outputs_class, outputs_coord = self(feats, batch_img_metas)

        # Use last layer
        cls_scores = outputs_class[-1]
        bbox_preds = outputs_coord[-1]

        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_meta = batch_img_metas[img_id]

            scores = F.softmax(cls_score, dim=-1)[:, :-1]  # Exclude background
            scores, labels = scores.max(dim=-1)

            # Convert to xyxy and rescale
            img_h, img_w = img_meta["img_shape"][:2]
            factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h])
            bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred) * factor

            # Filter low scores
            # Note: DETR typically returns all queries, but we might filter for evaluation
            valid_mask = scores > 0.0  # Return all

            from visdet.engine.structures import InstanceData

            results = InstanceData()
            results.bboxes = bbox_pred[valid_mask]
            results.scores = scores[valid_mask]
            results.labels = labels[valid_mask]
            result_list.append(results)

        # Add to data samples
        for data_sample, results in zip(batch_data_samples, result_list):
            data_sample.pred_instances = results

        return batch_data_samples
