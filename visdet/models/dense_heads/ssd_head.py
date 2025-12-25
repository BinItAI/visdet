# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from visdet.cv.cnn import ConvModule
from visdet.models.utils import multi_apply
from visdet.registry import MODELS
from .anchor_head import AnchorHead


@MODELS.register_module()
class SSDHead(AnchorHead):
    """SSD head used in https://arxiv.org/abs/1512.02325.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (tuple[int]): Number of channels in the input feature maps.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Default: 0.
        feat_channels (int): Number of hidden channels when stacked_convs
            > 0. Default: 256.
        use_depthwise (bool): Whether to use DepthwiseSeparableConv.
            Default: False.
        conv_cfg (dict): Dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: None.
        act_cfg (dict): Dictionary to construct and config activation layer.
            Default: None.
        anchor_generator (dict): Config dict for anchor generator.
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes. Default False.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        num_classes: int = 80,
        in_channels: tuple[int] = (512, 1024, 512, 256, 256, 256),
        stacked_convs: int = 0,
        feat_channels: int = 256,
        use_depthwise: bool = False,
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
        anchor_generator: dict = dict(
            type="SSDAnchorGenerator",
            scale_major=False,
            input_size=300,
            strides=[8, 16, 32, 64, 100, 300],
            ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
            basesize_ratio_range=(0.1, 0.9),
        ),
        bbox_coder: dict = dict(
            type="DeltaXYWHBBoxCoder",
            clip_border=True,
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        reg_decoded_bbox: bool = False,
        train_cfg: dict | None = None,
        test_cfg: dict | None = None,
        init_cfg: dict | list[dict] = dict(type="Xavier", layer="Conv2d", distribution="uniform", bias=0),
        **kwargs,
    ) -> None:
        self.use_depthwise = use_depthwise
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        super().__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            bbox_coder=bbox_coder,
            reg_decoded_bbox=reg_decoded_bbox,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            **kwargs,
        )

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for channel, num_base_priors in zip(self.in_channels, self.prior_generator.num_base_priors):
            cls_layers = []
            reg_layers = []
            in_channel = channel
            # build stacked conv tower, not used in default ssd
            for i in range(self.stacked_convs):
                cls_layers.append(
                    ConvModule(
                        in_channel,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    )
                )
                reg_layers.append(
                    ConvModule(
                        in_channel,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    )
                )
                in_channel = self.feat_channels

            # Note: Depthwise separable conv not fully implemented here for simplicity
            # following standard SSD implementation.

            cls_layers.append(
                nn.Conv2d(
                    in_channel,
                    num_base_priors * self.cls_out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            reg_layers.append(
                nn.Conv2d(
                    in_channel,
                    num_base_priors * 4,
                    kernel_size=3,
                    padding=1,
                )
            )
            self.cls_convs.append(nn.Sequential(*cls_layers))
            self.reg_convs.append(nn.Sequential(*reg_layers))

    def forward(self, feats: tuple[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Forward features from the upstream network."""
        cls_scores = []
        bbox_preds = []
        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs, self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))
        return cls_scores, bbox_preds

    def loss_by_feat_single(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        anchor: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        bbox_weights: Tensor,
        avg_factor: int,
    ) -> tuple[Tensor, Tensor]:
        """Compute loss of a single image."""
        # SSD uses Hard Negative Mining, which is usually implemented here.
        # However, for simplicity and consistency with AnchorHead,
        # we can use the default loss if OHEM is not strictly required
        # or implement it here.

        # MMDetection's SSDHead.loss_single implements HNM.

        loss_cls_all = F.cross_entropy(cls_score, labels, reduction="none") * label_weights
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero(as_tuple=False).reshape(-1)
        neg_inds = (labels == self.num_classes).nonzero(as_tuple=False).view(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.train_cfg.get("neg_pos_ratio", 3) * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)

        if num_neg_samples > 0:
            topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
            loss_cls_neg = topk_loss_cls_neg.sum()
        else:
            loss_cls_neg = loss_cls_all.new_tensor(0.0)

        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / avg_factor

        if self.reg_decoded_bbox:
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)

        # Use SmoothL1Loss for SSD
        loss_bbox = self.loss_bbox(
            bbox_pred[pos_inds],
            bbox_targets[pos_inds],
            bbox_weights[pos_inds],
            avg_factor=avg_factor,
        )
        return loss_cls, loss_bbox

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
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            avg_factor,
        ) = cls_reg_targets

        num_images = len(batch_img_metas)
        all_cls_scores = torch.cat(
            [s.permute(0, 2, 3, 1).reshape(num_images, -1, self.cls_out_channels) for s in cls_scores],
            1,
        )
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list, -1).view(num_images, -1)
        all_bbox_preds = torch.cat([b.permute(0, 2, 3, 1).reshape(num_images, -1, 4) for b in bbox_preds], -2)
        all_bbox_targets = torch.cat(bbox_targets_list, -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list, -2).view(num_images, -1, 4)

        all_anchors = []
        for i in range(num_images):
            all_anchors.append(torch.cat(anchor_list[i]))

        # In SSD, avg_factor is usually num_total_pos
        # But we need to handle it per image if we use multi_apply

        losses_cls, losses_bbox = multi_apply(
            self.loss_by_feat_single,
            all_cls_scores,
            all_bbox_preds,
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            avg_factor=avg_factor,
        )
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
