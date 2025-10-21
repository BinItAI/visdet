# ruff: noqa
# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.
"""Cascade RoI head for visdet."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from visengine.structures import InstanceData

from visdet.models.task_modules.samplers import SamplingResult
from visdet.models.test_time_augs import merge_aug_masks
from visdet.models.utils import empty_instances, unpack_gt_instances
from visdet.registry import MODELS, TASK_UTILS
from visdet.structures import SampleList
from visdet.structures.bbox import bbox2roi, get_box_tensor
from visdet.utils import ConfigType, InstanceList, MultiConfig, OptConfigType, OptMultiConfig

from .base_roi_head import BaseRoIHead


@MODELS.register_module()
class CascadeRoIHead(BaseRoIHead):
    """Cascade RoI head including one bbox head and one mask head.

    https://arxiv.org/abs/1712.00726
    """

    def __init__(
        self,
        num_stages: int,
        stage_loss_weights: list[float] | tuple[float, ...],
        bbox_roi_extractor: OptMultiConfig = None,
        bbox_head: OptMultiConfig = None,
        mask_roi_extractor: OptMultiConfig = None,
        mask_head: OptMultiConfig = None,
        shared_head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert shared_head is None, "Shared head is not supported in Cascade RCNN."

        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        super().__init__(
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            shared_head=shared_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
        )

    def init_bbox_head(self, bbox_roi_extractor: MultiConfig, bbox_head: MultiConfig) -> None:
        """Initialize box head and box RoI extractor."""
        self.bbox_roi_extractor = nn.ModuleList()
        self.bbox_head = nn.ModuleList()
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [bbox_roi_extractor for _ in range(self.num_stages)]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
        for roi_extractor_cfg, head_cfg in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(MODELS.build(roi_extractor_cfg))
            self.bbox_head.append(MODELS.build(head_cfg))

    def init_mask_head(self, mask_roi_extractor: MultiConfig, mask_head: MultiConfig) -> None:
        """Initialize mask head and mask RoI extractor."""
        if mask_head is None:
            return

        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head_cfg in mask_head:
            self.mask_head.append(MODELS.build(head_cfg))

        if mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.mask_roi_extractor = nn.ModuleList()
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [mask_roi_extractor for _ in range(self.num_stages)]
            assert len(mask_roi_extractor) == self.num_stages
            for roi_extractor_cfg in mask_roi_extractor:
                self.mask_roi_extractor.append(MODELS.build(roi_extractor_cfg))
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor

    def init_assigner_sampler(self) -> None:
        """Initialize assigner and sampler for each stage."""
        self.bbox_assigner: list | None = []
        self.bbox_sampler: list | None = []
        if self.train_cfg is not None:
            assert isinstance(self.train_cfg, (list, tuple)), "Cascade RCNN expects list-style train_cfg for each stage."
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.bbox_assigner.append(TASK_UTILS.build(rcnn_train_cfg["assigner"]))
                self.current_stage = idx
                self.bbox_sampler.append(
                    TASK_UTILS.build(rcnn_train_cfg["sampler"], default_args=dict(context=self)),
                )

    def _bbox_forward(self, stage: int, x: Tuple[Tensor, ...], rois: Tensor) -> dict:
        """Box head forward function used in both training and testing."""
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[: bbox_roi_extractor.num_inputs], rois)
        cls_score, bbox_pred = bbox_head(bbox_feats)
        return dict(cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

    def bbox_loss(self, stage: int, x: Tuple[Tensor, ...], sampling_results: List[SamplingResult]) -> dict:
        """Run forward function and calculate loss for bbox head in training."""
        bbox_head = self.bbox_head[stage]
        rois = bbox2roi([res.priors for res in sampling_results])
        bbox_results = self._bbox_forward(stage, x, rois)
        bbox_results.update(rois=rois)
        bbox_loss_and_target = bbox_head.loss_and_target(
            cls_score=bbox_results["cls_score"],
            bbox_pred=bbox_results["bbox_pred"],
            rois=rois,
            sampling_results=sampling_results,
            rcnn_train_cfg=self.train_cfg[stage],
        )
        bbox_results.update(bbox_loss_and_target)
        return bbox_results

    def _mask_forward(self, stage: int, x: Tuple[Tensor, ...], rois: Tensor) -> dict:
        """Mask head forward function used in both training and testing."""
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[: mask_roi_extractor.num_inputs], rois)
        mask_preds = mask_head(mask_feats)
        return dict(mask_preds=mask_preds)

    def mask_loss(
        self,
        stage: int,
        x: Tuple[Tensor, ...],
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
    ) -> dict:
        """Run forward function and calculate loss for mask head in training."""
        pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
        mask_results = self._mask_forward(stage, x, pos_rois)
        mask_head = self.mask_head[stage]
        mask_loss_and_target = mask_head.loss_and_target(
            mask_preds=mask_results["mask_preds"],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg[stage],
        )
        mask_results.update(mask_loss_and_target)
        return mask_results

    def loss(self, x: Tuple[Tensor, ...], rpn_results_list: InstanceList, batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection RoI."""
        assert len(rpn_results_list) == len(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = unpack_gt_instances(batch_data_samples)

        num_imgs = len(batch_data_samples)
        losses: dict[str, Tensor] = {}
        results_list = rpn_results_list
        for stage in range(self.num_stages):
            self.current_stage = stage
            stage_loss_weight = self.stage_loss_weights[stage]

            sampling_results: list[SamplingResult] = []
            if self.with_bbox or self.with_mask:
                bbox_assigner = self.bbox_assigner[stage]
                bbox_sampler = self.bbox_sampler[stage]
                for i in range(num_imgs):
                    results = results_list[i]
                    results.priors = results.pop("bboxes")
                    assign_result = bbox_assigner.assign(results, batch_gt_instances[i], batch_gt_instances_ignore[i])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        results,
                        batch_gt_instances[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x],
                    )
                    sampling_results.append(sampling_result)

            bbox_results = self.bbox_loss(stage, x, sampling_results)
            for name, value in bbox_results["loss_bbox"].items():
                losses[f"s{stage}.{name}"] = value * stage_loss_weight if "loss" in name else value

            if self.with_mask:
                mask_results = self.mask_loss(stage, x, sampling_results, batch_gt_instances)
                for name, value in mask_results["loss_mask"].items():
                    losses[f"s{stage}.{name}"] = value * stage_loss_weight if "loss" in name else value

            if stage < self.num_stages - 1:
                bbox_head = self.bbox_head[stage]
                with torch.no_grad():
                    results_list = bbox_head.refine_bboxes(sampling_results, bbox_results, batch_img_metas)
                    if results_list is None:
                        break
        return losses

    def predict_bbox(
        self,
        x: Tuple[Tensor, ...],
        batch_img_metas: List[dict],
        rpn_results_list: InstanceList,
        rcnn_test_cfg: ConfigType,
        rescale: bool = False,
        **kwargs,
    ) -> InstanceList:
        """Perform forward propagation of the bbox head and predict detection results."""
        proposals = [res.bboxes for res in rpn_results_list]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type="bbox",
                box_type=self.bbox_head[-1].predict_box_type,
                num_classes=self.bbox_head[-1].num_classes,
                score_per_cls=rcnn_test_cfg is None,
            )

        rois, cls_scores, bbox_preds = self._refine_roi(
            x=x,
            rois=rois,
            batch_img_metas=batch_img_metas,
            num_proposals_per_img=num_proposals_per_img,
            **kwargs,
        )

        results_list = self.bbox_head[-1].predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rescale=rescale,
            rcnn_test_cfg=rcnn_test_cfg,
        )
        return results_list

    def predict_mask(
        self,
        x: Tuple[Tensor, ...],
        batch_img_metas: List[dict],
        results_list: List[InstanceData],
        rescale: bool = False,
    ) -> List[InstanceData]:
        """Perform forward propagation of the mask head and predict detection results."""
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type="mask",
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary,
            )

        num_mask_rois_per_img = [len(res) for res in results_list]
        aug_masks: list[list[Tensor]] = []
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, mask_rois)
            mask_preds = mask_results["mask_preds"].split(num_mask_rois_per_img, 0)
            aug_masks.append([m.sigmoid().detach() for m in mask_preds])

        merged_masks = []
        for img_idx, img_meta in enumerate(batch_img_metas):
            aug_mask = [mask[img_idx] for mask in aug_masks]
            merged_mask = merge_aug_masks(aug_mask, img_meta)
            merged_masks.append(merged_mask)

        return self.mask_head[-1].predict_by_feat(
            mask_preds=merged_masks,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale,
            activate_map=True,
        )

    def _refine_roi(
        self,
        x: Tuple[Tensor, ...],
        rois: Tensor,
        batch_img_metas: List[dict],
        num_proposals_per_img: Sequence[int],
        **kwargs,
    ) -> tuple:
        """Multi-stage refinement of RoI."""
        ms_scores: list[list[Tensor]] = []
        for stage in range(self.num_stages):
            bbox_results = self._bbox_forward(stage=stage, x=x, rois=rois, **kwargs)
            cls_scores = bbox_results["cls_score"]
            bbox_preds = bbox_results["bbox_pred"]

            rois = rois.split(num_proposals_per_img, 0)
            cls_scores = cls_scores.split(num_proposals_per_img, 0)
            ms_scores.append(cls_scores)

            if bbox_preds is not None:
                if isinstance(bbox_preds, torch.Tensor):
                    bbox_preds = bbox_preds.split(num_proposals_per_img, 0)
                else:
                    bbox_preds = self.bbox_head[stage].bbox_pred_split(bbox_preds, num_proposals_per_img)
            else:
                bbox_preds = (None,) * len(batch_img_metas)

            if stage < self.num_stages - 1:
                bbox_head = self.bbox_head[stage]
                if bbox_head.custom_activation:
                    cls_scores = [bbox_head.loss_cls.get_activation(score) for score in cls_scores]
                refine_rois_list = []
                for img_idx, img_meta in enumerate(batch_img_metas):
                    if rois[img_idx].shape[0] == 0:
                        continue
                    bbox_label = cls_scores[img_idx][:, :-1].argmax(dim=1)
                    refined_bboxes = bbox_head.regress_by_class(rois[img_idx][:, 1:], bbox_label, bbox_preds[img_idx], img_meta)
                    refined_bboxes = get_box_tensor(refined_bboxes)
                    refined_rois = torch.cat([rois[img_idx][:, [0]], refined_bboxes], dim=1)
                    refine_rois_list.append(refined_rois)
                rois = torch.cat(refine_rois_list) if refine_rois_list else rois[0].new_zeros((0, 5))

        cls_scores = [sum(score_set[i] for score_set in ms_scores) / float(len(ms_scores)) for i in range(len(batch_img_metas))]
        return rois, cls_scores, bbox_preds

    def forward(self, x: Tuple[Tensor, ...], rpn_results_list: InstanceList, batch_data_samples: SampleList) -> tuple:
        """Forward method for visualization/debug."""
        results: tuple = ()
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]
        proposals = [rpn_results.bboxes for rpn_results in rpn_results_list]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = bbox2roi(proposals)
        if self.with_bbox:
            rois, cls_scores, bbox_preds = self._refine_roi(x, rois, batch_img_metas, num_proposals_per_img)
            results = results + (cls_scores, bbox_preds)
        if self.with_mask:
            aug_masks = []
            if isinstance(rois, (list, tuple)):
                rois = torch.cat(rois)
            for stage in range(self.num_stages):
                mask_results = self._mask_forward(stage, x, rois)
                mask_preds = mask_results["mask_preds"].split(num_proposals_per_img, 0)
                aug_masks.append([m.sigmoid().detach() for m in mask_preds])

            merged_masks = []
            for img_idx, img_meta in enumerate(batch_img_metas):
                aug_mask = [mask[img_idx] for mask in aug_masks]
                merged_mask = merge_aug_masks(aug_mask, img_meta)
                merged_masks.append(merged_mask)
            results = results + (merged_masks,)
        return results
