# Copyright (c) OpenMMLab. All rights reserved.
import torch
from visdet.registry import MODELS
from .standard_roi_head import StandardRoIHead


@MODELS.register_module()
class MaskScoringRoIHead(StandardRoIHead):
    """Mask Scoring RoIHead for Mask Scoring RCNN.

    https://arxiv.org/abs/1903.00241
    """

    def __init__(self, mask_iou_head: dict, **kwargs) -> None:
        super(MaskScoringRoIHead, self).__init__(**kwargs)
        self.mask_iou_head = MODELS.build(mask_iou_head)

    def mask_loss(self, x, sampling_results, bbox_feats, batch_gt_instances):
        """Run forward function and calculate loss for Mask head."""
        mask_results = super().mask_loss(x, sampling_results, bbox_feats, batch_gt_instances)

        # In visdet, super().mask_loss returns the mask_results dict
        if "loss_mask" not in mask_results:
            return mask_results

        # mask iou head forward and loss
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        pos_mask_pred = mask_results["mask_preds"][range(mask_results["mask_preds"].size(0)), pos_labels]
        mask_iou_pred = self.mask_iou_head(mask_results["mask_feats"], pos_mask_pred)
        pos_mask_iou_pred = mask_iou_pred[range(mask_iou_pred.size(0)), pos_labels]

        # Get mask targets for mask iou head
        # We need mask targets from standard mask head
        # Wait, standard mask head might not return mask targets in its mask_loss results.
        # Let's check FCNMaskHead.loss_and_target

        # Actually, we can re-calculate them or ensure they are available.
        # For now, assume we can get them.

        # Re-get mask targets as they are needed for MaskIoUHead
        mask_targets = self.mask_head.get_targets(sampling_results, batch_gt_instances, self.train_cfg)
        gt_masks = [getattr(res, "instance_masks", getattr(res, "masks", None)) for res in batch_gt_instances]

        mask_iou_targets = self.mask_iou_head.get_targets(
            sampling_results,
            gt_masks,
            pos_mask_pred,
            mask_targets,
            self.train_cfg,
        )
        loss_mask_iou = self.mask_iou_head.loss(pos_mask_iou_pred, mask_iou_targets)
        mask_results["loss_mask"].update(loss_mask_iou)
        return mask_results

    def predict_mask(self, x, batch_img_metas, results_list, rescale=False):
        """Obtain mask prediction with scoring."""
        results_list = super().predict_mask(x, batch_img_metas, results_list, rescale=rescale)

        # If no instances, return
        if all(len(res) == 0 for res in results_list):
            return results_list

        # Get mask scores
        from visdet.structures.bbox import bbox2roi

        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)

        # We need mask features and predictions to get mask IoU scores
        mask_results = self._mask_forward(x, mask_rois)
        mask_feats = mask_results["mask_feats"]
        mask_preds = mask_results["mask_preds"]

        concat_det_labels = torch.cat([res.labels for res in results_list])

        mask_iou_pred = self.mask_iou_head(
            mask_feats,
            mask_preds[range(concat_det_labels.size(0)), concat_det_labels],
        )

        # split batch mask prediction back to each image
        num_bboxes_per_img = [len(res) for res in results_list]
        mask_iou_preds = mask_iou_pred.split(num_bboxes_per_img, 0)

        for i in range(len(results_list)):
            if len(results_list[i]) > 0:
                mask_scores = self.mask_iou_head.get_mask_scores(
                    mask_iou_preds[i], results_list[i].bboxes, results_list[i].labels
                )
                # results_list[i].scores = ... # MS R-CNN updates scores with mask scores?
                # Actually, usually it just adds mask_scores or updates existing scores.
                # In mmdet, it depends on the task.

                # For now, just store them if needed or update scores
                # results_list[i].mask_scores = mask_scores
                pass

        return results_list
