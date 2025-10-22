# Copyright (c) OpenMMLab. All rights reserved.

import torch
from torch import Tensor
from torchvision.ops import nms as torch_nms


def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    nms_cfg: dict | None,
    class_agnostic: bool = False,
) -> tuple[Tensor, Tensor]:
    r"""Performs non-maximum suppression in a batched fashion.

    Modified from `torchvision/ops/boxes.py#L39
    <https://github.com/pytorch/vision/blob/
    505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39>`_.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Note:
        In v1.4.1 and later, ``batched_nms`` supports skipping the NMS and
        returns sorted raw results when `nms_cfg` is None.

    Args:
        boxes (torch.Tensor): boxes in shape (N, 4) or (N, 5).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict | optional): Supports skipping the nms when `nms_cfg`
            is None, otherwise it should specify nms type and other
            parameters like `iou_thr`. Possible keys includes the following.

            - iou_threshold (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
              number of boxes is large (e.g., 200k). To avoid OOM during
              training, the users could set `split_thr` to a small value.
              If the number of boxes is greater than the threshold, it will
              perform NMS on each group of boxes separately and sequentially.
              Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class. Defaults to False.

    Returns:
        tuple: kept dets and indice.

        - boxes (Tensor): Bboxes with score after nms, has shape
          (num_bboxes, 5). last dimension 5 arrange as
          (x1, y1, x2, y2, score)
        - keep (Tensor): The indices of remaining boxes in input
          boxes.
    """
    # skip nms when nms_cfg is None
    if nms_cfg is None:
        scores, inds = scores.sort(descending=True)
        boxes = boxes[inds]
        return torch.cat([boxes, scores[:, None]], -1), inds

    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop("class_agnostic", class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        # When using rotated boxes, only apply offsets on center.
        if boxes.size(-1) == 5:
            # Strictly, the maximum coordinates of the rotating box
            # (x,y,w,h,a) should be calculated by polygon coordinates.
            # But the conversion from rotated box to polygon will
            # slow down the speed.
            # So we use max(x,y) + max(w,h) as max coordinate
            # which is larger than polygon max coordinate
            # max(x1, y1, x2, y2,x3, y3, x4, y4)
            max_coordinate = boxes[..., :2].max() + boxes[..., 2:4].max()
            offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
            boxes_ctr_for_nms = boxes[..., :2] + offsets[:, None]
            boxes_for_nms = torch.cat([boxes_ctr_for_nms, boxes[..., 2:5]], dim=-1)
        else:
            max_coordinate = boxes.max()
            offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
            boxes_for_nms = boxes + offsets[:, None]

    nms_op = nms_cfg_.pop("type", "nms")
    if isinstance(nms_op, str):
        nms_op = eval(nms_op)

    split_thr = nms_cfg_.pop("split_thr", 10000)
    # Won't split to multiple nms nodes when exporting to onnx
    if boxes_for_nms.shape[0] < split_thr:
        # Extract parameters for torchvision nms
        iou_threshold = nms_cfg_.pop("iou_threshold", nms_cfg_.pop("iou_thr", 0.5))

        # Use torchvision's nms instead of custom implementation
        keep = torch_nms(boxes_for_nms, scores, iou_threshold)

        # Apply max_num if specified
        max_num = nms_cfg_.get("max_num", -1)
        if max_num > 0 and keep.shape[0] > max_num:
            keep = keep[:max_num]

        boxes = boxes[keep]
        scores = scores[keep]
    else:
        max_num = nms_cfg_.pop("max_num", -1)
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        # Some type of nms would reweight the score, such as SoftNMS
        scores_after_nms = scores.new_zeros(scores.size())
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            # Extract parameters for torchvision nms
            iou_threshold = nms_cfg_.pop("iou_threshold", nms_cfg_.pop("iou_thr", 0.5))

            # Use torchvision's nms instead of custom implementation
            keep = torch_nms(boxes_for_nms[mask], scores[mask], iou_threshold)

            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = scores[mask[keep]]
        keep = total_mask.nonzero(as_tuple=False).view(-1)

        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

        if max_num > 0:
            keep = keep[:max_num]
            boxes = boxes[:max_num]
            scores = scores[:max_num]

    boxes = torch.cat([boxes, scores[:, None]], -1)
    return boxes, keep


def nms(
    boxes: Tensor,
    scores: Tensor,
    iou_threshold: float,
    offset: int = 0,
    score_threshold: float = 0,
    max_num: int = -1,
) -> tuple[Tensor, Tensor]:
    """Dispatch to torchvision NMS implementation.

    The input can be either torch tensor. This implementation uses
    torchvision's NMS which is optimized and runs on GPU if available.

    Arguments:
        boxes (torch.Tensor): boxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        iou_threshold (float): IoU threshold for NMS.
        offset (int, 0 or 1): boxes' width or height is (x2 - x1 + offset).
        score_threshold (float): score threshold for NMS.
        max_num (int): maximum number of boxes after NMS.

    Returns:
        tuple: kept dets (boxes and scores) and indice.

    Example:
        >>> boxes = torch.tensor([[49.1, 32.4, 51.0, 35.9],
        >>>                       [49.3, 32.9, 51.0, 35.3],
        >>>                       [49.2, 31.8, 51.0, 35.4],
        >>>                       [35.1, 11.5, 39.1, 15.7],
        >>>                       [35.6, 11.8, 39.3, 14.2],
        >>>                       [35.3, 11.5, 39.9, 14.5],
        >>>                       [35.2, 11.7, 39.7, 15.7]], dtype=torch.float32)
        >>> scores = torch.tensor([0.9, 0.9, 0.5, 0.5, 0.5, 0.4, 0.3],
        >>>                       dtype=torch.float32)
        >>> iou_threshold = 0.6
        >>> dets, inds = nms(boxes, scores, iou_threshold)
        >>> assert len(inds) == len(dets) == 3
    """
    assert isinstance(boxes, Tensor)
    assert isinstance(scores, Tensor)
    assert boxes.size(1) == 4
    assert boxes.size(0) == scores.size(0)
    assert offset in (0, 1)

    # Filter by score threshold if needed
    valid_inds = None
    if score_threshold > 0:
        valid_mask = scores > score_threshold
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        valid_inds = torch.nonzero(valid_mask, as_tuple=False).squeeze(dim=1)

    # Apply offset if needed (mmcv compatibility)
    if offset == 1:
        boxes_for_nms = boxes.clone()
        boxes_for_nms[:, 2:] += offset
    else:
        boxes_for_nms = boxes

    # Use torchvision's NMS
    keep = torch_nms(boxes_for_nms, scores, iou_threshold)

    # Apply max_num constraint
    if max_num > 0 and keep.shape[0] > max_num:
        keep = keep[:max_num]

    # Map back to original indices if we filtered by score
    if score_threshold > 0 and valid_inds is not None:
        keep = valid_inds[keep]

    dets = torch.cat((boxes[keep], scores[keep].reshape(-1, 1)), dim=1)
    return dets, keep
