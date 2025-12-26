# Assigners

In object detection training, an **assigner** is the component that decides *which predictions should learn from which ground-truth objects*.

Given a set of **candidates** (anchors, points, proposals, or transformer queries) and the **ground-truth** boxes/labels, the assigner produces a mapping that marks each candidate as:

- **positive** (assigned to a specific GT instance),
- **negative/background**, or
- **ignore** (excluded from loss computation).

This assignment is a critical part of the training target generation pipeline because it controls the positive/negative balance, influences which samples get gradients, and (for some detectors) defines the training objective itself (e.g. one-to-one matching in DETR-style models).

## Inputs and outputs

### Inputs
Different detectors feed different “candidate” types into the assigner:

- **Two-stage detectors** (RPN / RoI heads): anchors or proposals.
- **One-stage anchor-based detectors** (RetinaNet, SSD): anchors.
- **One-stage anchor-free detectors** (FCOS-style): points / priors.
- **Transformer-based detectors** (DETR / Deformable DETR): fixed number of queries.

In visdet, assigners are task utilities built from config dicts via `TASK_UTILS` (see `visdet/models/task_modules/builder.py:7`). Most modern assigners operate on `InstanceData` containers.

### Output: `AssignResult`
Assigners return an `AssignResult` (see `visdet/models/task_modules/assigners/assign_result.py:8`). The most important field is:

- `gt_inds`: for each candidate, the **1-based** index of the assigned GT box.
  - `gt_inds > 0`: positive (assigned to GT index `gt_inds - 1`)
  - `gt_inds == 0`: negative/background
  - `gt_inds == -1`: ignore

If class labels are provided, `labels` stores the corresponding class label per candidate.

## Where assigners appear in configs
Most models configure assigners under `train_cfg`. For example, a two-stage model typically defines one assigner for RPN and another for the RoI head:

```python
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type="MaxIoUAssigner",
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1,
        ),
        # sampler=...
    ),
    rcnn=dict(
        assigner=dict(
            type="MaxIoUAssigner",
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            ignore_iof_thr=-1,
        ),
        # sampler=...
    ),
)
```

## Assigner variants

!!! note "Implementation status"

    At the moment, visdet’s runtime exposes `MaxIoUAssigner` under `visdet.models.task_modules.assigners`.

    The repository also contains upstream reference implementations (used by the config zoo) under `archive/mmdet/core/bbox/assigners/`. The descriptions below cover the assigners you’ll see referenced in configs.

### `MaxIoUAssigner`
**Type:** IoU-threshold assignment (many-to-one)

Assigns each candidate to the GT with maximum IoU, using thresholds to decide positives/negatives, plus an optional “low-quality match” step that forces at least one positive per GT.

Common in:
- RPN
- RoI heads
- Anchor-based one-stage detectors

Key knobs:
- `pos_iou_thr`, `neg_iou_thr`, `min_pos_iou`
- `match_low_quality`, `gt_max_assign_all`
- `ignore_iof_thr` / `ignore_wrt_candidates` for ignored GT regions

See also: `visdet/models/task_modules/assigners/max_iou_assigner.py:83`.

### `ApproxMaxIoUAssigner`
**Type:** IoU-threshold assignment over *groups* of candidates

Variant of `MaxIoUAssigner` used in SABL-style setups where each “base” box has multiple approximations. The max IoU over a group drives assignment.

Key knobs (same spirit as MaxIoU):
- `pos_iou_thr`, `neg_iou_thr`, `min_pos_iou`, `match_low_quality`

Reference: `archive/mmdet/core/bbox/assigners/approx_max_iou_assigner.py:10`.

### `ATSSAssigner`
**Type:** adaptive sample selection (many-to-one)

Implements ATSS (“Adaptive Training Sample Selection”): per-FPN-level top-k candidates (by center distance) are selected for each GT, then an adaptive IoU threshold (mean + std) chooses positives.

Common in:
- ATSS
- VFNet
- DDOD (dynamic-cost variant)

Key knobs:
- `topk` (candidates per FPN level)
- `alpha` (optional) enables dynamic-cost mode used in DDOD

Reference: `archive/mmdet/core/bbox/assigners/atss_assigner.py:13`.

### `TaskAlignedAssigner`
**Type:** alignment-metric assignment (many-to-one)

Used in TOOD. Selects top-k candidates per GT based on an alignment metric combining classification confidence and IoU.

Key knobs:
- `topk`
- `alpha`, `beta` (weights in the alignment metric)

Reference: `archive/mmdet/core/bbox/assigners/task_aligned_assigner.py:13`.

### `HungarianAssigner`
**Type:** one-to-one matching (set prediction)

Performs global one-to-one assignment between predictions and GT using the Hungarian algorithm (linear sum assignment) on a weighted cost matrix.

Common in:
- DETR
- Deformable DETR
- QueryInst / Sparse R-CNN-style query matching

Key knobs (costs):
- `cls_cost`, `reg_cost`, `iou_cost`

Reference: `archive/mmdet/core/bbox/assigners/hungarian_assigner.py:13`.

### `MaskHungarianAssigner`
**Type:** one-to-one matching for masks

Hungarian matching where the costs are classification + mask focal cost + dice cost. Used for mask-query models.

Key knobs:
- `cls_cost`, `mask_cost`, `dice_cost`

Reference: `archive/mmdet/core/bbox/assigners/mask_hungarian_assigner.py:13`.

### `SimOTAAssigner`
**Type:** dynamic-k matching (many-to-one)

Implements SimOTA (used in YOLOX). Builds a cost matrix from classification and IoU costs, applies center/box constraints, then uses a dynamic-k strategy to determine how many positives each GT gets.

Key knobs:
- `center_radius`
- `candidate_topk` (for dynamic-k computation)
- `cls_weight`, `iou_weight`

Reference: `archive/mmdet/core/bbox/assigners/sim_ota_assigner.py:14`.

### `GridAssigner`
**Type:** IoU-threshold assignment with grid responsibility

Similar to `MaxIoUAssigner`, but only candidates marked as responsible for a particular grid cell can become positives. Used in YOLOv3-style training.

Key knobs:
- `pos_iou_thr`, `neg_iou_thr`, `min_pos_iou`
- `gt_max_assign_all`

Reference: `archive/mmdet/core/bbox/assigners/grid_assigner.py:11`.

### `UniformAssigner`
**Type:** uniform matching via L1 distance

Selects a fixed number of positives per GT (`match_times`) based on L1 distance between predicted boxes / anchors and GT boxes. Also supports ignoring ambiguous samples via overlap thresholds.

Key knobs:
- `match_times`
- `pos_ignore_thr`, `neg_ignore_thr`

Reference: `archive/mmdet/core/bbox/assigners/uniform_assigner.py:12`.

### `RegionAssigner`
**Type:** region-based assignment on feature maps

Assigns anchors as positive if their centers fall inside a GT “center region”, and ignores anchors in a larger “ignore region” (including adjacent FPN levels).

Key knobs:
- `center_ratio`, `ignore_ratio`

Reference: `archive/mmdet/core/bbox/assigners/region_assigner.py:39`.

### `CenterRegionAssigner`
**Type:** center-kernel assignment (FSAF-style)

Marks priors whose centers fall inside a scaled “core” region as positives, and treats an outer “shadow” ring specially (often ignored / loss-weighted). Prioritizes smaller GTs when overlaps occur.

Key knobs:
- `pos_scale`, `neg_scale`
- `min_pos_iof`, `ignore_gt_scale`
- `foreground_dominate`

Reference: `archive/mmdet/core/bbox/assigners/center_region_assigner.py:74`.

### `PointAssigner`
**Type:** point-to-GT assignment

Assigns points (with an associated stride/level) to GT boxes by selecting the closest `pos_num` points per GT at the appropriate feature level.

Key knobs:
- `scale` (controls which level a GT maps to)
- `pos_num` (positives per GT per level)

Reference: `archive/mmdet/core/bbox/assigners/point_assigner.py:10`.

### `AscendMaxIoUAssigner`
**Type:** batched MaxIoU assignment

A batched variant of MaxIoU assignment designed for Ascend/NPU execution, returning an `AscendAssignResult` with per-image positive/negative masks.

Reference: `archive/mmdet/core/bbox/assigners/ascend_max_iou_assigner.py:12`.
