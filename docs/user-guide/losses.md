# Losses

visdet uses a registry-based system for losses. In configs, you typically specify a loss by its class name:

```python
loss_cls = dict(type="FocalLoss", gamma=2.0, alpha=0.25, loss_weight=1.0)
loss_bbox = dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0)
```

All losses follow a common API with `reduction`, `loss_weight`, and optional `weight` and `avg_factor` parameters for flexible loss computation.

## Classification Losses

### CrossEntropyLoss

Standard cross-entropy loss for classification tasks.[^ce]

```
loss = -sum(target * log(softmax(pred)))
```

With `use_sigmoid=True`, it uses binary cross-entropy instead:

```
loss = -sum(target * log(sigmoid(pred)) + (1 - target) * log(1 - sigmoid(pred)))
```

**Key Hyperparameters:**

- `use_sigmoid` (default: False): Use sigmoid instead of softmax
- `use_mask` (default: False): Use mask cross-entropy for instance segmentation
- `class_weight` (optional): Per-class weights for imbalanced datasets
- `ignore_index` (default: -100): Label index to ignore in loss computation
- `avg_non_ignore` (default: False): Average only over non-ignored elements

**Characteristics:**

- Standard choice for multi-class classification
- Works well when classes are relatively balanced
- Can handle class imbalance with `class_weight`
- Supports mask predictions for instance segmentation

```python
loss_cls = dict(
    type="CrossEntropyLoss",
    use_sigmoid=False,
    loss_weight=1.0,
)
```

### FocalLoss

Focal Loss addresses class imbalance by down-weighting easy examples and focusing on hard negatives.[^focal]

```
p_t = sigmoid(pred) if target == 1 else (1 - sigmoid(pred))
focal_weight = alpha * (1 - p_t)^gamma
loss = -focal_weight * log(p_t)
```

**Key Hyperparameters:**

- `gamma` (default: 2.0): Focusing parameter that reduces loss for well-classified examples
- `alpha` (default: 0.25): Balancing factor for positive vs negative samples
- `use_sigmoid` (default: True): Only sigmoid mode is supported

**Characteristics:**

- Designed for extreme class imbalance (e.g., object detection with many background samples)
- `gamma=0` reduces to standard cross-entropy
- Higher `gamma` values focus more on hard examples
- Standard classification loss for one-stage detectors like RetinaNet

```python
loss_cls = dict(
    type="FocalLoss",
    gamma=2.0,
    alpha=0.25,
    loss_weight=1.0,
)
```

### QualityFocalLoss (GFL)

Quality Focal Loss extends Focal Loss to jointly represent classification and localization quality.[^gfl]

```
# For negatives (background):
loss = BCE(pred, 0) * sigmoid(pred)^beta

# For positives:
scale_factor = |quality_score - sigmoid(pred)|
loss = BCE(pred, quality_score) * scale_factor^beta
```

**Key Hyperparameters:**

- `beta` (default: 2.0): Modulating factor similar to gamma in Focal Loss
- `use_sigmoid` (default: True): Only sigmoid mode is supported

**Characteristics:**

- Target is a tuple of (category_label, quality_score) where quality_score is typically IoU
- Unifies classification and localization quality into a single prediction
- Eliminates the need for separate centerness prediction
- Used in GFL (Generalized Focal Loss) detector

```python
loss_cls = dict(
    type="QualityFocalLoss",
    beta=2.0,
    loss_weight=1.0,
)
```

---

## Regression Losses

### L1Loss

Simple L1 (Mean Absolute Error) loss for regression tasks.

```
loss = |pred - target|
```

**Key Hyperparameters:**

- `reduction` (default: "mean"): Options are "none", "mean", "sum"
- `loss_weight` (default: 1.0): Weight multiplier for the loss

**Characteristics:**

- More robust to outliers than L2 loss
- Gradient is constant regardless of error magnitude
- Can cause instability when errors are very small

```python
loss_bbox = dict(
    type="L1Loss",
    loss_weight=1.0,
)
```

### SmoothL1Loss

Smooth L1 loss (Huber loss) combines L1 and L2 to get benefits of both.[^smoothl1]

```
if |pred - target| < beta:
    loss = 0.5 * (pred - target)^2 / beta
else:
    loss = |pred - target| - 0.5 * beta
```

**Key Hyperparameters:**

- `beta` (default: 1.0): Threshold between L2 and L1 behavior
- `reduction` (default: "mean"): Options are "none", "mean", "sum"

**Characteristics:**

- Smooth gradient near zero (like L2) prevents instability
- Linear for large errors (like L1) for robustness to outliers
- Standard regression loss for two-stage detectors like Faster R-CNN
- `beta` controls the transition point between quadratic and linear

```python
loss_bbox = dict(
    type="SmoothL1Loss",
    beta=1.0,
    loss_weight=1.0,
)
```

### MSELoss

Mean Squared Error (L2) loss for regression.

```
loss = (pred - target)^2
```

**Key Hyperparameters:**

- `reduction` (default: "mean"): Options are "none", "mean", "sum"
- `loss_weight` (default: 1.0): Weight multiplier for the loss

**Characteristics:**

- Penalizes large errors more heavily than L1
- Sensitive to outliers
- Smooth gradient everywhere
- Commonly used for centerness prediction

```python
loss_centerness = dict(
    type="MSELoss",
    loss_weight=1.0,
)
```

### BalancedL1Loss

Balanced L1 Loss from Libra R-CNN promotes balanced learning between classification and localization.[^libra]

```
b = e^(gamma/alpha) - 1
if |diff| < beta:
    loss = (alpha/b) * (b*diff + 1) * log(b*diff/beta + 1) - alpha*diff
else:
    loss = gamma*diff + gamma/b - alpha*beta
```

**Key Hyperparameters:**

- `alpha` (default: 0.5): Controls the upper bound of the loss
- `gamma` (default: 1.5): Controls the gradient at the origin
- `beta` (default: 1.0): Threshold between regions (like Smooth L1)

**Characteristics:**

- Designed to balance the contribution of samples at different levels
- Promotes equal training for samples with different IoU values
- Improves localization accuracy especially for high-IoU samples
- More complex but can provide better accuracy than SmoothL1Loss

```python
loss_bbox = dict(
    type="BalancedL1Loss",
    alpha=0.5,
    gamma=1.5,
    beta=1.0,
    loss_weight=1.0,
)
```

---

## IoU-Based Losses

### IoULoss

IoU Loss directly optimizes Intersection over Union for bounding box regression.[^iou]

```
iou = intersection(pred, target) / union(pred, target)

if mode == "linear":
    loss = 1 - iou
elif mode == "square":
    loss = 1 - iou^2
elif mode == "log":
    loss = -log(iou)
```

**Key Hyperparameters:**

- `mode` (default: "log"): Loss scaling mode - "linear", "square", or "log"
- `eps` (default: 1e-6): Small value for numerical stability

**Characteristics:**

- Scale-invariant: treats small and large boxes equally
- Directly optimizes the evaluation metric
- Log mode provides larger gradients for low IoU predictions
- Used in anchor-free detectors like FCOS

```python
loss_bbox = dict(
    type="IoULoss",
    mode="log",
    loss_weight=1.0,
)
```

### GIoULoss

Generalized IoU Loss extends IoU to handle non-overlapping boxes.[^giou]

```
# C is the smallest enclosing box containing both pred and target
giou = iou - (area(C) - union) / area(C)
loss = 1 - giou
```

**Key Hyperparameters:**

- `eps` (default: 1e-6): Small value for numerical stability
- `reduction` (default: "mean"): Options are "none", "mean", "sum"

**Characteristics:**

- Works even when boxes don't overlap (IoU = 0)
- Provides gradient signal for non-overlapping boxes
- GIoU ranges from -1 to 1, where 1 is perfect overlap
- Better convergence than IoU loss for boxes that are far apart
- Standard regression loss for modern detectors like ATSS

```python
loss_bbox = dict(
    type="GIoULoss",
    loss_weight=2.0,
)
```

---

## Distribution Losses

### DistributionFocalLoss

Distribution Focal Loss learns a discretized distribution over box offsets instead of direct regression.[^gfl]

```
# Label is a continuous value, discretized to neighboring integers
left = floor(label)
right = left + 1
weight_left = right - label
weight_right = label - left

loss = CE(pred, left) * weight_left + CE(pred, right) * weight_right
```

**Key Hyperparameters:**

- `reduction` (default: "mean"): Options are "none", "mean", "sum"
- `loss_weight` (default: 1.0): Weight multiplier for the loss

**Characteristics:**

- Models localization uncertainty as a distribution
- Allows the network to express ambiguous boundary locations
- Works with General Distribution representation for bbox regression
- Used together with QualityFocalLoss in GFL detector

```python
loss_dfl = dict(
    type="DistributionFocalLoss",
    loss_weight=0.25,
)
```

---

## Loss Combinations for Popular Detectors

### RetinaNet

RetinaNet uses Focal Loss for classification to handle extreme class imbalance, paired with Smooth L1 for box regression.[^focal]

```python
loss_cls = dict(
    type="FocalLoss",
    use_sigmoid=True,
    gamma=2.0,
    alpha=0.25,
    loss_weight=1.0,
)
loss_bbox = dict(
    type="SmoothL1Loss",
    beta=1.0,
    loss_weight=1.0,
)
```

### FCOS

FCOS (Fully Convolutional One-Stage) uses Focal Loss for classification, IoU Loss for box regression, and CrossEntropyLoss for centerness.[^fcos]

```python
loss_cls = dict(
    type="FocalLoss",
    use_sigmoid=True,
    gamma=2.0,
    alpha=0.25,
    loss_weight=1.0,
)
loss_bbox = dict(
    type="IoULoss",
    mode="log",
    loss_weight=1.0,
)
loss_centerness = dict(
    type="CrossEntropyLoss",
    use_sigmoid=True,
    loss_weight=1.0,
)
```

### ATSS

ATSS (Adaptive Training Sample Selection) uses Focal Loss for classification, GIoU Loss for better box regression, and CrossEntropyLoss for centerness.[^atss]

```python
loss_cls = dict(
    type="FocalLoss",
    use_sigmoid=True,
    gamma=2.0,
    alpha=0.25,
    loss_weight=1.0,
)
loss_bbox = dict(
    type="GIoULoss",
    loss_weight=2.0,
)
loss_centerness = dict(
    type="CrossEntropyLoss",
    use_sigmoid=True,
    loss_weight=1.0,
)
```

### GFL (Generalized Focal Loss)

GFL unifies classification and localization quality with QualityFocalLoss, and uses DistributionFocalLoss for learning box distributions.[^gfl]

```python
loss_cls = dict(
    type="QualityFocalLoss",
    use_sigmoid=True,
    beta=2.0,
    loss_weight=1.0,
)
loss_bbox = dict(
    type="GIoULoss",
    loss_weight=2.0,
)
loss_dfl = dict(
    type="DistributionFocalLoss",
    loss_weight=0.25,
)
```

---

## Listing Available Losses

Because the available set can change depending on installed optional dependencies, you can list what your environment has registered:

```python
from visdet.registry import MODELS

# Filter for loss modules
losses = [name for name in sorted(MODELS.module_dict.keys()) if "Loss" in name]
print(losses)
```

!!! note
    All losses inherit from `nn.Module` and follow the same forward signature: `forward(pred, target, weight=None, avg_factor=None, reduction_override=None)`.

[^ce]: Murphy, K. (2012), *Machine Learning: A Probabilistic Perspective*. MIT Press.
[^focal]: Lin et al. (2017), *Focal Loss for Dense Object Detection*. https://arxiv.org/abs/1708.02002
[^gfl]: Li et al. (2020), *Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection*. https://arxiv.org/abs/2006.04388
[^smoothl1]: Girshick (2015), *Fast R-CNN*. https://arxiv.org/abs/1504.08083
[^libra]: Pang et al. (2019), *Libra R-CNN: Towards Balanced Learning for Object Detection*. https://arxiv.org/abs/1904.02701
[^iou]: Yu et al. (2016), *UnitBox: An Advanced Object Detection Network*. https://arxiv.org/abs/1608.01471
[^giou]: Rezatofighi et al. (2019), *Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression*. https://arxiv.org/abs/1902.09630
[^fcos]: Tian et al. (2019), *FCOS: Fully Convolutional One-Stage Object Detection*. https://arxiv.org/abs/1904.01355
[^atss]: Zhang et al. (2020), *Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection*. https://arxiv.org/abs/1912.02424
