# Threshold Optimization for F-beta Metrics

## Overview

One of the most critical but often overlooked aspects of deploying object detection models is **threshold optimization**. While training focuses on learning good representations, production deployment requires carefully tuning confidence thresholds to maximize task-specific metrics like F-beta scores.

Surprisingly, there's a significant gap in the ecosystem: **very few tools exist to help practitioners systematically optimize detection thresholds for F-beta or other metrics**. This guide addresses that gap and provides practical approaches to threshold optimization.

## The Problem

### Why Threshold Optimization Matters

Object detection models output confidence scores for each detection, but the optimal threshold depends on your specific use case:

- **High Recall Applications** (e.g., medical screening): You want to catch every possible case, accepting more false positives. Use **F₂ or F₀.₅** scores where recall is weighted higher.
- **High Precision Applications** (e.g., automated actions): You only want to act on high-confidence detections. Use **F₀.₅** where precision is weighted higher.
- **Balanced Applications**: Standard **F₁** score works well for general use cases.

### The Tooling Gap

Despite the importance of threshold optimization, the detection ecosystem lacks good tools for this:

**What's Missing:**
- No built-in threshold optimization in major frameworks (MMDetection, Detectron2, YOLO)
- Classification libraries (scikit-learn) have `precision_recall_curve` but detection is multi-class
- COCO metrics focus on AP at fixed IoU thresholds, not confidence thresholds
- Most tutorials stop at "pick a threshold of 0.5" which is rarely optimal

**Why This is Weird:**
- Binary classification has had tools like `sklearn.metrics.precision_recall_curve` for decades
- Threshold optimization is a well-studied problem in binary classification
- The detection community has somehow not adapted these tools effectively

## F-beta Score Primer

The F-beta score is a weighted harmonic mean of precision and recall:

$$
F_\beta = (1 + \beta^2) \cdot \frac{\text{precision} \cdot \text{recall}}{\beta^2 \cdot \text{precision} + \text{recall}}
$$

**Key β values:**
- **β = 1**: F₁ score - balanced precision and recall
- **β = 2**: F₂ score - weighs recall 2× more than precision (catch more positives)
- **β = 0.5**: F₀.₅ score - weighs precision 2× more than recall (fewer false alarms)

## Practical Threshold Optimization

### Step 1: Collect Predictions with Scores

First, run inference on your validation set and save **all** predictions with their confidence scores (don't apply a threshold yet):

```python
from visdet import SimpleRunner
import pickle

# Load trained model
runner = SimpleRunner.from_checkpoint('work_dirs/my_model/best.pth')

# Get predictions with scores
results = []
for img_path in val_images:
    preds = runner.predict(img_path, score_thr=0.0)  # Keep all predictions!
    results.append({
        'img_path': img_path,
        'boxes': preds.pred_instances.bboxes,
        'scores': preds.pred_instances.scores,
        'labels': preds.pred_instances.labels
    })

# Save for analysis
with open('val_predictions.pkl', 'wb') as f:
    pickle.dump(results, f)
```

### Step 2: Compute F-beta Across Thresholds

Now systematically evaluate F-beta scores across different confidence thresholds:

```python
import numpy as np
from typing import List, Tuple

def compute_fbeta_at_threshold(
    predictions: List[dict],
    ground_truths: List[dict],
    threshold: float,
    beta: float = 1.0,
    iou_threshold: float = 0.5
) -> Tuple[float, float, float]:
    """
    Compute F-beta score at a specific confidence threshold.

    Args:
        predictions: List of prediction dicts with boxes, scores, labels
        ground_truths: List of ground truth dicts with boxes, labels
        threshold: Confidence threshold to apply
        beta: Beta value for F-beta score (1.0 = F1, 2.0 = F2, 0.5 = F0.5)
        iou_threshold: IoU threshold for matching predictions to ground truth

    Returns:
        (fbeta, precision, recall)
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred, gt in zip(predictions, ground_truths):
        # Filter predictions by confidence threshold
        mask = pred['scores'] >= threshold
        pred_boxes = pred['boxes'][mask]
        pred_labels = pred['labels'][mask]

        gt_boxes = gt['boxes']
        gt_labels = gt['labels']

        # Match predictions to ground truth (simplified - real implementation needs IoU matching)
        matched_gt = set()

        for pred_box, pred_label in zip(pred_boxes, pred_labels):
            # Find best matching ground truth
            best_iou = 0
            best_idx = -1

            for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if pred_label != gt_label:
                    continue

                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= iou_threshold and best_idx not in matched_gt:
                true_positives += 1
                matched_gt.add(best_idx)
            else:
                false_positives += 1

        # Unmatched ground truths are false negatives
        false_negatives += len(gt_boxes) - len(matched_gt)

    # Compute metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    if precision + recall == 0:
        return 0.0, precision, recall

    fbeta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

    return fbeta, precision, recall


def optimize_threshold_for_fbeta(
    predictions: List[dict],
    ground_truths: List[dict],
    beta: float = 1.0,
    iou_threshold: float = 0.5,
    num_thresholds: int = 100
) -> Tuple[float, float, float, float]:
    """
    Find optimal confidence threshold for F-beta score.

    Returns:
        (optimal_threshold, best_fbeta, precision, recall)
    """
    thresholds = np.linspace(0, 1, num_thresholds)

    best_fbeta = 0
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0

    results = []

    for threshold in thresholds:
        fbeta, precision, recall = compute_fbeta_at_threshold(
            predictions, ground_truths, threshold, beta, iou_threshold
        )

        results.append({
            'threshold': threshold,
            'fbeta': fbeta,
            'precision': precision,
            'recall': recall
        })

        if fbeta > best_fbeta:
            best_fbeta = fbeta
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    return best_threshold, best_fbeta, best_precision, best_recall, results


# Example usage
optimal_threshold, best_f1, prec, rec, all_results = optimize_threshold_for_fbeta(
    predictions,
    ground_truths,
    beta=1.0  # F1 score
)

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"F1 score: {best_f1:.3f}")
print(f"Precision: {prec:.3f}, Recall: {rec:.3f}")
```

### Step 3: Visualize Precision-Recall Trade-off

Understanding the precision-recall trade-off at different thresholds is crucial:

```python
import matplotlib.pyplot as plt

def plot_threshold_analysis(results: List[dict], beta: float = 1.0):
    """Visualize how metrics change with threshold."""
    thresholds = [r['threshold'] for r in results]
    fbeta_scores = [r['fbeta'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Metrics vs Threshold
    ax1.plot(thresholds, fbeta_scores, 'b-', linewidth=2, label=f'F{beta}')
    ax1.plot(thresholds, precisions, 'r--', label='Precision')
    ax1.plot(thresholds, recalls, 'g--', label='Recall')
    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Metrics vs Confidence Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mark optimal threshold
    best_idx = np.argmax(fbeta_scores)
    ax1.axvline(thresholds[best_idx], color='k', linestyle=':',
                label=f'Optimal: {thresholds[best_idx]:.2f}')

    # Plot 2: Precision-Recall Curve
    ax2.plot(recalls, precisions, 'b-', linewidth=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.grid(True, alpha=0.3)

    # Mark optimal point
    ax2.plot(recalls[best_idx], precisions[best_idx], 'r*',
             markersize=15, label='Optimal')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('threshold_optimization.png', dpi=150)
    print(f"Saved visualization to threshold_optimization.png")

plot_threshold_analysis(all_results, beta=1.0)
```

### Step 4: Per-Class Threshold Optimization

For multi-class detection, you may want different thresholds per class:

```python
def optimize_per_class_thresholds(
    predictions: List[dict],
    ground_truths: List[dict],
    num_classes: int,
    beta: float = 1.0
) -> dict:
    """Find optimal threshold for each class separately."""

    optimal_thresholds = {}

    for class_id in range(num_classes):
        # Filter to single class
        class_preds = []
        class_gts = []

        for pred, gt in zip(predictions, ground_truths):
            class_pred_mask = pred['labels'] == class_id
            class_gt_mask = gt['labels'] == class_id

            class_preds.append({
                'boxes': pred['boxes'][class_pred_mask],
                'scores': pred['scores'][class_pred_mask],
                'labels': pred['labels'][class_pred_mask]
            })

            class_gts.append({
                'boxes': gt['boxes'][class_gt_mask],
                'labels': gt['labels'][class_gt_mask]
            })

        # Optimize for this class
        threshold, fbeta, prec, rec, _ = optimize_threshold_for_fbeta(
            class_preds, class_gts, beta=beta
        )

        optimal_thresholds[class_id] = {
            'threshold': threshold,
            'fbeta': fbeta,
            'precision': prec,
            'recall': rec
        }

    return optimal_thresholds

# Find per-class optimal thresholds
class_thresholds = optimize_per_class_thresholds(
    predictions, ground_truths, num_classes=69, beta=1.0
)

for class_id, metrics in class_thresholds.items():
    print(f"Class {class_id}: threshold={metrics['threshold']:.3f}, "
          f"F1={metrics['fbeta']:.3f}")
```

## Production Deployment

Once you've found optimal thresholds, apply them during inference:

```python
class OptimizedDetector:
    """Detector with optimized per-class thresholds."""

    def __init__(self, model, class_thresholds: dict):
        self.model = model
        self.class_thresholds = class_thresholds

    def predict(self, image):
        # Get all predictions (no threshold)
        preds = self.model.predict(image, score_thr=0.0)

        # Apply per-class thresholds
        keep_mask = np.zeros(len(preds.pred_instances.scores), dtype=bool)

        for class_id, threshold_info in self.class_thresholds.items():
            threshold = threshold_info['threshold']
            class_mask = (preds.pred_instances.labels == class_id) & \
                        (preds.pred_instances.scores >= threshold)
            keep_mask |= class_mask

        # Filter predictions
        preds.pred_instances.bboxes = preds.pred_instances.bboxes[keep_mask]
        preds.pred_instances.scores = preds.pred_instances.scores[keep_mask]
        preds.pred_instances.labels = preds.pred_instances.labels[keep_mask]

        return preds

# Use in production
detector = OptimizedDetector(model, class_thresholds)
results = detector.predict(test_image)
```

## Common Pitfalls

### 1. Optimizing on Training Set
**Never** optimize thresholds on the training set - always use a held-out validation set. Better yet, use a separate "threshold tuning" set distinct from your final test set.

### 2. Ignoring Class Imbalance
Rare classes may need different thresholds than common classes. Per-class optimization addresses this.

### 3. Using Wrong Beta Value
Choose β based on your application:
- **Medical/Safety**: β = 2 (prioritize recall)
- **Automated Actions**: β = 0.5 (prioritize precision)
- **General**: β = 1 (balanced)

### 4. Not Re-optimizing After Model Updates
Thresholds should be re-optimized whenever you retrain or fine-tune your model.

## Why Isn't This Built Into Frameworks?

This is genuinely puzzling. The classification community solved this problem decades ago, but object detection frameworks have not adopted similar tools. Possible reasons:

1. **Complexity**: Multi-class detection with IoU matching is more complex than binary classification
2. **COCO Metrics Dominance**: The field focuses on Average Precision which uses all thresholds
3. **Research vs Production Gap**: Most detection research cares about AP, not deployment metrics
4. **Fragmentation**: Every team builds their own solution instead of contributing to frameworks

This represents a real opportunity for the detection community to improve tooling.

## Future: Built-in Threshold Optimization

We're considering adding threshold optimization tools directly to visdet:

```python
# Future API (planned)
from visdet.evaluation import ThresholdOptimizer

optimizer = ThresholdOptimizer(
    model=model,
    val_dataset=val_dataset,
    beta=1.0,
    iou_threshold=0.5
)

# Automatically find optimal thresholds
results = optimizer.optimize()

# Apply optimized thresholds
model.set_thresholds(results.optimal_thresholds)
```

If you're interested in this functionality, please open an issue on our GitHub!

## References

- [Scikit-learn Precision-Recall Curves](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html) - Binary classification baseline
- [F-beta Score](https://en.wikipedia.org/wiki/F-score) - Wikipedia overview
- [COCO Detection Evaluation](https://cocodataset.org/#detection-eval) - Why AP is not enough for deployment

## Summary

**Key Takeaways:**
1. Threshold optimization is critical for production but under-tooled in detection
2. F-beta scores let you balance precision vs recall based on your application
3. Per-class thresholds often outperform single global thresholds
4. Always optimize on validation data, never training or test sets
5. The detection community needs better built-in tools for this

We hope this guide helps fill the gap until better framework-level tools exist. If you develop threshold optimization tools, please consider contributing them to open-source detection frameworks!
