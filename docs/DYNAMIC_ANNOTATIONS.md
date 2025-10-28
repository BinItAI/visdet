# Dynamic Annotation Files in SimpleRunner

## Overview

SimpleRunner now supports specifying training and validation annotation files directly through the `train_ann_file` and `val_ann_file` parameters. This feature enables seamless integration with ML pipelines where annotation files are generated on-the-fly from upstream data sources.

## Motivation: Why Dynamic Annotation Files Matter

In realistic ML pipelines, annotation files are rarely static. They're typically generated dynamically from various upstream sources:

### Common Real-World Scenarios

1. **MLflow Experiment Tracking**
   - Annotation files downloaded from MLflow artifact store for specific experiment runs
   - Each experiment gets versioned annotation files
   - Easy to track which annotations were used for each model

2. **Apache Airflow Data Pipelines**
   - Daily data processing jobs generate updated annotation files
   - Training automatically pulls latest available files
   - Natural integration with enterprise data workflows

3. **Cross-Validation Workflows**
   - Generate K different train/val splits programmatically for robust model evaluation
   - Iterate over folds without manual file management
   - No need to create separate preset files for each fold

4. **A/B Testing and Experimentation**
   - Generate different data distributions on-the-fly
   - Compare model performance across variations
   - Configuration-driven experiment definitions

5. **CI/CD Integration Testing**
   - Generate small synthetic datasets for fast test runs
   - Validate pipeline before committing
   - Avoid slow tests on full production datasets

6. **DVC (Data Version Control)**
   - Pull versioned datasets based on git tags or experiment IDs
   - Annotation files automatically updated with data versions
   - Full reproducibility of training runs

### Why This Matters

**Without dynamic annotation file support**, users must:
- Create separate dataset preset YAML files for each annotation file (not scalable)
- Use verbose dict overrides with `_base_` (obscures intent)
- Modify preset files in-place (anti-pattern, breaks version control)
- Use string manipulation to construct file paths (error-prone)

**With dynamic annotation file support**:
- Clean, explicit API for specifying files
- IDE autocomplete and type hints
- Integrates naturally with Python ML frameworks (MLflow, Airflow, etc.)
- Early validation catches configuration errors

## API Reference

### `SimpleRunner.__init__(..., train_ann_file=None, val_ann_file=None)`

#### Parameters

- **`train_ann_file`** (`Optional[str]`, default: `None`)
  - Path to training annotation file in COCO format
  - Overrides `ann_file` in dataset config if provided
  - Can be absolute or relative path
  - File existence is validated before training

- **`val_ann_file`** (`Optional[str]`, default: `None`)
  - Path to validation annotation file in COCO format
  - Overrides `val_ann_file` in dataset config if provided
  - Enables validation even if dataset preset doesn't define one
  - Can be absolute or relative path
  - File existence is validated before training

#### Behavior

- **Initialization**: When SimpleRunner is created, annotation files are validated to exist
- **Priority**: Explicit parameters override dataset preset definitions
- **Backward Compatible**: Existing code without these parameters works unchanged
- **DDP Safe**: Parameters are stored for worker recreation in distributed training

#### Returns

Returns a SimpleRunner instance with configurations ready for training.

#### Raises

- **`FileNotFoundError`**: If specified annotation file doesn't exist
  - Caught immediately during initialization
  - Error message includes both provided and resolved paths
  - Helps catch configuration errors before training starts

## Usage Examples

### Example 1: Basic Dynamic Annotation File

```python
from visdet import SimpleRunner

# Specify annotation files explicitly
runner = SimpleRunner(
    model='mask_rcnn_swin_s',
    dataset='cmr_instance_segmentation',  # Use preset for pipeline/metainfo
    train_ann_file='/path/to/train.json',  # Dynamic path
    val_ann_file='/path/to/val.json',      # Dynamic path
    epochs=12
)
runner.train()
```

### Example 2: MLflow Integration

```python
import mlflow
from visdet import SimpleRunner

# Get latest experiment artifacts from MLflow
client = mlflow.tracking.MlflowClient()
experiment_id = mlflow.get_experiment_by_name("mask_rcnn_experiments").experiment_id

# Search for latest run
runs = client.search_runs(experiment_ids=[experiment_id])
latest_run = runs[0]

# Download annotation files from artifact store
train_ann = client.download_artifacts(
    latest_run.info.run_id,
    'data/train_annotations.json'
)
val_ann = client.download_artifacts(
    latest_run.info.run_id,
    'data/val_annotations.json'
)

# Train with artifacts from MLflow
runner = SimpleRunner(
    model='mask_rcnn_swin_s',
    dataset='cmr_instance_segmentation',
    train_ann_file=train_ann,
    val_ann_file=val_ann,
    epochs=12,
    work_dir='/tmp/experiment_run'
)
runner.train()
```

### Example 3: Cross-Validation Loop

```python
from pathlib import Path
from sklearn.model_selection import KFold
from visdet import SimpleRunner

# Assume you have generated K fold splits in a directory
folds_dir = Path('/data/cv_folds')

# Iterate over folds
for fold_idx in range(5):
    train_ann = folds_dir / f'fold_{fold_idx}_train.json'
    val_ann = folds_dir / f'fold_{fold_idx}_val.json'

    print(f"\nTraining Fold {fold_idx + 1}/5")

    runner = SimpleRunner(
        model='mask_rcnn_swin_s',
        dataset='cmr_instance_segmentation',
        train_ann_file=str(train_ann),
        val_ann_file=str(val_ann),
        epochs=12,
        work_dir=f'./work_dirs/fold_{fold_idx}'
    )
    runner.train()

    # Evaluate results per fold...
```

### Example 4: A/B Testing

```python
from visdet import SimpleRunner
from pathlib import Path

# Define experiment variants
experiments = {
    'baseline': '/data/experiments/baseline/train.json',
    'augmented': '/data/experiments/augmented/train.json',
    'balanced': '/data/experiments/balanced/train.json',
}

results = {}

for variant_name, train_ann in experiments.items():
    val_ann = train_ann.replace('train.json', 'val.json')

    print(f"\nTraining {variant_name.upper()} variant...")

    runner = SimpleRunner(
        model='mask_rcnn_swin_s',
        dataset='cmr_instance_segmentation',
        train_ann_file=train_ann,
        val_ann_file=val_ann,
        epochs=12,
        work_dir=f'./work_dirs/{variant_name}'
    )
    runner.train()

    # Log results for comparison
    results[variant_name] = {
        'train_ann': train_ann,
        'val_ann': val_ann,
        # ... model metrics ...
    }
```

### Example 5: CI/CD Integration Testing

```python
import json
import tempfile
from pathlib import Path
from visdet import SimpleRunner

def create_minimal_coco_annotation(output_path: Path, num_images: int = 5):
    """Create minimal COCO annotation for fast CI testing."""
    annotation = {
        "images": [
            {"id": i, "file_name": f"image_{i}.jpg"}
            for i in range(num_images)
        ],
        "annotations": [
            {
                "id": i,
                "image_id": i,
                "category_id": 1,
                "bbox": [0, 0, 100, 100],
                "area": 10000,
                "iscrowd": 0
            }
            for i in range(num_images)
        ],
        "categories": [{"id": 1, "name": "object"}]
    }
    output_path.write_text(json.dumps(annotation))

# In CI/CD pipeline: generate minimal synthetic data
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)

    train_ann = tmpdir / "train.json"
    val_ann = tmpdir / "val.json"

    create_minimal_coco_annotation(train_ann, num_images=5)
    create_minimal_coco_annotation(val_ann, num_images=2)

    # Fast integration test with minimal data
    runner = SimpleRunner(
        model='mask_rcnn_swin_s',
        dataset='cmr_instance_segmentation',
        train_ann_file=str(train_ann),
        val_ann_file=str(val_ann),
        epochs=1,  # Just one epoch for testing
        work_dir=str(tmpdir / 'work_dirs')
    )

    # Validates pipeline works, runs quickly
    runner.train()
```

## Migration Guide

### From Dict Overrides to Dynamic Parameters

**Before (verbose):**
```python
runner = SimpleRunner(
    model='mask_rcnn_swin_s',
    dataset={
        "_base_": "cmr_instance_segmentation",
        "ann_file": "/path/to/train.json",
        "val_ann_file": "/path/to/val.json"
    }
)
```

**After (clearer):**
```python
runner = SimpleRunner(
    model='mask_rcnn_swin_s',
    dataset='cmr_instance_segmentation',
    train_ann_file='/path/to/train.json',
    val_ann_file='/path/to/val.json'
)
```

### Benefits of Migration

- ✅ Intent is explicit and clear
- ✅ IDE provides autocomplete for parameters
- ✅ Type hints make code safer
- ✅ Easier to debug configuration issues
- ✅ Less code duplication in loops

## Best Practices

### 1. Use Absolute Paths

Absolute paths are more reliable and less ambiguous:

```python
from pathlib import Path

# Good: absolute path
train_ann = Path('/home/user/data/annotations/train.json').resolve()

# Okay: relative path is made absolute
train_ann = (Path.cwd() / 'data' / 'annotations' / 'train.json').resolve()

# Avoid: relative paths without context
train_ann = 'data/annotations/train.json'  # depends on cwd
```

### 2. Validate Paths Before Training

The SimpleRunner will validate files exist, but validate early in your pipeline:

```python
from pathlib import Path

def validate_annotation_files(train_ann, val_ann=None):
    """Validate annotation files before training."""
    train_path = Path(train_ann)
    if not train_path.exists():
        raise FileNotFoundError(f"Training annotations not found: {train_path}")

    if val_ann:
        val_path = Path(val_ann)
        if not val_path.exists():
            raise FileNotFoundError(f"Validation annotations not found: {val_path}")

    return train_path, val_path if val_ann else None

train_ann, val_ann = validate_annotation_files(
    '/mlflow/artifacts/train.json',
    '/mlflow/artifacts/val.json'
)

runner = SimpleRunner(
    model='mask_rcnn_swin_s',
    dataset='cmr_instance_segmentation',
    train_ann_file=str(train_ann),
    val_ann_file=str(val_ann)
)
```

### 3. Use Environment Variables for Flexibility

```python
import os
from pathlib import Path

train_ann = Path(os.environ.get('TRAIN_ANN_FILE'))
val_ann = Path(os.environ.get('VAL_ANN_FILE'))

runner = SimpleRunner(
    model='mask_rcnn_swin_s',
    dataset='cmr_instance_segmentation',
    train_ann_file=str(train_ann),
    val_ann_file=str(val_ann)
)
```

Usage:
```bash
export TRAIN_ANN_FILE=/data/experiment_123/train.json
export VAL_ANN_FILE=/data/experiment_123/val.json
python train.py
```

### 4. Log Configuration

Always log which annotation files are being used:

```python
import logging

logger = logging.getLogger(__name__)

logger.info(f"Training with annotations: {train_ann}")
logger.info(f"Validating with annotations: {val_ann}")

runner = SimpleRunner(
    model='mask_rcnn_swin_s',
    dataset='cmr_instance_segmentation',
    train_ann_file=train_ann,
    val_ann_file=val_ann
)
runner.train()
```

### 5. Handle Missing Files Gracefully

```python
from pathlib import Path

def get_annotation_files_with_fallback(
    experiment_id: str,
    fallback_dir: Path
) -> tuple:
    """Get annotation files with fallback to default."""
    # Try to get from experiment tracking system
    mlflow_train = download_from_mlflow(experiment_id, 'train.json')
    mlflow_val = download_from_mlflow(experiment_id, 'val.json')

    if mlflow_train and mlflow_val:
        return mlflow_train, mlflow_val

    # Fall back to local files
    logger.warning(f"MLflow files not found for {experiment_id}, using fallback")
    return (
        fallback_dir / 'train.json',
        fallback_dir / 'val.json'
    )

train_ann, val_ann = get_annotation_files_with_fallback(
    'experiment_abc',
    Path('/data/default_annotations')
)

runner = SimpleRunner(
    model='mask_rcnn_swin_s',
    dataset='cmr_instance_segmentation',
    train_ann_file=str(train_ann),
    val_ann_file=str(val_ann)
)
```

## Troubleshooting

### FileNotFoundError During Initialization

**Problem:** SimpleRunner raises `FileNotFoundError` when you provide a path.

**Solution:**
1. Check the file actually exists: `ls -la /path/to/file.json`
2. Use absolute paths instead of relative: `Path(path).resolve()`
3. Check for typos in the path
4. Verify your working directory: `os.getcwd()`

### Annotation File Not Being Used

**Problem:** Your train_ann_file or val_ann_file isn't being used, the preset defaults are used instead.

**Solution:**
- Make sure you're passing the parameters: `train_ann_file=/path`, not just the dict
- Check for typos in parameter names: `train_ann_file` (not `train_annotation_file`)
- Verify the file is actually different from the preset

### Missing Validation Annotations

**Problem:** You want validation but val_ann_file isn't being used.

**Causes:**
- Dataset preset might have `val_ann_file` already defined (preset takes precedence)
- Forgot to specify both `val_ann_file` AND that validation data exists

**Solution:**
```python
# Make sure val_ann_file is specified
runner = SimpleRunner(
    model='mask_rcnn_swin_s',
    dataset='cmr_instance_segmentation',
    train_ann_file=train_path,
    val_ann_file=val_path,  # Important!
    epochs=12
)
```

## Performance Notes

- **Validation time**: Files are validated once during __init__, adds minimal overhead
- **Multi-GPU (DDP)**: Parameters are stored for worker recreation, no extra cost
- **Large files**: No performance difference between dynamic and preset files

## Backward Compatibility

✅ **Fully backward compatible**:
- Existing code without these parameters works unchanged
- Default values are `None` (use preset definitions)
- No breaking changes to existing APIs

## Related Features

- **Dataset Presets**: Still useful for defining pipelines, metainfo, and default paths
- **DDP (Distributed Data Parallel)**: Fully compatible, parameters stored for workers
- **Config Overrides**: Can still use dict overrides for other config parameters

## See Also

- [Examples](../examples/dynamic_annotation_workflow.py)
- [SimpleRunner Source Code](../visdet/runner.py)
- [Test Suite](../tests/test_simple_runner_annotation_files.py)
