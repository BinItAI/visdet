# Configuration Files (YAML)

visdet uses a YAML-only configuration layout.

## Structure

```
configs/
├── datasets/      # Dataset presets (COCO, CMR, ...)
├── models/        # Model presets (Mask R-CNN, RTMDet, ...)
├── optimizers/    # Optimizer presets
├── schedulers/    # Scheduler presets
└── README.md
```

## Usage

### Load a YAML config

```python
from visdet.engine.config import Config

cfg = Config.fromfile("configs/models/rtmdet_s.yaml")
```

### Use with `SimpleRunner`

```python
from visdet.runner import SimpleRunner

runner = SimpleRunner(
    model="rtmdet_s",
    dataset="coco_detection",
    optimizer="adamw_default",
    scheduler="1cycle",
)
```

## Notes

- The legacy MMDetection-style Python config zoo has been removed.
- Model-name discovery for inferencers is handled via YAML presets.
