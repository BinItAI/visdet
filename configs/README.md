# YAML Configuration Files

This directory contains modular YAML configuration files for visdet training and inference.

## Structure

```
configs/
├── components/          # Reusable component library
│   ├── backbones/      # Backbone configs (Swin Transformer, etc.)
│   ├── necks/          # Neck configs (FPN, etc.)
│   ├── heads/          # Head configs (RPN, RoI heads, etc.)
│   ├── datasets/       # Dataset configs (COCO, custom datasets)
│   ├── optimizers/     # Optimizer configs (AdamW, SGD, etc.)
│   └── schedules/      # Training schedule configs (1x, 2x, etc.)
└── experiments/         # Full experiment configs
    └── mask_rcnn_swin_tiny_coco.yaml
```

## Quick Start

### Load a config in Python

```python
from visdet.engine.config import Config

cfg = Config.fromfile('configs/experiments/mask_rcnn_swin_tiny_coco.yaml')
```

### Use with training scripts

```bash
# Using the simple training script
python scripts/train.py configs/experiments/mask_rcnn_swin_tiny_coco.yaml

# Using the full training script
python tools/train.py configs/experiments/mask_rcnn_swin_tiny_coco.yaml
```

## Component Configs

Component configs define individual building blocks that can be referenced in experiment configs.

### Example: Backbone Config

`components/backbones/swin_tiny.yaml`:
```yaml
type: SwinTransformer
embed_dims: 96
depths: [2, 2, 6, 2]
num_heads: [3, 6, 12, 24]
window_size: 7
```

### Example: Optimizer Config

`components/optimizers/adamw_default.yaml`:
```yaml
type: AdamW
lr: 0.0001
betas: [0.9, 0.999]
weight_decay: 0.05
```

## Experiment Configs

Experiment configs define complete training setups by:
1. Inheriting base configs with `_base_`
2. Referencing components with `$ref`
3. Overriding specific values

### Example: Full Experiment

`experiments/mask_rcnn_swin_tiny_coco.yaml`:
```yaml
# Inherit base configurations
_base_:
  - ../components/datasets/coco_instance.yaml
  - ../components/optimizers/adamw_default.yaml
  - ../components/schedules/1x.yaml

# Model definition with component references
model:
  type: MaskRCNN
  backbone:
    $ref: ../components/backbones/swin_tiny.yaml  # Reference external file
  neck:
    $ref: ../components/necks/fpn_256.yaml
  # ... rest of model config

# Override base values
max_epochs: 24  # Override from 12 (in 1x.yaml)
val_interval: 2  # Override from 1
```

## Key Features

### _base_ Inheritance

Merge configurations from parent files:
```yaml
_base_:
  - base_config1.yaml
  - base_config2.yaml
```

### $ref Resolution

Reference other YAML files:
```yaml
backbone:
  $ref: ./backbones/swin_tiny.yaml
```

### Config Overrides

Override inherited values:
```yaml
_base_:
  - ../components/schedules/1x.yaml  # Defines max_epochs: 12

max_epochs: 24  # Override to 24
```

## Available Configs

### Backbones
- `swin_tiny.yaml` - Swin Transformer Tiny (96 dims, [2,2,6,2] depths)

### Necks
- `fpn_256.yaml` - Feature Pyramid Network with 256 channels

### Datasets
- `coco_instance.yaml` - COCO instance segmentation dataset

### Optimizers
- `adamw_default.yaml` - AdamW optimizer (lr=0.0001, wd=0.05)

### Schedules
- `1x.yaml` - Standard 12-epoch COCO schedule

### Experiments
- `mask_rcnn_swin_tiny_coco.yaml` - Mask R-CNN with Swin Tiny backbone

## Creating New Configs

### 1. Create a Component Config

```yaml
# configs/components/backbones/my_backbone.yaml
type: MyBackbone
param1: value1
param2: value2
```

### 2. Reference in Experiment

```yaml
# configs/experiments/my_experiment.yaml
_base_:
  - ../components/datasets/coco_instance.yaml
  - ../components/optimizers/adamw_default.yaml

model:
  type: MyDetector
  backbone:
    $ref: ../components/backbones/my_backbone.yaml
```

### 3. Test Loading

```bash
python -c "from visdet.engine.config import Config; cfg = Config.fromfile('configs/experiments/my_experiment.yaml'); print(cfg.model.backbone.type)"
```

## Documentation

For detailed documentation, see:
- `docs/yaml_config_system.md` - Complete YAML config system guide
- `scripts/test_yaml_simple.py` - Test suite and usage examples

## Migration from Python Configs

Python `.py` configs are deprecated but still work with warnings. A migration tool will be provided to convert existing configs to YAML format.

```bash
# Coming soon
python tools/convert_config.py --input config.py --output configs/
```
