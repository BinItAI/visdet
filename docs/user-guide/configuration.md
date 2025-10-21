# Configuration System

visdet uses a powerful configuration system based on Python files. This guide explains how to work with configurations.

## Configuration Files

Configuration files are Python files that define:

- Model architecture
- Dataset settings
- Training schedules
- Runtime settings

## Basic Structure

A typical config file:

```python
# Model
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    # ...
)

# Dataset
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    # ...
)

# Training schedule
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
```

## Config Inheritance

Configs can inherit from base configs using `_base_`:

```python
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# Override specific settings
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=80)))
```

## Common Configurations

### Model Settings

```python
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        # ...
    ),
    # ...
)
```

### Dataset Settings

```python
data = dict(
    samples_per_gpu=2,  # Batch size per GPU
    workers_per_gpu=2,  # Data loader workers
    train=dict(
        type='CocoDataset',
        ann_file='path/to/annotations.json',
        img_prefix='path/to/images/',
        # ...
    ),
)
```

### Training Settings

```python
# Optimizer
optimizer = dict(
    type='SGD',
    lr=0.02,
    momentum=0.9,
    weight_decay=0.0001)

# Learning rate schedule
lr_config = dict(
    policy='step',
    step=[8, 11])

# Training epochs
runner = dict(
    type='EpochBasedRunner',
    max_epochs=12)
```

## Modifying Configs

### From Command Line

Override config values via command line:

```bash
python tools/train.py config.py --cfg-options optimizer.lr=0.01
```

### Programmatically

```python
from mmcv import Config

cfg = Config.fromfile('config.py')
cfg.optimizer.lr = 0.01
cfg.dump('modified_config.py')
```

## Environment Variables

Use environment variables in configs:

```python
import os

data_root = os.getenv('DATA_ROOT', 'data/coco/')
```

## Best Practices

1. **Use Base Configs**: Leverage inheritance to avoid duplication
2. **Document Changes**: Add comments explaining modifications
3. **Version Control**: Keep configs in version control
4. **Organize**: Group related configs in directories
5. **Validate**: Test configs before long training runs

## See Also

- [Training Guide](training.md)
- [Config Tutorial](../tutorials/config.md)
- [Base Configs](https://github.com/BinItAI/visdet/tree/master/configs/_base_)
