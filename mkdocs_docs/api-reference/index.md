# API Reference

Welcome to the visdet API reference documentation.

## Overview

visdet provides a comprehensive API for object detection tasks, built on top of MMDetection. The API is organized into several key modules:

## Core Modules

### [Detection APIs](../api/core.md)
High-level APIs for model initialization, training, and inference.

- `init_detector`: Initialize a detector from config
- `inference_detector`: Run inference on images
- `train_detector`: Train a detection model
- `show_result_pyplot`: Visualize detection results

### [Models](../api/models.md)
Model architectures and components.

- **Detectors**: Two-stage and single-stage detectors
- **Backbones**: ResNet, ResNeXt, Swin Transformer, etc.
- **Necks**: FPN, PAFPN, etc.
- **Heads**: Detection heads for various architectures

### [Datasets](../api/datasets.md)
Dataset classes and data pipelines.

- **Dataset Classes**: COCO, VOC, custom datasets
- **Transforms**: Data augmentation and preprocessing
- **Loaders**: DataLoader utilities

## Quick Examples

### Initialize a Model

```python
from mmdet.apis import init_detector

config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')
```

### Run Inference

```python
from mmdet.apis import inference_detector, show_result_pyplot

img = 'demo/demo.jpg'
result = inference_detector(model, img)
show_result_pyplot(model, img, result, score_thr=0.3)
```

### Work with Datasets

```python
from mmdet.datasets import build_dataset
from mmcv import Config

cfg = Config.fromfile('config.py')
dataset = build_dataset(cfg.data.train)
```

## Module Organization

```
mmdet/
├── apis/          # High-level APIs
├── models/        # Model architectures
│   ├── detectors/ # Detector implementations
│   ├── backbones/ # Backbone networks
│   ├── necks/     # Neck modules
│   └── heads/     # Detection heads
├── datasets/      # Dataset classes
│   ├── pipelines/ # Data transforms
│   └── samplers/  # Data samplers
├── core/          # Core utilities
└── utils/         # Helper functions
```

## Additional Resources

- [User Guide](../user-guide/training.md)
- [Tutorials](../tutorials/config.md)
- [GitHub Repository](https://github.com/BinItAI/visdet)

## API Stability

The API follows semantic versioning. While we strive to maintain backward compatibility, some APIs may change between major versions. Deprecated functions will include warnings with migration guidance.
