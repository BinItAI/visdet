# Architecture

This document describes the architecture and design principles of visdet.

## Overview

visdet is built on MMDetection, a comprehensive object detection framework. The architecture follows a modular design that separates concerns and promotes code reusability.

## Core Components

### 1. Models

The model architecture is composed of four main components:

```
Input Image
    ↓
Backbone (Feature Extraction)
    ↓
Neck (Feature Fusion)
    ↓
Head (Detection/Segmentation)
    ↓
Output (Boxes, Masks, etc.)
```

#### Backbone
Extracts multi-scale features from input images.

- **Supported**: ResNet, ResNeXt, Swin Transformer, ConvNeXt, etc.
- **Configurable**: Depth, normalization, activation functions

#### Neck
Fuses features from different scales.

- **FPN**: Feature Pyramid Network
- **PAFPN**: Path Aggregation FPN
- **BiFPN**: Bidirectional FPN

#### Head
Produces final predictions.

- **RPNHead**: Region Proposal Network
- **BBoxHead**: Bounding box regression and classification
- **MaskHead**: Instance segmentation masks

### 2. Datasets

Dataset components handle data loading and preprocessing:

```python
Dataset
    ↓
Pipeline (Transforms)
    ↓
DataLoader
    ↓
Batch
```

Key features:
- **Lazy Loading**: Efficient memory usage
- **Caching**: Speed up repeated access
- **Augmentation**: Extensive data augmentation support

### 3. Training Loop

The training loop is managed by the Runner:

```
Runner
├── Hooks (Callbacks)
│   ├── LoggerHook
│   ├── CheckpointHook
│   ├── EvalHook
│   └── CustomHooks
├── Optimizer
├── LR Scheduler
└── Workflow
```

### 4. Registry System

All components are registered in a registry system for dynamic instantiation:

```python
from mmdet.models import DETECTORS, BACKBONES, NECKS, HEADS

@BACKBONES.register_module()
class MyBackbone:
    pass

# Instantiate from config
backbone = BACKBONES.build(dict(type='MyBackbone', ...))
```

## Design Principles

### Modularity

Each component is self-contained and can be used independently:

```python
# Use only the backbone
from mmdet.models import build_backbone

backbone = build_backbone(dict(
    type='ResNet',
    depth=50,
    num_stages=4,
))
```

### Configurability

Everything is configurable through Python config files:

```python
model = dict(
    type='FasterRCNN',
    backbone=dict(type='ResNet', depth=50),
    neck=dict(type='FPN', ...),
    rpn_head=dict(type='RPNHead', ...),
    roi_head=dict(type='StandardRoIHead', ...),
)
```

### Extensibility

Easy to add custom components:

```python
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import TwoStageDetector

@DETECTORS.register_module()
class MyDetector(TwoStageDetector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add custom logic
```

## Data Flow

### Training

```
1. Load batch from DataLoader
2. Forward pass through model
3. Calculate losses
4. Backward pass
5. Update weights
6. Run hooks (logging, checkpointing, etc.)
7. Repeat
```

### Inference

```
1. Load image
2. Preprocess (resize, normalize, etc.)
3. Forward pass
4. Post-process (NMS, score filtering)
5. Return results
```

## Key Abstractions

### BaseDetector

All detectors inherit from `BaseDetector`:

```python
class BaseDetector(BaseModule, metaclass=ABCMeta):
    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """Training forward"""
        pass

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        """Test without augmentation"""
        pass

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Unified forward interface"""
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.simple_test(img, img_metas, **kwargs)
```

### Hook System

Hooks provide callbacks at different training stages:

```python
class Hook:
    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass
```

## Performance Considerations

### Mixed Precision Training

Uses automatic mixed precision (AMP) for faster training:

```python
optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale=512.
)
```

### Gradient Accumulation

For larger effective batch sizes:

```python
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=4
)
```

### Multi-GPU Training

Distributed training with DDP:

```bash
bash tools/dist_train.sh config.py 8  # 8 GPUs
```

## See Also

- [Contributing Guide](contributing.md)
- [API Reference](../api-reference/index.md)
- [Configuration Guide](../user-guide/configuration.md)
