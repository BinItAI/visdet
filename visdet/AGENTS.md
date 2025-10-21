# visdet

This is a minimal version of MMDetection, supporting only Swin Mask R-CNN for object detection and instance segmentation.

Be very very careful with any edits to the underlying model code, always check against
the reference mmdetection repo we have locally.

## Key Principles

1. **Single Model Focus**: Only support Swin Transformer + Mask R-CNN
2. **COCO Format**: Only support COCO-style datasets
3. **Essential Components**: Keep only what's needed for this specific model

## What to Keep

### Models

- **Backbone**: SwinTransformer only
- **Neck**: FPN only
- **Head**: RPNHead, StandardRoIHead (with bbox and mask branches)
- **Detector**: MaskRCNN (two-stage detector)

### Data

- COCO dataset format support
- Essential data transforms for training/inference
- DetDataPreprocessor

### Evaluation

- COCO metrics for object detection and instance segmentation

## What to Remove

- All other backbones (ResNet, RegNet, etc.)
- All other detectors (YOLO, RetinaNet, DETR, etc.)
- All other necks (PAFPN, NAS-FPN, etc.)
- Video/tracking components
- 3D detection components
- Panoptic segmentation
- All other dataset formats

## Dependencies

- visengine for training infrastructure
- viscv for image operations
- pycocotools for COCO evaluation

---

*For machine learning guidelines, see the machine_learning/AGENTS.md file.*
*For general repository guidelines, see the root AGENTS.md file.*
