# visdet - Repository Guidelines

This is a minimal version of MMDetection, supporting only Swin Mask R-CNN for object detection and instance segmentation.

## Repository Structure

### Documentation Files

- **README.md** - Main project documentation (keep at root)
- **AGENTS.md** - This file, repository-wide guidelines (keep at root)
- **Other capitalised markdown files** - Should be written to the `scratch_pads/` directory

The `scratch_pads/` directory is intended for experimental documentation, planning documents, and other markdown files that are not part of the core repository documentation. This keeps the root directory clean and organized.

## Key Principles

1. **Single Model Focus**: Only support Swin Transformer + Mask R-CNN
2. **COCO Format**: Only support COCO-style datasets
3. **Essential Components**: Keep only what's needed for this specific model
4. **Absolute Imports**: Always use absolute imports (e.g., `from visdet.engine import X`) instead of relative imports (e.g., `from .engine import X`) to avoid circular import issues

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

- Training infrastructure (visengine)
- Image operations (viscv)
- pycocotools for COCO evaluation

---

*For model-specific guidelines, see `visdet/AGENTS.md`*
*For personal development guidelines, see `~/.claude/CLAUDE.md` (local only)*
