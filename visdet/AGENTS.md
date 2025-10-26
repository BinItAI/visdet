# visdet

This is a minimal version of MMDetection, supporting only Swin Mask R-CNN for object detection and instance segmentation.

Be very very careful with any edits to the underlying model code, always check against
the reference mmdetection repo we have locally.

## Key Principles

1. **Single Model Focus**: Only support Swin Transformer + Mask R-CNN
2. **COCO Format**: Only support COCO-style datasets
3. **Essential Components**: Keep only what's needed for this specific model
4. **Absolute Imports**: Always use absolute imports (e.g., `from visdet.engine import X`) instead of relative imports (e.g., `from .engine import X`) to avoid circular import issues
5. **No sys.modules Manipulation**: Never use `sys.modules` hacks to create module aliases (e.g., `sys.modules["old_name"] = new_module`). This pollutes global state and makes debugging difficult. Instead, update imports directly at their source or use proper re-exports in `__init__.py` files for backward compatibility
6. **No Try/Except Around Imports**: Do not wrap imports in try/except blocks unless explicitly required for a specific use case. This masks missing dependencies and makes debugging difficult. If an import fails, it should fail loudly so that dependency issues are caught immediately

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
