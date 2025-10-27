# visdet

This is a port of MMDetection with a focus on essential object detection and instance segmentation models. It supports multiple modern backbones (Swin Transformer, ResNet) with standard detection heads (Faster R-CNN, Mask R-CNN).

Be very careful with any edits to the underlying model code, always check against
the reference mmdetection repo we have locally.

## Key Principles

1. **Essential Model Support**: Focus on widely-used architectures and configurations that have strong proven track records
2. **COCO Format**: Only support COCO-style datasets
3. **Absolute Imports**: Always use absolute imports (e.g., `from visdet.engine import X`) instead of relative imports (e.g., `from .engine import X`) to avoid circular import issues
4. **Pure PyTorch**: Avoid custom CUDA operations when possible. Use pure PyTorch implementations or standard operations with graceful fallbacks
5. **No sys.modules Manipulation**: Never use `sys.modules` hacks to create module aliases (e.g., `sys.modules["old_name"] = new_module`). This pollutes global state and makes debugging difficult. Instead, update imports directly at their source or use proper re-exports in `__init__.py` files for backward compatibility
6. **No Try/Except Around Imports**: Do not wrap imports in try/except blocks unless explicitly required for a specific use case. This masks missing dependencies and makes debugging difficult. If an import fails, it should fail loudly so that dependency issues are caught immediately

## What to Keep

### Models

- **Backbones**: SwinTransformer, ResNet (18, 34, 50, 101, 152), ResNeXt (50, 101, 152), RegNet (400mf, 800mf, 1.6gf, 3.2gf, 4.0gf, 6.4gf, 8.0gf, 12gf), ResNeSt (50, 101, 152, 200), HRNet (18, 32, 48, 64), Res2Net (50, 101, 152), ViT (Base, Large)
- **Neck**: FPN
- **Heads**: RPNHead, StandardRoIHead (with bbox and mask branches)
- **Detectors**: MaskRCNN, Faster R-CNN (two-stage detectors)

### Data

- COCO dataset format support
- Essential data transforms for training/inference
- DetDataPreprocessor

### Evaluation

- COCO metrics for object detection and instance segmentation

## What to Remove

- Experimental/complex backbones (RegNet, EfficientNet, etc.) - focus on proven architectures
- Specialized detectors (YOLO, RetinaNet, DETR, etc.) - focus on two-stage detectors
- Experimental necks (PAFPN, NAS-FPN, etc.) - keep core necks (FPN)
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
