# visdet

A comprehensive port of MMDetection supporting all detection architectures that work with pure PyTorch (no custom CUDA extensions required). Includes two-stage detectors, single-stage detectors, anchor-free methods, and transformer-based models.

Be very careful with any edits to the underlying model code, always check against
the reference mmdetection repo we have locally.

## Key Principles

1. **Comprehensive Model Support**: Support all MMDetection model architectures that don't require custom CUDA extensions
2. **Pure PyTorch**: Use pure PyTorch implementations and torchvision ops. No deformable convolutions, custom CUDA kernels, or CUDA-only operations
3. **COCO Format**: Only support COCO-style datasets
4. **Absolute Imports**: Always use absolute imports (e.g., `from visdet.engine import X`) instead of relative imports (e.g., `from .engine import X`) to avoid circular import issues
5. **No sys.modules Manipulation**: Never use `sys.modules` hacks to create module aliases. Update imports directly or use proper re-exports in `__init__.py` files
6. **No Try/Except Around Imports**: Do not wrap imports in try/except blocks unless explicitly required. Import failures should fail loudly

## Supported Models

### Detectors

**Two-Stage:**
- Faster R-CNN, Mask R-CNN, Cascade R-CNN, Cascade Mask R-CNN
- HTC, TridentNet, Double Head R-CNN, Libra R-CNN, PointRend

**Single-Stage:**
- RetinaNet, FCOS, ATSS, GFL, VFNet, TOOD, PAA, FreeAnchor, FSAF
- SSD, YOLOV3, YOLOX, YOLOF, RTMDet, CenterNet, FoveaBox

**Transformer-Based:**
- DETR, Conditional DETR, DAB-DETR, DN-DETR, DINO, Sparse R-CNN

**Instance Segmentation:**
- QueryInst, Mask2Former, MaskFormer, YOLACT

### Backbones

- ResNet (18, 34, 50, 101, 152), ResNeXt, ResNeSt, Res2Net, RegNet
- SwinTransformer, HRNet, VGG, MobileNet (V2, V3), EfficientNet
- ConvNeXt, ConvNeXtV2, PVT, PVTv2, CSPDarknet, CSPNet

### Necks

- FPN, PAFPN, BFP, NASFPN, HRFPN, YOLOV3Neck, YOLOXPAFPN, SSDNeck, ChannelMapper

### Excluded (Require CUDA Extensions)

- Deformable DETR, CornerNet, CentripetalNet
- SOLO, SOLOv2, CondInst
- DetectoRS, Grid R-CNN, RepPoints, Guided Anchor
- FPN-CARAFE, any model using DCN

## Data

- COCO dataset format support
- Essential data transforms for training/inference
- DetDataPreprocessor

## Evaluation

- COCO metrics for object detection and instance segmentation

## Dependencies

- visengine for training infrastructure
- viscv for image operations
- pycocotools for COCO evaluation
- torchvision for RoI ops and NMS

---

*For machine learning guidelines, see the machine_learning/AGENTS.md file.*
*For general repository guidelines, see the root AGENTS.md file.*
