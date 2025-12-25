# MMDetection Model Support in VisDet

This document provides a comparison of models available in [MMDetection](https://github.com/open-mmlab/mmdetection) and their support status in VisDet.

## Summary

VisDet currently focuses on **two-stage detectors** with a clean, typed, and well-tested codebase. We prioritize quality over quantity, ensuring each supported model works reliably.

## Detector Architectures

| Model | MMDetection | VisDet | Notes |
|-------|:-----------:|:------:|-------|
| **Two-Stage Detectors** |
| Faster R-CNN | ✅ | ✅ | Core detector, fully supported |
| Mask R-CNN | ✅ | ✅ | Instance segmentation supported |
| Cascade R-CNN | ✅ | ✅ | Multi-stage refinement |
| Cascade Mask R-CNN | ✅ | ✅ | Via Cascade R-CNN + mask head |
| Fast R-CNN | ✅ | ❌ | |
| RPN | ✅ | ✅ | Region Proposal Network |
| HTC (Hybrid Task Cascade) | ✅ | ❌ | |
| MS R-CNN (Mask Scoring) | ✅ | ❌ | |
| SCNet | ✅ | ❌ | |
| TridentNet | ✅ | ❌ | |
| Sparse R-CNN | ✅ | ❌ | |
| QueryInst | ✅ | ❌ | |
| Grid R-CNN | ✅ | ❌ | |
| Double Heads | ✅ | ❌ | |
| Dynamic R-CNN | ✅ | ❌ | |
| Libra R-CNN | ✅ | ❌ | |
| Groie | ✅ | ❌ | |
| DetectoRS | ✅ | ❌ | |
| **One-Stage Detectors** |
| RetinaNet | ✅ | ❌ | |
| SSD | ✅ | ❌ | |
| FCOS | ✅ | ❌ | |
| ATSS | ✅ | ❌ | |
| GFL | ✅ | ❌ | |
| VFNet | ✅ | ❌ | |
| YOLACT | ✅ | ❌ | |
| YOLOv3 | ✅ | ❌ | |
| YOLOX | ✅ | ❌ | |
| YOLOF | ✅ | ❌ | |
| RTMDet | ✅ | ❌ | |
| TOOD | ✅ | ❌ | |
| PAA | ✅ | ❌ | |
| DDOD | ✅ | ❌ | |
| FSAF | ✅ | ❌ | |
| FreeAnchor | ✅ | ❌ | |
| FoveaBox | ✅ | ❌ | |
| CornerNet | ✅ | ❌ | |
| CenterNet | ✅ | ❌ | |
| CentripetalNet | ✅ | ❌ | |
| RepPoints | ✅ | ❌ | |
| GHM | ✅ | ❌ | |
| NAS-FPN | ✅ | ❌ | |
| NAS-FCOS | ✅ | ❌ | |
| AutoAssign | ✅ | ❌ | |
| SABL | ✅ | ❌ | |
| **Transformer-Based Detectors** |
| DETR | ✅ | ❌ | |
| Deformable DETR | ✅ | ❌ | |
| Conditional DETR | ✅ | ❌ | |
| DAB-DETR | ✅ | ❌ | |
| DINO | ✅ | ❌ | |
| DDQ | ✅ | ❌ | |
| Grounding DINO | ✅ | ❌ | |
| MM-Grounding-DINO | ✅ | ❌ | |
| GLIP | ✅ | ❌ | |
| **Panoptic/Instance Segmentation** |
| MaskFormer | ✅ | ❌ | |
| Mask2Former | ✅ | ❌ | |
| Panoptic FPN | ✅ | ❌ | |
| SOLO | ✅ | ❌ | |
| SOLOv2 | ✅ | ❌ | |
| CondInst | ✅ | ❌ | |
| BoxInst | ✅ | ❌ | |
| Point Rend | ✅ | ❌ | |
| **Tracking** |
| ByteTrack | ✅ | ❌ | |
| QDTrack | ✅ | ❌ | |
| SORT | ✅ | ❌ | |
| DeepSORT | ✅ | ❌ | |
| OC-SORT | ✅ | ❌ | |
| StrongSORT | ✅ | ❌ | |
| MaskTrack R-CNN | ✅ | ❌ | |
| **Knowledge Distillation** |
| LAD | ✅ | ❌ | |
| LD | ✅ | ❌ | |

## Backbones

| Backbone | MMDetection | VisDet | Notes |
|----------|:-----------:|:------:|-------|
| ResNet | ✅ | ✅ | ResNet-18/34/50/101/152 |
| ResNeXt | ✅ | ✅ | |
| Res2Net | ✅ | ✅ | |
| ResNeSt | ✅ | ✅ | |
| RegNet | ✅ | ✅ | |
| HRNet | ✅ | ✅ | |
| Swin Transformer | ✅ | ✅ | |
| ConvNeXt | ✅ | ❌ | |
| PVT | ✅ | ❌ | |
| EfficientNet | ✅ | ❌ | |
| VGG | ✅ | ❌ | |
| MobileNet | ✅ | ❌ | |
| DetectoRS ResNet | ✅ | ❌ | |
| CSPDarknet | ✅ | ❌ | |
| CSPNeXt | ✅ | ❌ | |

## Necks

| Neck | MMDetection | VisDet | Notes |
|------|:-----------:|:------:|-------|
| FPN | ✅ | ✅ | Feature Pyramid Network |
| PAFPN | ✅ | ❌ | |
| BiFPN | ✅ | ❌ | |
| NAS-FPN | ✅ | ❌ | |
| CARAFE FPN | ✅ | ❌ | |
| FPG | ✅ | ❌ | |
| RFNext | ✅ | ❌ | |
| DyHead | ✅ | ❌ | |

## Techniques & Modules

| Technique | MMDetection | VisDet | Notes |
|-----------|:-----------:|:------:|-------|
| DCN (Deformable Conv) | ✅ | ❌ | |
| DCNv2 | ✅ | ❌ | |
| Group Normalization | ✅ | ❌ | |
| Weight Standardization | ✅ | ❌ | |
| Guided Anchoring | ✅ | ❌ | |
| CARAFE | ✅ | ❌ | |
| InstaBoost | ✅ | ❌ | |
| Albumentations | ✅ | ✅ | Via configs |
| Simple Copy-Paste | ✅ | ❌ | |
| Seesaw Loss | ✅ | ❌ | |
| PISA | ✅ | ❌ | |
| Soft Teacher | ✅ | ❌ | |

## Datasets

| Dataset | MMDetection | VisDet | Notes |
|---------|:-----------:|:------:|-------|
| COCO | ✅ | ✅ | |
| PASCAL VOC | ✅ | ✅ | |
| LVIS | ✅ | ✅ | |
| Objects365 | ✅ | ✅ | |
| OpenImages | ✅ | ✅ | |
| Cityscapes | ✅ | ✅ | |
| WIDER Face | ✅ | ✅ | |
| DeepFashion | ✅ | ❌ | |

## Roadmap

We plan to add support for:
- [ ] RetinaNet (one-stage anchor-based)
- [ ] FCOS (anchor-free)
- [ ] DETR family (transformer-based)
- [ ] RTMDet (real-time)
- [ ] More backbones (ConvNeXt, EfficientNet)

## Contributing

Want to help add support for a model? See our [contribution guide](../CONTRIBUTING.md) or open an issue to discuss!
