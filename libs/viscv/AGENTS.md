# viscv

This is a trimmed-down version of MMCV, designed to support only the minimal computer vision operations needed for Swin Mask R-CNN.

## Key Principles

1. **No C++ Extensions**: All operations should be implemented in pure PyTorch or use torchvision ops
2. **Minimal Dependencies**: Only depend on PyTorch, torchvision, and numpy
3. **No ext_loader**: Remove all references to MMCV's extension loader system

## What to Keep

- Image I/O operations
- Basic image transformations (resize, normalize, etc.)
- Color space conversions
- Bounding box operations (pure Python/PyTorch)
- Basic visualization utilities

## What to Remove

- All C++ extensions and CUDA kernels
- Complex ops that require compilation (deformable conv, etc.)
- Video processing capabilities
- 3D operations
- Ops not used by Swin Mask R-CNN

## Import Structure

All imports should use absolute paths: `from viscv.image import imread`

---

*For machine learning guidelines, see the machine_learning/AGENTS.md file.*
*For general repository guidelines, see the root AGENTS.md file.*
