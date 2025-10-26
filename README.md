<div align="center">
  <img src="resources/visia-logo-white.svg" width="600"/>
  <div>&nbsp;</div>

  <p>
    <a href="https://pypi.org/project/visdet"><img src="https://img.shields.io/pypi/v/visdet?style=flat-square&color=blue" alt="PyPI"/></a>
    <a href="https://binitai.github.io/visdet/"><img src="https://img.shields.io/badge/docs-latest-brightgreen?style=flat-square" alt="Documentation"/></a>
  </p>

  <!-- Navigation -->
  <h3>
    <a href="https://binitai.github.io/visdet/">📘 Documentation</a> •
    <a href="https://binitai.github.io/visdet/getting-started/installation/">🛠️ Installation</a> •
    <a href="https://binitai.github.io/visdet/model-zoo/">👀 Model Zoo</a> •
    <a href="https://binitai.github.io/visdet/about/changelog/">🆕 Changelog</a>
  </h3>

</div>

<br>

> **Note**: This is a fork of the MMDetection library, customized for [Visia](https://www.visia.ai/). The original project: [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)

---

## ✨ Why visdet?

<div align="center">

**Simplified installation • No CUDA compilation • Pure Python/PyTorch**

</div>

**Simplified Installation & Dependencies**
- **Integrated Dependencies**: MMCV and MMEngine are bundled directly into the package as `visdet.cv` and `visdet.engine`, eliminating complex multi-package dependency management
- **No Custom CUDA Required**: All custom CUDA operations have been removed, making installation straightforward with just `uv pip install visdet`
- **Python-Only Implementation**: Pure Python/PyTorch implementation means faster installation and better compatibility across different environments
- **Unified Namespace**: All functionality accessible through a single coherent API (`visdet.cv` for computer vision ops, `visdet.engine` for training infrastructure)

This makes visdet significantly easier to install and deploy compared to the original MMDetection, which required careful coordination of multiple packages and custom CUDA compilation.

> **What happened to MMCV and MMEngine?** They've been integrated into visdet under the `visdet.cv` and `visdet.engine` namespaces respectively. Instead of managing separate `mmcv`, `mmengine`, and `mmdet` packages, everything you need is now in one place.

---

## 🧠 Modern Training Philosophy

visdet draws inspiration from the pioneering work of [**fast.ai**](https://www.fast.ai/), which demonstrated that common-sense training techniques could dramatically improve both accessibility and performance in deep learning. The abandoned [**icevision**](https://github.com/airctic/icevision) project attempted to bring these ideas to object detection but is no longer maintained.

**We're continuing that mission** by porting battle-tested techniques from image classification and LLM training:

- **Progressive Image Resizing**: Start training with smaller images, gradually increase resolution for faster convergence and better performance
- **Learning Rate Finders**: Automatically discover optimal learning rates instead of manual tuning
- **Discriminative Learning Rates**: Apply different learning rates to different network layers
- **1cycle Learning Rate Schedules**: Achieve better generalization with cyclical learning rates
- **Modern Fine-tuning Techniques**: Bringing approaches from LLM training (LoRA-style adaptations) to object detection

These techniques are proven in image classification and LLM fine-tuning but have been largely absent from object detection frameworks. visdet aims to make them **accessible and practical** for detection tasks, with sensible defaults and clear documentation.

> **Philosophy**: If a technique works reliably for ImageNet classification or LLM fine-tuning, it should work for object detection too. We're bringing the best ideas from across deep learning to vision tasks.
>
> See fast.ai's [ImageNet training guide](https://www.fast.ai/posts/2018-08-10-fastai-diu-imagenet.html) for an example of how these techniques work in practice.

---

## 🔮 Future Integrations

visdet is committed to integrating cutting-edge tools that improve performance, developer experience, and training efficiency:

### [Kornia](https://kornia.github.io/)
Differentiable computer vision library for PyTorch with geometric transformations, filtering, and augmentation pipelines. **Planned integration** for enhanced data augmentation capabilities with full gradient support.

### [Triton](https://triton-lang.org/)
OpenAI's Python-like GPU programming language for writing high-performance kernels without CUDA expertise. Could enable custom operators achieving performance comparable to expert-level CUDA code.

### [SPDL](https://facebookresearch.github.io/spdl/)
Meta's Scalable and Performant Data Loading library with built-in performance observability. Under evaluation for replacing current data loading bottlenecks.

### [DALI](https://developer.nvidia.com/dali)
NVIDIA's GPU-accelerated data loading library that offloads preprocessing to the GPU. Being considered for systems with high GPU-to-CPU ratios where CPU preprocessing becomes a bottleneck.

### [Modal](https://modal.com/)
Serverless GPU compute platform for Python that makes cloud training and inference effortless. Zero infrastructure setup with elastic GPU scaling and 100x faster cold starts than Docker. Could enable seamless cloud-based training workflows.

### [Tutel](https://github.com/microsoft/tutel)
Microsoft's highly optimized Mixture of Experts (MoE) implementation for PyTorch. Enables efficient sparse model training with dynamic expert routing and load balancing. Could enable scaling to much larger models while maintaining computational efficiency through conditional computation.

### [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
Microsoft's deep learning optimization library featuring ZeRO (Zero Redundancy Optimizer) for training massive models with limited GPU memory. Includes model compression techniques, efficient training optimizations, and inference acceleration. Could enable training larger detection models and faster inference through quantization and compression.

---

## 🚀 Quick Start

### Installation

```bash
# Using uv (recommended)
uv pip install visdet

# Or using pip (don't do this though you massochist)
pip install visdet
```

For detailed installation instructions, see the [Installation Guide](https://binitai.github.io/visdet/getting-started/installation/).

### Training a Model

```python
from visdet import SimpleRunner

# Simple, string-based API - just like Hugging Face or Ultralytics YOLO
runner = SimpleRunner(
    model='mask_rcnn_swin_s',
    dataset='coco_instance_segmentation',
    optimizer='adamw_8bit',
    scheduler='1cycle'
)

runner.train()
```

**Discover available presets:**
```python
SimpleRunner.list_models()       # ['mask_rcnn_swin_s', ...]
SimpleRunner.list_datasets()     # ['coco_instance_segmentation', ...]
SimpleRunner.show_preset('mask_rcnn_swin_s')  # View full config
```

**Customize via inheritance:**
```python
runner = SimpleRunner(
    model={
        '_base_': 'mask_rcnn_swin_s',
        'backbone': {'embed_dims': 128}  # Override specific params
    },
    dataset='coco_instance_segmentation'
)
```

For more examples and tutorials, visit the [Documentation](https://binitai.github.io/visdet/).

---

## 📚 Documentation

Comprehensive guides and tutorials available at [binitai.github.io/visdet](https://binitai.github.io/visdet/)

<details>
<summary><b>📖 Available Tutorials</b></summary>

- [Configuration System](https://binitai.github.io/visdet/tutorials/config/)
- [Custom Datasets](https://binitai.github.io/visdet/tutorials/customize_dataset/)
- [Data Pipelines](https://binitai.github.io/visdet/tutorials/data_pipeline/)
- [Custom Models](https://binitai.github.io/visdet/tutorials/customize_models/)
- [Runtime Settings](https://binitai.github.io/visdet/tutorials/customize_runtime/)
- [Custom Losses](https://binitai.github.io/visdet/tutorials/customize_losses/)
- [Fine-tuning](https://binitai.github.io/visdet/tutorials/finetune/)
- [ONNX Export](https://binitai.github.io/visdet/tutorials/pytorch2onnx/)
- [TensorRT Conversion](https://binitai.github.io/visdet/tutorials/onnx2tensorrt/)
- [Weight Initialization](https://binitai.github.io/visdet/tutorials/init_cfg/)

</details>

---

## 🎯 Model Zoo

Pre-trained models and benchmarks available in the [Model Zoo](https://binitai.github.io/visdet/model-zoo/).

---

## 🤝 Contributing

We welcome contributions! Please see the [Contributing Guide](https://binitai.github.io/visdet/development/contributing/) for details.

- 🐛 [Report Issues](https://github.com/BinItAI/visdet/issues/new)
- 💡 [Request Features](https://github.com/BinItAI/visdet/issues/new)
- 🔧 [Submit Pull Requests](https://github.com/BinItAI/visdet/pulls)

---

## 👥 Contributors

We appreciate all contributions to visdet! Special thanks to our contributors:

| Contributor | Contributions |
|---|---|
| [GeorgePearse](https://github.com/GeorgePearse) | 118 |

[View all contributors →](https://github.com/BinItAI/visdet/graphs/contributors)

---

## 📄 License

This project is released under the [Apache 2.0 License](https://binitai.github.io/visdet/about/license/).

---

## 🔗 Related Projects

visdet is part of a rich ecosystem of object detection frameworks. Here's how visdet compares to other notable projects:

### [MMDetection](https://github.com/open-mmlab/mmdetection)
The original framework that visdet is based on. A comprehensive object detection toolbox with modular design, supporting 40+ architectures including detection, instance segmentation, and panoptic segmentation. Part of the OpenMMLab project with extensive model zoo and state-of-the-art implementations.

**Choose MMDetection if:** You need the original framework with all dependencies or prefer the traditional MMDetection workflow with MMCV and MMEngine as separate packages.

### [Detectron2](https://github.com/facebookresearch/detectron2)
Facebook AI Research's production-grade detection library. Supports object detection, instance segmentation, panoptic segmentation, DensePose, and more. Known for excellent performance and deployment flexibility with TorchScript/Caffe2 export. The foundation for many research projects.

**Choose Detectron2 if:** You need production deployment, Facebook ecosystem integration, or prefer Facebook's design philosophy and tooling.

### [detrex](https://github.com/IDEA-Research/detrex)
A specialized research platform built on top of Detectron2, focused specifically on Transformer-based detection algorithms (DETR variants). Provides unified modular design for 20+ Transformer models including DETR, Deformable-DETR, DINO, and MaskDINO. Uses LazyConfig for flexible configuration.

**Choose detrex if:** You're doing cutting-edge Transformer-based detection research or want to experiment with DETR variants.

### visdet (this project)
A streamlined fork of MMDetection with integrated dependencies, no CUDA compilation requirements, and modern training techniques from fast.ai and LLM fine-tuning.

**Choose visdet if:** You want simplified installation (no CUDA compilation), pure Python/PyTorch implementation, and modern training techniques like progressive resizing, 1cycle schedules, and learning rate finders.

---

## 🙏 Acknowledgements

visdet is built on top of the excellent [MMDetection](https://github.com/open-mmlab/mmdetection) framework from OpenMMLab. We are grateful to all contributors of the original project.

---

<div align="center">
  <sub>Built with ❤️ by the <a href="https://www.visia.ai/">Visia</a> ML Engineering team</sub>
</div>
