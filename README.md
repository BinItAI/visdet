<div align="center">
  <h1>visdet</h1>
  <p><strong>The only Object Detection research framework with sane usability</strong></p>

  <p>
    <em>A modern, actively maintained fork of <a href="https://github.com/open-mmlab/mmdetection">MMDetection</a></em>
  </p>

  <div>&nbsp;</div>

[![License](https://img.shields.io/github/license/BinItAI/visdet.svg)](https://github.com/BinItAI/visdet/blob/master/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

## About

**visdet** is a modern fork of the groundbreaking [MMDetection](https://github.com/open-mmlab/mmdetection) object detection framework. MMDetection was - and remains - one of the most comprehensive and influential detection toolboxes ever created. The MMDetection team won the **COCO Detection Challenge in 2018**, and their work has been cited thousands of times, setting the standard for object detection research.

However, MMDetection is now **archived and no longer actively maintained**. The original codebase, while powerful, relied on older Python packaging standards and tooling that made it increasingly difficult to use and maintain as the ecosystem evolved.

**visdet** continues this legacy with:
- ✅ **Active maintenance** and regular updates
- ✅ **Modern Python tooling** (pyproject.toml, uv, pre-commit)
- ✅ **Sane defaults** that actually work out of the box
- ✅ **Better developer experience** with clear error messages and documentation
- ✅ **All the power** of the original MMDetection, modernized

<img src="https://user-images.githubusercontent.com/12907710/137271636-56ba1cd2-b110-4812-8221-b4c120320aa9.png"/>

---

## Why visdet?

### 🔧 Modern Python Ecosystem

| MMDetection (Legacy) | visdet (Modern) |
|---------------------|-----------------|
| setup.py | **pyproject.toml** with hatchling |
| pip | **uv** for blazing-fast installs |
| Manual formatting | **pre-commit hooks** with ruff |
| Readthedocs | **MkDocs Material** theme |
| Complex configs | **Sane defaults** |

### 🎯 Sane Usability

What makes visdet "sane"?

- **Works out of the box**: No more cryptic config errors or missing dependencies
- **Modern documentation**: Clean, searchable MkDocs instead of legacy Sphinx
- **Clear error messages**: Know exactly what went wrong and how to fix it
- **Active community**: Get help, report issues, contribute improvements
- **Modular architecture**: Clean separation with `visdet`, `viscv`, and `visengine` packages
- **Better type hints**: Fully typed Python for better IDE support and fewer bugs

### 🚀 All the Power, None of the Pain

visdet inherits MMDetection's comprehensive model zoo:
- **60+ detection algorithms** from Fast R-CNN to DETR
- **State-of-the-art performance** on COCO, LVIS, and other benchmarks
- **Modular design** for easy customization
- **Production-ready** with ONNX/TensorRT export

But with modern conveniences:
- Install with `uv pip install` instead of wrestling with mmcv versions
- Use standard Python packaging instead of custom MIM installers
- Get clear error messages instead of inscrutable tracebacks
- Read beautiful documentation instead of outdated wikis

---

## What's New in visdet

### 🎉 Recent Updates

**Modern Build System** (2024-10)
- Migrated to pyproject.toml and uv for dependency management
- Added pre-commit hooks for consistent code quality
- Implemented GitHub Actions for CI/CD

**Better Documentation** (2024-10)
- MkDocs with Material theme for modern, searchable docs
- Auto-generated API documentation
- Dark mode support

**Core Improvements** (2024-10)
- Added three core packages for Swin Mask R-CNN implementation
- GitHub Actions workflow for deployment
- Better type hints and code organization

---

## Installation

### Prerequisites

- Python 3.10, 3.11, or 3.12
- PyTorch 1.5 or higher
- CUDA (for GPU support)

### Quick Install

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a new environment and install visdet
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### Verify Installation

```python
import mmdet
print(mmdet.__version__)
```

For detailed installation instructions, see our [Installation Guide](docs/en/get_started.md).

---

## Getting Started

### Basic Usage

```python
from mmdet.apis import init_detector, inference_detector

# Load config and checkpoint
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco.pth'

# Initialize model
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Run inference
img = 'demo/demo.jpg'
result = inference_detector(model, img)
```

### Training a Model

```bash
python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py
```

### More Tutorials

- [Training with your own dataset](docs/en/2_new_data_model.md)
- [Customizing models](docs/en/tutorials/customize_models.md)
- [Exporting to ONNX](docs/en/tutorials/pytorch2onnx.md)
- [Fine-tuning pretrained models](docs/en/tutorials/finetune.md)

---

## Model Zoo

visdet supports **60+ detection algorithms** out of the box:

<details>
<summary><b>Object Detection</b> (click to expand)</summary>

- Fast R-CNN, Faster R-CNN, RPN
- SSD, RetinaNet
- Cascade R-CNN
- YOLOv3, YOLOX
- CornerNet, CenterNet
- FCOS, ATSS
- DETR, Deformable DETR
- And many more...

</details>

<details>
<summary><b>Instance Segmentation</b></summary>

- Mask R-CNN, Cascade Mask R-CNN
- Hybrid Task Cascade (HTC)
- YOLACT, SOLOv2
- PointRend
- Mask2Former
- QueryInst
- And more...

</details>

<details>
<summary><b>Panoptic Segmentation</b></summary>

- Panoptic FPN
- MaskFormer
- Mask2Former

</details>

For complete model zoo with pretrained weights, see [Model Zoo](docs/en/model_zoo.md).

---

## Key Features

### 🧩 Modular Design

visdet decomposes detection into reusable components:

```
Detection Framework = Backbone + Neck + Head + Loss
```

Mix and match components to create custom detectors:

- **Backbones**: ResNet, ResNeXt, Swin, PVT, ConvNeXt, and more
- **Necks**: FPN, PAFPN, BiFPN, NAS-FPN
- **Heads**: RPNHead, RetinaHead, FCOSHead, DETR
- **Losses**: CrossEntropy, FocalLoss, GIoU, DIoU

### ⚡ High Performance

- GPU-accelerated bbox and mask operations
- Faster than or comparable to Detectron2 and other frameworks
- Multi-GPU training support
- Mixed precision training (FP16)

### 🎓 Research-Friendly

- Easy to implement new algorithms
- Comprehensive tutorials and documentation
- Active community for support
- Regular updates with latest research

---

## Documentation

- 📘 [Full Documentation](https://binitai.github.io/visdet/) (coming soon with MkDocs)
- 🛠️ [Installation Guide](docs/en/get_started.md)
- 👀 [Model Zoo](docs/en/model_zoo.md)
- 📚 [Tutorials](docs/en/tutorials/)
- 🤔 [FAQ](docs/en/faq.md)

---

## Contributing

We welcome contributions! visdet is built by the community, for the community.

- 🐛 [Report Issues](https://github.com/BinItAI/visdet/issues/new/choose)
- 💡 [Feature Requests](https://github.com/BinItAI/visdet/issues/new/choose)
- 🔧 [Pull Requests](https://github.com/BinItAI/visdet/pulls)
- 📖 [Contributing Guide](.github/CONTRIBUTING.md)

---

## Acknowledgements

### MMDetection Legacy

visdet stands on the shoulders of giants. We are deeply grateful to the **OpenMMLab team** and the original MMDetection contributors for creating such a groundbreaking framework. Their work:

- Won the **COCO Detection Challenge 2018**
- Has been cited thousands of times in research papers
- Set the standard for object detection toolboxes
- Enabled countless research breakthroughs

While MMDetection is no longer actively maintained, its impact on the computer vision community cannot be overstated. visdet aims to honor this legacy by continuing to provide a world-class detection framework with modern tooling and active support.

### Citation

If you use visdet in your research, please cite both visdet and the original MMDetection:

```bibtex
@misc{visdet2024,
  title={visdet: Modern Object Detection Framework},
  author={visdet Contributors},
  howpublished={\url{https://github.com/BinItAI/visdet}},
  year={2024}
}

@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal = {arXiv preprint arXiv:1906.07155},
  year    = {2019}
}
```

---

## License

This project is released under the [Apache 2.0 license](LICENSE), same as the original MMDetection.

---

## Related Projects

### OpenMMLab Ecosystem

While we've forked from MMDetection, we acknowledge the excellent work across the entire OpenMMLab ecosystem:

- [MMEngine](https://github.com/open-mmlab/mmengine) - Foundational library for training
- [MMCV](https://github.com/open-mmlab/mmcv) - Computer vision primitives
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) - Semantic segmentation
- [MMPose](https://github.com/open-mmlab/mmpose) - Pose estimation
- [MMTracking](https://github.com/open-mmlab/mmtracking) - Video object tracking

### Our Ecosystem

- **visdet**: Object detection (this repo)
- **viscv**: Computer vision utilities
- **visengine**: Training engine and infrastructure

---

<div align="center">

**Built with ❤️ by the visdet community**

[⭐ Star us on GitHub](https://github.com/BinItAI/visdet) | [📖 Read the Docs](https://binitai.github.io/visdet/) | [💬 Join Discussions](https://github.com/BinItAI/visdet/discussions)

</div>
