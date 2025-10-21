<div align="center">
  <img src="resources/visia-logo-white.svg" width="600"/>
  <div>&nbsp;</div>

  <!-- Core Badges -->
  <p>
    <a href="https://pypi.org/project/visdet"><img src="https://img.shields.io/pypi/v/visdet?style=flat-square&color=blue" alt="PyPI"/></a>
    <a href="https://binitai.github.io/visdet/"><img src="https://img.shields.io/badge/docs-latest-brightgreen?style=flat-square" alt="Documentation"/></a>
    <a href="https://github.com/BinItAI/visdet/actions/workflows/publish-pypi.yml"><img src="https://img.shields.io/github/actions/workflow/status/BinItAI/visdet/publish-pypi.yml?style=flat-square&label=build" alt="Build Status"/></a>
    <a href="https://github.com/BinItAI/visdet/blob/master/LICENSE"><img src="https://img.shields.io/github/license/BinItAI/visdet.svg?style=flat-square&color=blue" alt="License"/></a>
  </p>

  <!-- Issue Badges -->
  <p>
    <a href="https://github.com/BinItAI/visdet/issues"><img src="https://img.shields.io/github/issues/BinItAI/visdet?style=flat-square&color=orange" alt="Open Issues"/></a>
    <a href="https://github.com/BinItAI/visdet/issues"><img src="https://img.shields.io/github/issues-closed/BinItAI/visdet?style=flat-square&color=green" alt="Closed Issues"/></a>
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
- **Integrated Dependencies**: MMCV and MMEngine are bundled directly into the package, eliminating complex multi-package dependency management
- **No Custom CUDA Required**: All custom CUDA operations have been removed, making installation straightforward with just `pip install visdet`
- **Python-Only Implementation**: Pure Python/PyTorch implementation means faster installation and better compatibility across different environments

This makes visdet significantly easier to install and deploy compared to the original MMDetection, which required careful coordination of multiple packages and custom CUDA compilation.

---

## 🚀 Quick Start

### Installation

```bash
# Using uv (recommended)
uv pip install visdet

# Or using pip
pip install visdet
```

For detailed installation instructions, see the [Installation Guide](https://binitai.github.io/visdet/getting-started/installation/).

### Basic Usage

```python
from visdet import ...
# Your code here
```

For tutorials and examples, visit the [Documentation](https://binitai.github.io/visdet/).

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

## 📄 License

This project is released under the [Apache 2.0 License](https://binitai.github.io/visdet/about/license/).

---

## 🙏 Acknowledgements

visdet is built on top of the excellent [MMDetection](https://github.com/open-mmlab/mmdetection) framework from OpenMMLab. We are grateful to all contributors of the original project.

---

<div align="center">
  <sub>Built with ❤️ by the <a href="https://www.visia.ai/">Visia</a> ML Engineering team</sub>
</div>
