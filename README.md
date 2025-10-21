<div align="center">
  <img src="resources/visia-logo-white.svg" width="600"/>
  <div>&nbsp;</div>

[![PyPI](https://img.shields.io/pypi/v/visdet)](https://pypi.org/project/visdet)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://binitai.github.io/visdet/)
[![PyPI Publishing](https://github.com/BinItAI/visdet/actions/workflows/publish-pypi.yml/badge.svg)](https://github.com/BinItAI/visdet/actions/workflows/publish-pypi.yml)
[![license](https://img.shields.io/github/license/BinItAI/visdet.svg)](https://github.com/BinItAI/visdet/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/BinItAI/visdet.svg)](https://github.com/BinItAI/visdet/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/BinItAI/visdet.svg)](https://github.com/BinItAI/visdet/issues)

[📘Documentation](https://binitai.github.io/visdet/) |
[🛠️Installation](https://binitai.github.io/visdet/getting-started/installation/) |
[👀Model Zoo](https://binitai.github.io/visdet/model-zoo/) |
[🆕Update News](https://binitai.github.io/visdet/about/changelog/) |
[🚀Ongoing Projects](https://github.com/open-mmlab/mmdetection/projects) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmdetection/issues/new/choose)

</div>

<div align="center">

English

</div>

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218352562-cdded397-b0f3-4ca1-b8dd-a60df8dca75b.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.gg/raweFPmdzG" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
</div>

> **Note**: This is a fork of the MMDetection library, customized for internal use at [Visia](https://www.visia.ai/). The original project can be found at https://github.com/open-mmlab/mmdetection
>
> Maintained by the Visia ML Engineering team

## Key Improvements in visdet

**Simplified Installation & Dependencies**
- **Integrated Dependencies**: MMCV and MMEngine are bundled directly into the package, eliminating complex multi-package dependency management
- **No Custom CUDA Required**: All custom CUDA operations have been removed, making installation straightforward with just `pip install visdet`
- **Python-Only Implementation**: Pure Python/PyTorch implementation means faster installation and better compatibility across different environments

This makes visdet significantly easier to install and deploy compared to the original MMDetection, which required careful coordination of multiple packages and custom CUDA compilation.
