<div align="center">
  <img src="assets/visia-logo.svg" width="600"/>
  <div>&nbsp;</div>

[📘Documentation](https://binitai.github.io/visdet/) |
[🛠️Installation](https://binitai.github.io/visdet/getting-started/installation/) |
[👀Model Zoo](https://binitai.github.io/visdet/model-zoo/) |
[🆕Update News](https://binitai.github.io/visdet/about/changelog/) |
[🤔Reporting Issues](https://github.com/BinItAI/visdet/issues)

</div>
## Introduction

MMDetection is an open source object detection toolbox based on PyTorch.

The master branch works with **PyTorch 1.5+**.

<img src="https://user-images.githubusercontent.com/12907710/137271636-56ba1cd2-b110-4812-8221-b4c120320aa9.png"/>

<details open>
<summary>Major features</summary>

- **Modular Design**

  We decompose the detection framework into different components and one can easily construct a customized object detection framework by combining different modules.

- **Support of multiple frameworks out of box**

  The toolbox directly supports popular and contemporary detection frameworks, *e.g.* Faster RCNN, Mask RCNN, RetinaNet, etc.

- **High efficiency**

  All basic bbox and mask operations run on GPUs. The training speed is faster than or comparable to other codebases, including [Detectron2](https://github.com/facebookresearch/detectron2), [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and [SimpleDet](https://github.com/TuSimple/simpledet).

- **State of the art**

  The toolbox stems from the codebase developed by the *MMDet* team, who won [COCO Detection Challenge](http://cocodataset.org/#detection-leaderboard) in 2018, and we keep pushing it forward.

</details>


## What's New

### 💎 Stable version

**2.28.2** was released in 27/2/2023:

- Fixed some known documentation, configuration and linking error issues

Please refer to the [changelog](about/changelog.md) for details and release history.

### 🌟 Preview of 3.x version

#### Highlight

We are excited to announce our latest work on real-time object recognition tasks, **RTMDet**, a family of fully convolutional single-stage detectors. RTMDet not only achieves the best parameter-accuracy trade-off on object detection from tiny to extra-large model sizes but also obtains new state-of-the-art performance on instance segmentation and rotated object detection tasks. Details can be found in the [technical report](https://arxiv.org/abs/2212.07784). Pre-trained models are [here](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/rtmdet).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/real-time-instance-segmentation-on-mscoco)](https://paperswithcode.com/sota/real-time-instance-segmentation-on-mscoco?p=rtmdet-an-empirical-study-of-designing-real)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/object-detection-in-aerial-images-on-dota-1)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-dota-1?p=rtmdet-an-empirical-study-of-designing-real)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rtmdet-an-empirical-study-of-designing-real/object-detection-in-aerial-images-on-hrsc2016)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-hrsc2016?p=rtmdet-an-empirical-study-of-designing-real)

| Task                     | Dataset | AP                                   | FPS(TRT FP16 BS1 3090) |
| ------------------------ | ------- | ------------------------------------ | ---------------------- |
| Object Detection         | COCO    | 52.8                                 | 322                    |
| Instance Segmentation    | COCO    | 44.6                                 | 188                    |
| Rotated Object Detection | DOTA    | 78.9(single-scale)/81.3(multi-scale) | 121                    |

<div align=center>
<img src="https://user-images.githubusercontent.com/12907710/208044554-1e8de6b5-48d8-44e4-a7b5-75076c7ebb71.png"/>
</div>

A brand new version of **MMDetection v3.0.0rc6** was released in 27/2/2023:

- Support [Boxinst](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/boxinst), [Objects365 Dataset](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/objects365), and [Separated and Occluded COCO metric](https://github.com/open-mmlab/mmdetection/tree/3.x/docs/en/user_guides/useful_tools.md#coco-separated--occluded-mask-metric)
- Support [ConvNeXt-V2](https://github.com/open-mmlab/mmdetection/tree/3.x/projects/ConvNeXt-V2), [DiffusionDet](https://github.com/open-mmlab/mmdetection/tree/3.x/projects/DiffusionDet), and inference of [EfficientDet](https://github.com/open-mmlab/mmdetection/tree/3.x/projects/EfficientDet) and [Detic](https://github.com/open-mmlab/mmdetection/tree/3.x/projects/Detic) in `Projects`
- Refactor [DETR](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/detr) series and support [Conditional-DETR](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/conditional_detr), [DAB-DETR](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/dab_detr), and [DINO](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/dino)
- Support DetInferencer, Test Time Augmentation, and auto import modules from registry
- Support RTMDet-Ins ONNXRuntime and TensorRT [deployment](https://github.com/open-mmlab/mmdetection/tree/3.x/configs/rtmdet/README.md#deployment-tutorial)
- Support [calculating FLOPs of detectors](https://github.com/open-mmlab/mmdetection/tree/3.x/docs/en/user_guides/useful_tools.md#Model-Complexity)

Find more new features in [3.x branch](https://github.com/open-mmlab/mmdetection/tree/3.x). Issues and PRs are welcome!

## Installation

Please refer to [Installation](getting-started/installation.md) for installation instructions.

## Getting Started

Please see the [Getting Started](getting-started/quick-start.md) guide for the basic usage of MMDetection. Tutorials available:

- [learn about configs](tutorials/config.md)
- [customize datasets](tutorials/customize_dataset.md)
- [customize data pipelines](tutorials/data_pipeline.md)
- [customize models](tutorials/customize_models.md)
- [customize runtime settings](tutorials/customize_runtime.md)
- [customize losses](tutorials/customize_losses.md)
- [finetuning models](tutorials/finetune.md)
- [export a model to ONNX](tutorials/pytorch2onnx.md)
- [export ONNX to TRT](tutorials/onnx2tensorrt.md)
- [weight initialization](tutorials/init_cfg.md)
- [how to xxx](tutorials/how_to.md)

## Overview of Benchmark and Model Zoo

Results and models are available in the [model zoo](model-zoo.md).

<div align="center">
  <b>Architectures</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Object Detection</b>
      </td>
      <td>
        <b>Instance Segmentation</b>
      </td>
      <td>
        <b>Panoptic Segmentation</b>
      </td>
      <td>
        <b>Other</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li>Fast R-CNN (ICCV'2015)</li>
            <li>Faster R-CNN (NeurIPS'2015)</li>
            <li>RPN (NeurIPS'2015)</li>
            <li>SSD (ECCV'2016)</li>
            <li>RetinaNet (ICCV'2017)</li>
            <li>Cascade R-CNN (CVPR'2018)</li>
            <li>YOLOv3 (ArXiv'2018)</li>
            <li>CornerNet (ECCV'2018)</li>
            <li>Grid R-CNN (CVPR'2019)</li>
            <li>Guided Anchoring (CVPR'2019)</li>
            <li>FSAF (CVPR'2019)</li>
            <li>CenterNet (ArXiv'2019)</li>
            <li>Libra R-CNN (CVPR'2019)</li>
            <li>TridentNet (ICCV'2019)</li>
            <li>FCOS (ICCV'2019)</li>
            <li>RepPoints (ICCV'2019)</li>
            <li>FreeAnchor (NeurIPS'2019)</li>
            <li>CascadeRPN (NeurIPS'2019)</li>
            <li>Foveabox (TIP'2020)</li>
            <li>Double-Head R-CNN (CVPR'2020)</li>
            <li>ATSS (CVPR'2020)</li>
            <li>NAS-FCOS (CVPR'2020)</li>
            <li>CentripetalNet (CVPR'2020)</li>
            <li>AutoAssign (ArXiv'2020)</li>
            <li>Side-Aware Boundary Localization (ECCV'2020)</li>
            <li>Dynamic R-CNN (ECCV'2020)</li>
            <li>DETR (ECCV'2020)</li>
            <li>PAA (ECCV'2020)</li>
            <li>VarifocalNet (CVPR'2021)</li>
            <li>Sparse R-CNN (CVPR'2021)</li>
            <li>YOLOF (CVPR'2021)</li>
            <li>YOLOX (ArXiv'2021)</li>
            <li>Deformable DETR (ICLR'2021)</li>
            <li>TOOD (ICCV'2021)</li>
            <li>DDOD (ACM MM'2021)</li>
      </ul>
      </td>
      <td>
        <ul>
          <li>Mask R-CNN (ICCV'2017)</li>
          <li>Cascade Mask R-CNN (CVPR'2018)</li>
          <li>Mask Scoring R-CNN (CVPR'2019)</li>
          <li>Hybrid Task Cascade (CVPR'2019)</li>
          <li>YOLACT (ICCV'2019)</li>
          <li>InstaBoost (ICCV'2019)</li>
          <li>SOLO (ECCV'2020)</li>
          <li>PointRend (CVPR'2020)</li>
          <li>DetectoRS (CVPR'2021)</li>
          <li>SOLOv2 (NeurIPS'2020)</li>
          <li>SCNet (AAAI'2021)</li>
          <li>QueryInst (ICCV'2021)</li>
          <li>Mask2Former (CVPR'2022)</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Panoptic FPN (CVPR'2019)</li>
          <li>MaskFormer (NeurIPS'2021)</li>
          <li>Mask2Former (CVPR'2022)</li>
        </ul>
      </td>
      <td>
        </ul>
          <li><b>Contrastive Learning</b></li>
        <ul>
        <ul>
          <li>SwAV (NeurIPS'2020)</li>
          <li>MoCo (CVPR'2020)</li>
          <li>MoCov2 (ArXiv'2020)</li>
        </ul>
        </ul>
        </ul>
          <li><b>Distillation</b></li>
        <ul>
        <ul>
          <li>Localization Distillation (CVPR'2022)</li>
          <li>Label Assignment Distillation (WACV'2022)</li>
        </ul>
        </ul>
      </ul>
        <li><b>Receptive Field Search</b></li>
      <ul>
        <ul>
          <li>RF-Next (TPAMI'2022)</li>
        </ul>
        </ul>
      </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

<div align="center">
  <b>Components</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Backbones</b>
      </td>
      <td>
        <b>Necks</b>
      </td>
      <td>
        <b>Loss</b>
      </td>
      <td>
        <b>Common</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
      <ul>
        <li>VGG (ICLR'2015)</li>
        <li>ResNet (CVPR'2016)</li>
        <li>ResNeXt (CVPR'2017)</li>
        <li>MobileNetV2 (CVPR'2018)</li>
        <li>HRNet (CVPR'2019)</li>
        <li>Generalized Attention (ICCV'2019)</li>
        <li>GCNet (ICCVW'2019)</li>
        <li>Res2Net (TPAMI'2020)</li>
        <li>RegNet (CVPR'2020)</li>
        <li>ResNeSt (CVPRW'2022)</li>
        <li>PVT (ICCV'2021)</li>
        <li>Swin (ICCV'2021)</li>
        <li>PVTv2 (CVMJ'2022)</li>
        <li>ResNet strikes back (NeurIPSW'2021)</li>
        <li>EfficientNet (ICML'2019)</li>
        <li>ConvNeXt (CVPR'2022)</li>
      </ul>
      </td>
      <td>
      <ul>
        <li>PAFPN (CVPR'2018)</li>
        <li>NAS-FPN (CVPR'2019)</li>
        <li>CARAFE (ICCV'2019)</li>
        <li>FPG (ArXiv'2020)</li>
        <li>GRoIE (ICPR'2020)</li>
        <li>DyHead (CVPR'2021)</li>
      </ul>
      </td>
      <td>
        <ul>
          <li>GHM (AAAI'2019)</li>
          <li>Generalized Focal Loss (NeurIPS'2020)</li>
          <li>Seasaw Loss (CVPR'2021)</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>OHEM (CVPR'2016)</li>
          <li>Group Normalization (ECCV'2018)</li>
          <li>DCN (ICCV'2017)</li>
          <li>DCNv2 (CVPR'2019)</li>
          <li>Weight Standardization (ArXiv'2019)</li>
          <li>Prime Sample Attention (CVPR'2020)</li>
          <li>Strong Baselines (CVPR'2021)</li>
          <li>Resnet strikes back (NeurIPSW'2021)</li>
          <li>RF-Next (TPAMI'2022)</li>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

See the [model zoo](model-zoo.md) for a complete list of supported methods.

## FAQ

Please refer to the documentation for frequently asked questions.

## Contributing

We appreciate all contributions to improve MMDetection. Ongoing projects can be found in our [GitHub Projects](https://github.com/open-mmlab/mmdetection/projects). Welcome community users to participate in these projects. Please refer to the [contributing guide](development/contributing.md) for the contributing guideline.

## Acknowledgement

MMDetection is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

## License

This project is released under the [Apache 2.0 license](about/license.md).
