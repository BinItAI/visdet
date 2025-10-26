# Model Zoo

This page provides a comprehensive overview of all available models in visdet.
Each model family includes performance metrics, configuration files, and pre-trained weights.

## Overview

visdet includes **91** model families covering various object detection and instance segmentation architectures.

## Table of Contents

- [Two-Stage Detectors](#twostagedetectors)
- [One-Stage Detectors](#onestagedetectors)
- [YOLO Series](#yoloseries)
- [Transformer-Based](#transformerbased)
- [Instance Segmentation](#instancesegmentation)
- [Panoptic Segmentation](#panopticsegmentation)
- [Specialized Backbones](#specializedbackbones)

## Two-Stage Detectors

### Cascade R-CNN

**Paper:** [Cascade R-CNN](https://arxiv.org/abs/1906.09756)

In object detection, the intersection over union (IoU) threshold is frequently used to define positives/negatives. The threshold used to train a detector defines its quality. While the commonly used t...

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP |                                                            Config                                                            |                                                                                                                                                                             Download                                                                                                                                                                              |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :--------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|    R-50-FPN     |  caffe  |   1x    |   4.2    |                |  40.4  |  [config]()  |   [model]( \| [log](   |
|    R-50-FPN     | pytorch |   1x    |   4.4    |      16.1      |  40.3  |     [config]()     |                          [model]( \| [log](                          |
|    R-50-FPN     | pytorch |   20e   |    -     |       -        |  41.0  |    [config]()     |             [model]( \| [log](              |
|    R-101-FPN    |  caffe  |   1x    |   6.2    |                |  42.3  | [config]()  | [model]( \| [log]( |
|    R-101-FPN    | pytorch |   1x    |   6.4    |      13.5      |  42.0  |    [config]()     |                        [model]( \| [log](                        |
|    R-101-FPN    | pytorch |   20e   |    -     |       -        |  42.5  |    [config]()    |           [model]( \| [log](           |
| X-101-32x4d-FPN | pytorch |   1x    |   7.6    |      10.9      |  43.7  | [config]()  |            [model]( \| [log](            |
| X-101-32x4d-FPN | pytorch |   20e   |   7.6    |                |  43.7  | [config]() |      [model]( \| [log](       |
| X-101-64x4d-FPN | pytorch |   1x    |   10.7   |                |  44.7  | [config]()  |        [model]( \| [log](         |
| X-101-64x4d-FPN | pytorch |   20e   |   10.7   |                |  44.5  | [config]() |      [model]( \| [log](       |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/cascade_rcnn/README.md)

---

### Faster R-CNN

**Paper:** [Faster R-CNN](https://arxiv.org/abs/1506.01497)

State-of-the-art object detection networks depend on region proposal algorithms to hypothesize object locations. Advances like SPPnet and Fast R-CNN have reduced the running time of these detection ne...

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP |                                                          Config                                                           |                                                                                                                                                                          Download                                                                                                                                                                           |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|     R-50-C4     |  caffe  |   1x    |    -     |       -        |  35.6  |  [config]()  |            [model]( \| [log](             |
|    R-50-DC5     |  caffe  |   1x    |    -     |       -        |  37.2  | [config]()  |          [model]( \| [log](           |
|    R-50-FPN     |  caffe  |   1x    |   3.8    |                |  37.8  | [config]()  |   [model]( \| [log](   |
|    R-50-FPN     | pytorch |   1x    |   4.0    |      21.4      |  37.4  |    [config]()     |                          [model]( \| [log](                          |
| R-50-FPN (FP16) | pytorch |   1x    |   3.4    |      28.8      |  37.5  |  [config]()  |                       [model]( \| [log](                       |
|    R-50-FPN     | pytorch |   2x    |    -     |       -        |  38.4  |    [config]()     |               [model]( \| [log](               |
|    R-101-FPN    |  caffe  |   1x    |   5.7    |                |  39.8  | [config]() | [model]( \| [log]( |
|    R-101-FPN    | pytorch |   1x    |   6.0    |      15.6      |  39.4  |    [config]()    |                        [model]( \| [log](                        |
|    R-101-FPN    | pytorch |   2x    |    -     |       -        |  39.8  |    [config]()    |             [model]( \| [log](             |
| X-101-32x4d-FPN | pytorch |   1x    |   7.2    |      13.8      |  41.2  | [config]() |            [model]( \| [log](            |
| X-101-32x4d-FPN | pytorch |   2x    |    -     |       -        |  41.2  | [config]() | [model]( \| [log]( |
| X-101-64x4d-FPN | pytorch |   1x    |   10.3   |      9.4       |  42.1  | [config]() |            [model]( \| [log](            |
| X-101-64x4d-FPN | pytorch |   2x    |    -     |       -        |  41.6  | [config]() |        [model]( \| [log](         |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/faster_rcnn/README.md)

---

### Grid R-CNN

**Paper:** [Grid R-CNN](https://arxiv.org/abs/1811.12030)

This paper proposes a novel object detection framework named Grid R-CNN, which adopts a grid guided localization mechanism for accurate object detection. Different from the traditional regression base...

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | box AP |                                                            Config                                                             |                                                                                                                                                                         Download                                                                                                                                                                          |
| :---------: | :-----: | :------: | :------------: | :----: | :---------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|    R-50     |   2x    |   5.1    |      15.0      |  40.4  |    [config]()     |               [model]( \| [log](               |
|    R-101    |   2x    |   7.0    |      12.6      |  41.5  |    [config]()    |             [model]( \| [log](             |
| X-101-32x4d |   2x    |   8.3    |      10.8      |  42.9  | [config]() | [model]( \| [log]( |
| X-101-64x4d |   2x    |   11.3   |      7.7       |  43.0  | [config]() | [model]( \| [log]( |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/grid_rcnn/README.md)

---

### HTC

**Paper:** [HTC](https://arxiv.org/abs/1901.07518)

Cascade is a classic yet powerful architecture that has boosted performance on various tasks. However, how to introduce cascade to instance segmentation remains an open question. A simple combination ...

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP |                                                     Config                                                      |                                                                                                                                                   Download                                                                                                                                                    |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :-------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|    R-50-FPN     | pytorch |   1x    |   8.2    |      5.8       |  42.3  |  37.4   |       [config]()        |                           [model]( \| [log](                           |
|    R-50-FPN     | pytorch |   20e   |   8.2    |       -        |  43.3  |  38.3   |       [config]()       |                         [model]( \| [log](                         |
|    R-101-FPN    | pytorch |   20e   |   10.2   |      5.5       |  44.8  |  39.6   |      [config]()       |                       [model]( \| [log](                       |
| X-101-32x4d-FPN | pytorch |   20e   |   11.4   |      5.0       |  46.1  |  40.5   | [config]() | [model]( \| [log]( |
| X-101-64x4d-FPN | pytorch |   20e   |   14.5   |      4.4       |  47.0  |  41.4   | [config]() | [model]( \| [log]( |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/htc/README.md)

---

### Mask R-CNN

**Paper:** [Mask R-CNN](https://arxiv.org/abs/1703.06870)

We present a conceptually simple, flexible, and general framework for object instance segmentation. Our approach efficiently detects objects in an image while simultaneously generating a high-quality ...

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP |                                                        Config                                                         |                                                                                                                                                                            Download                                                                                                                                                                             |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :-------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|    R-50-FPN     |  caffe  |   1x    |   4.3    |                |  38.0  |  34.4   | [config]()  |   [model]( \| [log](    |
|    R-50-FPN     | pytorch |   1x    |   4.4    |      16.1      |  38.2  |  34.7   |    [config]()     |                                  [model]( \| [log](                                  |
| R-50-FPN (FP16) | pytorch |   1x    |   3.6    |      24.1      |  38.1  |  34.7   |  [config]()  |                             [model]( \| [log](                             |
|    R-50-FPN     | pytorch |   2x    |    -     |       -        |  39.2  |  35.4   |    [config]()     |               [model]( \| [log](               |
|    R-101-FPN    |  caffe  |   1x    |          |                |  40.4  |  36.4   | [config]() |                [model]( \| [log](                 |
|    R-101-FPN    | pytorch |   1x    |   6.4    |      13.5      |  40.0  |  36.1   |    [config]()    |                                [model]( \| [log](                                |
|    R-101-FPN    | pytorch |   2x    |    -     |       -        |  40.8  |  36.6   |    [config]()    |             [model]( \| [log](             |
| X-101-32x4d-FPN | pytorch |   1x    |   7.6    |      11.3      |  41.9  |  37.5   | [config]() |                    [model]( \| [log](                    |
| X-101-32x4d-FPN | pytorch |   2x    |    -     |       -        |  42.2  |  37.8   | [config]() | [model]( \| [log]( |
| X-101-64x4d-FPN | pytorch |   1x    |   10.7   |      8.0       |  42.8  |  38.4   | [config]() |                    [model]( \| [log](                    |
| X-101-64x4d-FPN | pytorch |   2x    |    -     |       -        |  42.7  |  38.1   | [config]() |                [model]( \| [log](                 |
| X-101-32x8d-FPN | pytorch |   1x    |   10.6   |       -        |  42.8  |  38.3   | [config]() |                [model]( \| [log](                 |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/mask_rcnn/README.md)

---

### MS R-CNN

**Paper:** [MS R-CNN](https://arxiv.org/abs/1903.00241)

Letting a deep network be aware of the quality of its own predictions is an interesting yet important problem. In the task of instance segmentation, the confidence of instance classification is used a...

|   Backbone   |  style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP |                                                      Config                                                       |                                                                                                                                                                      Download                                                                                                                                                                       |
| :----------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :---------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50-FPN   |  caffe  |   1x    |   4.5    |                |  38.2  |  36.0   | [config]()  |                  [model]( \| [log](                   |
|   R-50-FPN   |  caffe  |   2x    |    -     |       -        |  38.8  |  36.3   | [config]()  |   [model]( \| [log](   |
|  R-101-FPN   |  caffe  |   1x    |   6.5    |                |  40.4  |  37.6   | [config]() | [model]( \| [log]( |
|  R-101-FPN   |  caffe  |   2x    |    -     |       -        |  41.1  |  38.1   | [config]() | [model]( \| [log]( |
| R-X101-32x4d | pytorch |   2x    |   7.9    |      11.0      |  41.8  |  38.7   | [config]() |                    [model]( \| [log](                    |
| R-X101-64x4d | pytorch |   1x    |   11.0   |      8.0       |  43.0  |  39.5   | [config]() |                    [model]( \| [log](                    |
| R-X101-64x4d | pytorch |   2x    |   11.0   |      8.0       |  42.6  |  39.5   | [config]() |                    [model]( \| [log](                    |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/ms_rcnn/README.md)

---

### QueryInst

We present QueryInst, a new perspective for instance segmentation. QueryInst is a multi-stage end-to-end system that treats instances of interest as learnable queries, enabling query based object dete...

|   Model   | Backbone  |  Style  | Lr schd | Number of Proposals | Multi-Scale | RandomCrop | box AP | mask AP |                                                                       Config                                                                       |                                                                                                                                                                                                                       Download                                                                                                                                                                                                                       |
| :-------: | :-------: | :-----: | :-----: | :-----------------: | :---------: | :--------: | :----: | :-----: | :------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| QueryInst | R-50-FPN  | pytorch |   1x    |         100         |    False    |   False    |  42.0  |  37.5   |                   [config]()                   |                                                                         [model]( \| [log](                                                                         |
| QueryInst | R-50-FPN  | pytorch |   3x    |         100         |    True     |   False    |  44.8  |  39.8   |           [config]()           |                                         [model]( \| [log](                                         |
| QueryInst | R-50-FPN  | pytorch |   3x    |         300         |    True     |    True    |  47.5  |  41.7   | [config]()  |   [model]( \| [log](   |
| QueryInst | R-101-FPN | pytorch |   3x    |         100         |    True     |   False    |  46.4  |  41.0   |          [config]()           |                                       [model]( \| [log](                                       |
| QueryInst | R-101-FPN | pytorch |   3x    |         300         |    True     |    True    |  49.0  |  42.9   | [config]() | [model]( \| [log]( |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/queryinst/README.md)

---

### Sparse R-CNN

**Paper:** [Sparse R-CNN](https://arxiv.org/abs/2011.12450)

We present Sparse R-CNN, a purely sparse method for object detection in images. Existing works on object detection heavily rely on dense object candidates, such as k anchor boxes pre-defined on all gr...

|    Model     | Backbone  |  Style  | Lr schd | Number of Proposals | Multi-Scale | RandomCrop | box AP |                                                                         Config                                                                         |                                                                                                                                                                                                                                 Download                                                                                                                                                                                                                                  |
| :----------: | :-------: | :-----: | :-----: | :-----------------: | :---------: | :--------: | :----: | :----------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Sparse R-CNN | R-50-FPN  | pytorch |   1x    |         100         |    False    |   False    |  37.9  |                   [config]()                   |                                                                         [model]( \| [log](                                                                         |
| Sparse R-CNN | R-50-FPN  | pytorch |   3x    |         100         |    True     |   False    |  42.8  |           [config]()           |                                         [model]( \| [log](                                         |
| Sparse R-CNN | R-50-FPN  | pytorch |   3x    |         300         |    True     |    True    |  45.0  | [config]()  |   [model]( \| [log](   |
| Sparse R-CNN | R-101-FPN | pytorch |   3x    |         100         |    True     |   False    |  44.2  |          [config]()           |                                       [model]( \| [log](                                       |
| Sparse R-CNN | R-101-FPN | pytorch |   3x    |         300         |    True     |    True    |  46.2  | [config]() | [model]( \| [log]( |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/sparse_rcnn/README.md)

---

## One-Stage Detectors

### ATSS

**Paper:** [ATSS](https://arxiv.org/abs/1912.02424)

Object detection has been dominated by anchor-based detectors for several years. Recently, anchor-free detectors have become popular due to the proposal of FPN and Focal Loss. In this paper, we first ...

| Backbone |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP |                                                Config                                                 |                                                                                                                            Download                                                                                                                             |
| :------: | :-----: | :-----: | :------: | :------------: | :----: | :---------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   | pytorch |   1x    |   3.7    |      19.7      |  39.4  | [config]()  | [model]( \| [log]( |
|  R-101   | pytorch |   1x    |   5.6    |      12.3      |  41.5  | [config]() |   [model]( \| [log](   |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/atss/README.md)

---

### AutoAssign

**Paper:** [AutoAssign](https://arxiv.org/abs/2007.03496)

Determining positive/negative samples for object detection is known as label assignment. Here we present an anchor-free detector named AutoAssign. It requires little human knowledge and achieves appea...

| Backbone | Style | Lr schd | Mem (GB) | box AP |                                                        Config                                                        |                                                                                                                                                        Download                                                                                                                                                         |
| :------: | :---: | :-----: | :------: | :----: | :------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   | caffe |   1x    |   4.08   |  40.4  | [config]() | [model]( \| [log]( |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/autoassign/README.md)

---

### DDOD

Deep learning-based dense object detectors have achieved great success in the past few years and have been applied to numerous multimedia applications such as video understanding. However, the current...

|   Model   | Backbone |  Style  | Lr schd | Mem (GB) | box AP |                                                Config                                                |                                                                                                                                Download                                                                                                                                |
| :-------: | :------: | :-----: | :-----: | :------: | :----: | :--------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| DDOD-ATSS |   R-50   | pytorch |   1x    |   3.4    |  41.7  | [config]() | [model]( \| [log]( |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/ddod/README.md)

---

### FCOS

**Paper:** [FCOS](https://arxiv.org/abs/1904.01355)

We propose a fully convolutional one-stage object detector (FCOS) to solve object detection in a per-pixel prediction fashion, analogue to semantic segmentation. Almost all state-of-the-art object det...

| Backbone | Style | GN  | MS train | Tricks | DCN | Lr schd | Mem (GB) | Inf time (fps) | box AP |                                                                         Config                                                                          |                                                                                                                                                                                          Download                                                                                                                                                                                          |
| :------: | :---: | :-: | :------: | :----: | :-: | :-----: | :------: | :------------: | :----: | :-----------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   | caffe |  Y  |    N     |   N    |  N  |   1x    |   3.6    |      22.7      |  36.6  |                   [config]()                    |                                                        [model]( \| [log](                                                         |
|   R-50   | caffe |  Y  |    N     |   Y    |  N  |   1x    |   3.7    |       -        |  38.7  |   [config]()   |       [model]( \| [log](       |
|   R-50   | caffe |  Y  |    N     |   Y    |  Y  |   1x    |   3.8    |       -        |  42.3  | [config]() | [model]( \| [log]( |
|  R-101   | caffe |  Y  |    N     |   N    |  N  |   1x    |   5.5    |      17.3      |  39.1  |                   [config]()                   |                                                       [model]( \| [log](                                                       |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/fcos/README.md)

---

### GFL

**Paper:** [GFL](https://arxiv.org/abs/2006.04388)

One-stage detector basically formulates object detection as dense classification and localization. The classification is usually optimized by Focal Loss and the box location is commonly learned under ...

|     Backbone      |  Style  | Lr schd | Multi-scale Training | Inf time (fps) | box AP |                                                            Config                                                             |                                                                                                                                                                                   Download                                                                                                                                                                                   |
| :---------------: | :-----: | :-----: | :------------------: | :------------: | :----: | :---------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|       R-50        | pytorch |   1x    |          No          |      19.5      |  40.2  |              [config]()               |                                                       [model]( \| [log](                                                       |
|       R-50        | pytorch |   2x    |         Yes          |      19.5      |  42.9  |          [config]()           |                                       [model]( \| [log](                                       |
|       R-101       | pytorch |   2x    |         Yes          |      14.7      |  44.7  |          [config]()          |                                     [model]( \| [log](                                     |
|    R-101-dcnv2    | pytorch |   2x    |         Yes          |      12.9      |  47.1  |    [config]()    |             [model]( \| [log](             |
|    X-101-32x4d    | pytorch |   2x    |         Yes          |      12.1      |  45.9  |       [config]()       |                         [model]( \| [log](                         |
| X-101-32x4d-dcnv2 | pytorch |   2x    |         Yes          |      10.7      |  48.1  | [config]() | [model]( \| [log]( |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/gfl/README.md)

---

### PAA

**Paper:** [PAA](https://arxiv.org/abs/2007.08103)

In object detection, determining which anchors to assign as positive or negative samples, known as anchor assignment, has been revealed as a core procedure that can significantly affect a model's perf...

| Backbone  | Lr schd | Mem (GB) | Score voting | box AP |                                                   Config                                                    |                                                                                                                                               Download                                                                                                                                               |
| :-------: | :-----: | :------: | :----------: | :----: | :---------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| R-50-FPN  |   12e   |   3.7    |     True     |  40.4  |     [config]()      |                     [model]( \| [log](                      |
| R-50-FPN  |   12e   |   3.7    |    False     |  40.2  |                                                      -                                                      |                                                                                                                                                                                                                                                                                                      |
| R-50-FPN  |   18e   |   3.7    |     True     |  41.4  |    [config]()     |                 [model]( \| [log](                  |
| R-50-FPN  |   18e   |   3.7    |    False     |  41.2  |                                                      -                                                      |                                                                                                                                                                                                                                                                                                      |
| R-50-FPN  |   24e   |   3.7    |     True     |  41.6  |     [config]()      |                     [model]( \| [log](                      |
| R-50-FPN  |   36e   |   3.7    |     True     |  43.3  | [config]()  |   [model]( \| [log](   |
| R-101-FPN |   12e   |   6.2    |     True     |  42.6  |     [config]()     |                   [model]( \| [log](                    |
| R-101-FPN |   12e   |   6.2    |    False     |  42.4  |                                                      -                                                      |                                                                                                                                                                                                                                                                                                      |
| R-101-FPN |   24e   |   6.2    |     True     |  43.5  |     [config]()     |                   [model]( \| [log](                    |
| R-101-FPN |   36e   |   6.2    |     True     |  45.1  | [config]() | [model]( \| [log]( |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/paa/README.md)

---

### RetinaNet

**Paper:** [RetinaNet](https://arxiv.org/abs/1708.02002)

The highest accuracy object detectors to date are based on a two-stage approach popularized by R-CNN, where a classifier is applied to a sparse set of candidate object locations. In contrast, one-stag...

|    Backbone     |  Style  |   Lr schd    | Mem (GB) | Inf time (fps) | box AP |                                                        Config                                                         |                                                                                                                                                         Download                                                                                                                                                          |
| :-------------: | :-----: | :----------: | :------: | :------------: | :----: | :-------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|    R-18-FPN     | pytorch |      1x      |   1.7    |                |  31.7  |    [config]()     |           [model]( \| [log](            |
|    R-18-FPN     | pytorch | 1x(1 x 8 BS) |   5.0    |                |  31.7  |  [config]()   |   [model]( \| [log](    |
|    R-50-FPN     |  caffe  |      1x      |   3.5    |      18.6      |  36.3  | [config]()  |   [model]( \| [log](   |
|    R-50-FPN     | pytorch |      1x      |   3.8    |      19.0      |  36.5  |    [config]()     |               [model]( \| [log](               |
| R-50-FPN (FP16) | pytorch |      1x      |   2.8    |      31.6      |  36.4  |  [config]()  |          [model]( \| [log](          |
|    R-50-FPN     | pytorch |      2x      |    -     |       -        |  37.4  |    [config]()     |               [model]( \| [log](               |
|    R-101-FPN    |  caffe  |      1x      |   5.5    |      14.7      |  38.5  | [config]() | [model]( \| [log]( |
|    R-101-FPN    | pytorch |      1x      |   5.7    |      15.0      |  38.5  |    [config]()    |             [model]( \| [log](             |
|    R-101-FPN    | pytorch |      2x      |    -     |       -        |  38.9  |    [config]()    |             [model]( \| [log](             |
| X-101-32x4d-FPN | pytorch |      1x      |   7.0    |      12.1      |  39.9  | [config]() | [model]( \| [log]( |
| X-101-32x4d-FPN | pytorch |      2x      |    -     |       -        |  40.1  | [config]() | [model]( \| [log]( |
| X-101-64x4d-FPN | pytorch |      1x      |   10.0   |      8.7       |  41.0  | [config]() | [model]( \| [log]( |
| X-101-64x4d-FPN | pytorch |      2x      |    -     |       -        |  40.8  | [config]() | [model]( \| [log]( |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/retinanet/README.md)

---

### TOOD

**Paper:** [TOOD](https://arxiv.org/abs/2108.07755)

One-stage object detection is commonly implemented by optimizing two sub-tasks: object classification and localization, using heads with two parallel branches, which might lead to a certain level of s...

|     Backbone      |  Style  | Anchor Type  | Lr schd | Multi-scale Training | Mem (GB) | Inf time (fps) | box AP |                             Config                             |                                                                                                                                                                       Download                                                                                                                                                                        |
| :---------------: | :-----: | :----------: | :-----: | :------------------: | :------: | :------------: | :----: | :------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|       R-50        | pytorch | Anchor-free  |   1x    |          N           |   4.1    |                |  42.4  |              [config](./tood_r50_fpn_1x_coco.py)               |                                           [model]( \| [log](                                           |
|       R-50        | pytorch | Anchor-based |   1x    |          N           |   4.1    |                |  42.4  |        [config](./tood_r50_fpn_anchor_based_1x_coco.py)        |                 [model]( \| [log](                 |
|       R-50        | pytorch | Anchor-free  |   2x    |          Y           |   4.1    |                |  44.5  |          [config](./tood_r50_fpn_mstrain_2x_coco.py)           |                           [model]( \| [log](                           |
|       R-101       | pytorch | Anchor-free  |   2x    |          Y           |   6.0    |                |  46.1  |          [config](./tood_r101_fpn_mstrain_2x_coco.py)          |                         [model]( \| [log](                         |
|    R-101-dcnv2    | pytorch | Anchor-free  |   2x    |          Y           |   6.2    |                |  49.3  |    [config](./tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco.py)    | [model]( \| [log]( |
|    X-101-64x4d    | pytorch | Anchor-free  |   2x    |          Y           |   10.2   |                |  47.6  |       [config](./tood_x101_64x4d_fpn_mstrain_2x_coco.py)       |             [model]( \| [log](             |
| X-101-64x4d-dcnv2 | pytorch | Anchor-free  |   2x    |          Y           |          |                |        | [config](./tood_x101_64x4d_fpn_dconv_c4-c5_mstrain_2x_coco.py) |                                                                                                                                                               [model](<>) \| [log](<>)                                                                                                                                                                |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/tood/README.md)

---

### VarifocalNet

**Paper:** [VarifocalNet](https://arxiv.org/abs/2008.13367)

Accurately ranking the vast number of candidate detections is crucial for dense object detectors to achieve high performance. Prior work uses the classification score or a combination of classificatio...

|  Backbone   |  Style  | DCN | MS train | Lr schd | Inf time (fps) | box AP (val) | box AP (test-dev) |                                                               Config                                                               |                                                                                                                                                                               Download                                                                                                                                                                               |
| :---------: | :-----: | :-: | :------: | :-----: | :------------: | :----------: | :---------------: | :--------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|    R-50     | pytorch |  N  |    N     |   1x    |       -        |     41.6     |       41.6        |               [config]()               |                                                          [model]( \| [log](                                                           |
|    R-50     | pytorch |  N  |    Y     |   2x    |       -        |     44.5     |       44.8        |           [config]()           |                                          [model]( \| [log](                                           |
|    R-50     | pytorch |  Y  |    Y     |   2x    |       -        |     47.8     |       48.0        |    [config]()     |               [model]( \| [log](               |
|    R-101    | pytorch |  N  |    N     |   1x    |       -        |     43.0     |       43.6        |              [config]()               |                                                       [model]( \| [log](                                                       |
|    R-101    | pytorch |  N  |    Y     |   2x    |       -        |     46.2     |       46.7        |          [config]()           |                                       [model]( \| [log](                                       |
|    R-101    | pytorch |  Y  |    Y     |   2x    |       -        |     49.0     |       49.2        |    [config]()    |             [model]( \| [log](             |
| X-101-32x4d | pytorch |  Y  |    Y     |   2x    |       -        |     49.7     |       50.0        | [config]() | [model]( \| [log]( |
| X-101-64x4d | pytorch |  Y  |    Y     |   2x    |       -        |     50.4     |       50.8        | [config]() | [model]( \| [log]( |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/vfnet/README.md)

---

## YOLO Series

### YOLACT

**Paper:** [YOLACT](https://arxiv.org/abs/1904.02689)

We present a simple, fully-convolutional model for real-time instance segmentation that achieves 29.8 mAP on MS COCO at 33.5 fps evaluated on a single Titan Xp, which is significantly faster than any ...

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/yolact/README.md)

---

### YOLOv3

**Paper:** [YOLOv3](https://arxiv.org/abs/1804.02767)

We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained this new network that's pretty swell. It's a little bigger than last time but more accurate...

|  Backbone  | Scale | Lr schd | Mem (GB) | Inf time (fps) | box AP |                                                      Config                                                      |                                                                                                                                                        Download                                                                                                                                                        |
| :--------: | :---: | :-----: | :------: | :------------: | :----: | :--------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| DarkNet-53 |  320  |  273e   |   2.7    |      63.9      |  27.9  |     [config]()     |                         [model]( \| [log](                         |
| DarkNet-53 |  416  |  273e   |   3.8    |      61.2      |  30.9  | [config]() |         [model]( \| [log](         |
| DarkNet-53 |  608  |  273e   |   7.4    |      48.1      |  33.7  | [config]() | [model]( \| [log]( |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/yolo/README.md)

---

### YOLOF

**Paper:** [YOLOF](https://arxiv.org/abs/2103.09460)

This paper revisits feature pyramids networks (FPN) for one-stage detectors and points out that the success of FPN is due to its divide-and-conquer solution to the optimization problem in object detec...

| Backbone | Style | Epoch | Lr schd | Mem (GB) | box AP |                                                  Config                                                   |                                                                                                                                         Download                                                                                                                                         |
| :------: | :---: | :---: | :-----: | :------: | :----: | :-------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| R-50-C5  | caffe |   Y   |   1x    |   8.3    |  37.5  | [config]() | [model]( \| [log]( |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/yolof/README.md)

---

### YOLOX

**Paper:** [YOLOX](https://arxiv.org/abs/2107.08430)

In this report, we present some experienced improvements to YOLO series, forming a new high-performance detector -- YOLOX. We switch the YOLO detector to an anchor-free manner and conduct other advanc...

|  Backbone  | size | Mem (GB) | box AP |                                                  Config                                                   |                                                                                                                                         Download                                                                                                                                         |
| :--------: | :--: | :------: | :----: | :-------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLOX-tiny | 416  |   3.5    |  32.0  | [config]() | [model]( \| [log]( |
|  YOLOX-s   | 640  |   7.6    |  40.5  |  [config]()   |       [model]( \| [log](       |
|  YOLOX-l   | 640  |   19.9   |  49.4  |  [config]()   |       [model]( \| [log](       |
|  YOLOX-x   | 640  |   28.1   |  50.9  |  [config]()   |       [model]( \| [log](       |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/yolox/README.md)

---

## Transformer-Based

### Deformable DETR

**Paper:** [Deformable DETR](https://arxiv.org/abs/2010.04159)

DETR has been recently proposed to eliminate the need for many hand-designed components in object detection while demonstrating good performance. However, it suffers from slow convergence and limited ...

| Backbone |                Model                | Lr schd | box AP |                                                                    Config                                                                    |                                                                                                                                                                                                         Download                                                                                                                                                                                                          |
| :------: | :---------------------------------: | :-----: | :----: | :------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   |           Deformable DETR           |   50e   |  44.5  |         [config]()         |                                 [model]( \| [log](                                 |
|   R-50   | + iterative bounding box refinement |   50e   |  46.1  |     [config]()      |                   [model]( \| [log](                   |
|   R-50   |    ++ two-stage Deformable DETR     |   50e   |  46.8  | [config]() | [model]( \| [log]( |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/deformable_detr/README.md)

---

### DETR

**Paper:** [DETR](https://arxiv.org/abs/2005.12872)

We present a new method that views object detection as a direct set prediction problem. Our approach streamlines the detection pipeline, effectively removing the need for many hand-designed components...

| Backbone | Model | Lr schd | Mem (GB) | Inf time (fps) | box AP |                                                 Config                                                 |                                                                                                                                    Download                                                                                                                                    |
| :------: | :---: | :-----: | :------: | :------------: | :----: | :----------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   | DETR  |  150e   |   7.9    |                |  40.1  | [config]() | [model]( \| [log]( |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/detr/README.md)

---

### Mask2Former

Image segmentation is about grouping pixels with different semantics, e.g., category or instance membership, where each choice of semantics defines a task. While only the semantics of each task differ...

| Backbone |  style  |   Pretrain   | Lr schd | Mem (GB) | Inf time (fps) |  PQ  | box mAP | mask mAP |                                                                         Config                                                                         |                                                                                                                                                                                                                             Download                                                                                                                                                                                                                             |
| :------: | :-----: | :----------: | :-----: | :------: | :------------: | :--: | :-----: | :------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   | pytorch | ImageNet-1K  |   50e   |   13.9   |       -        | 51.9 |  44.8   |   41.9   |            [config]()            |                                             [model]( \| [log](                                             |
|  R-101   | pytorch | ImageNet-1K  |   50e   |   16.1   |       -        | 52.4 |  45.3   |   42.4   |           [config]()            |                                           [model]( \| [log](                                           |
|  Swin-T  |    -    | ImageNet-1K  |   50e   |   15.9   |       -        | 53.4 |  46.3   |   43.4   |     [config]()      |                   [model]( \| [log](                   |
|  Swin-S  |    -    | ImageNet-1K  |   50e   |   19.1   |       -        | 54.5 |  47.8   |   44.5   |     [config]()      |                   [model]( \| [log](                   |
|  Swin-B  |    -    | ImageNet-1K  |   50e   |   26.0   |       -        | 55.1 |  48.2   |   44.9   |     [config]()     |                 [model]( \| [log](                 |
|  Swin-B  |    -    | ImageNet-21K |   50e   |   25.8   |       -        | 56.3 |  50.0   |   46.3   |  [config]()  |     [model]( \| [log](     |
|  Swin-L  |    -    | ImageNet-21K |  100e   |   21.1   |       -        | 57.6 |  52.2   |   48.5   | [config]() | [model]( \| [log]( |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/mask2former/README.md)

---

### MaskFormer

**Paper:** [MaskFormer](https://arxiv.org/abs/2107.06278)

Modern approaches typically formulate semantic segmentation as a per-pixel classification task, while instance-level segmentation is handled with an alternative mask classification. Our key insight: m...

| Backbone |  style  | Lr schd | Mem (GB) | Inf time (fps) |   PQ   |   SQ   |   RQ   | PQ_th  | SQ_th  | RQ_th  | PQ_st  | SQ_st  | RQ_st  |                                                                Config                                                                 |                                                                                                                                                                                            Download                                                                                                                                                                                            |                                                                         detail                                                                          |
| :------: | :-----: | :-----: | :------: | :------------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :-----------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   | pytorch |   75e   |   16.2   |       -        | 46.854 | 80.617 | 57.085 | 51.089 | 81.511 | 61.853 | 40.463 | 79.269 | 49.888 |      [config]()       |                       [model]( \| [log](                       | This version was mentioned in Table XI, in paper [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527) |
|  Swin-L  | pytorch |  300e   |   27.2   |       -        | 53.249 | 81.704 | 64.231 | 58.798 | 82.923 | 70.282 | 44.874 | 79.863 | 55.097 | [config]() | [model]( \| [log]( |                                                                            -                                                                            |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/maskformer/README.md)

---

## Instance Segmentation

### SOLO

**Paper:** [SOLO](https://arxiv.org/abs/1912.04488)

We present a new, embarrassingly simple approach to instance segmentation in images. Compared to many other dense prediction tasks, e.g., semantic segmentation, it is the arbitrary number of instances...

| Backbone |  Style  | MS train | Lr schd | Mem (GB) | Inf time (fps) | mask AP |                                                                                                                                Download                                                                                                                                |
| :------: | :-----: | :------: | :-----: | :------: | :------------: | :-----: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   | pytorch |    N     |   1x    |   8.0    |      14.0      |  33.1   | [model]( \| [log]( |
|   R-50   | pytorch |    Y     |   3x    |   7.4    |      14.0      |  35.9   | [model]( \| [log]( |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/solo/README.md)

---

### SOLOv2

**Paper:** [SOLOv2](https://arxiv.org/abs/2003.10152)

In this work, we aim at building a simple, direct, and fast instance segmentation
framework with strong performance. We follow the principle of the SOLO method of
Wang et al. "SOLO: segmenting objects...

|  Backbone  |  Style  | MS train | Lr schd | Mem (GB) | mask AP |                                                    Config                                                     |                                                                                                                                                Download                                                                                                                                                |
| :--------: | :-----: | :------: | :-----: | :------: | :-----: | :-----------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|    R-50    | pytorch |    N     |   1x    |   5.1    |  34.8   |   [config]()    |      [model](           \| [log](      |
|    R-50    | pytorch |    Y     |   3x    |   5.1    |  37.5   |   [config]()    |      [model](           \| [log](      |
|   R-101    | pytorch |    Y     |   3x    |   6.9    |  39.1   |   [config]()   |     [model](         \| [log](     |
| R-101(DCN) | pytorch |    Y     |   3x    |   7.1    |  41.2   | [config]() | [model]( \| [log]( |
| X-101(DCN) | pytorch |    Y     |   3x    |   11.3   |  42.4   | [config]() | [model]( \| [log]( |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/solov2/README.md)

---

## Panoptic Segmentation

### Panoptic FPN

**Paper:** [Panoptic FPN](https://arxiv.org/abs/1901.02446)

The recently introduced panoptic segmentation task has renewed our community's interest in unifying the tasks of instance segmentation (for thing classes) and semantic segmentation (for stuff classes)...

| Backbone  |  style  | Lr schd | Mem (GB) | Inf time (fps) |  PQ  |  SQ  |  RQ  | PQ_th | SQ_th | RQ_th | PQ_st | SQ_st | RQ_st |                                                            Config                                                             |                                                                                                                                                                          Download                                                                                                                                                                          |
| :-------: | :-----: | :-----: | :------: | :------------: | :--: | :--: | :--: | :---: | :---: | :---: | :---: | :---: | :---: | :---------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| R-50-FPN  | pytorch |   1x    |   4.7    |                | 40.2 | 77.8 | 49.3 | 47.8  | 80.9  | 57.5  | 28.9  | 73.1  | 37.0  |     [config]()      |                   [model]( \| [log](                   |
| R-50-FPN  | pytorch |   3x    |    -     |       -        | 42.5 | 78.1 | 51.7 | 50.3  | 81.5  | 60.3  | 30.7  | 73.0  | 38.8  | [config]()  |   [model]( \| [log](   |
| R-101-FPN | pytorch |   1x    |   6.7    |                | 42.2 | 78.3 | 51.4 | 50.1  | 81.4  | 59.9  | 30.3  | 73.6  | 38.5  |     [config]()     |                 [model]( \| [log](                 |
| R-101-FPN | pytorch |   3x    |    -     |       -        | 44.1 | 78.9 | 53.6 | 52.1  | 81.7  | 62.3  | 32.0  | 74.6  | 40.3  | [config]() | [model]( \| [log]( |
| R2-50-FPN | pytorch |   1x    |    -     |       -        | 42.5 | 78.0 | 51.8 | 50.0  | 81.4  | 60.0  | 31.1  | 72.8  | 39.4  |   [config]()    |                   [model]( \| [log](                   |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/panoptic_fpn/README.md)

---

## Specialized Backbones

### ConvNeXt

**Paper:** [ConvNeXt](https://arxiv.org/abs/2201.03545)

The "Roaring 20s" of visual recognition began with the introduction of Vision Transformers (ViTs), which quickly superseded ConvNets as the state-of-the-art image classification model. A vanilla ViT, ...

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/convnext/README.md)

---

### EfficientNet

**Paper:** [EfficientNet](https://arxiv.org/abs/1905.11946v5)

|    Backbone     |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP |                                                             Config                                                              |                                                                                                                                                                              Download                                                                                                                                                                              |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Efficientnet-b3 | pytorch |   1x    |    -     |       -        |  40.5  | [config]() | [model]( \| [log]( |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/efficientnet/README.md)

---

### HRNet

**Paper:** [HRNet](https://arxiv.org/abs/1902.09212)

This is an official pytorch implementation of Deep High-Resolution Representation Learning for Human Pose Estimation. In this work, we are interested in the human pose estimation problem with a focus ...

|   Backbone   |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP |                                                      Config                                                       |                                                                                                                                                         Download                                                                                                                                                         |
| :----------: | :-----: | :-----: | :------: | :------------: | :----: | :---------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| HRNetV2p-W18 | pytorch |   1x    |   6.6    |      13.4      |  36.9  | [config]() |    [model]( \| [log](     |
| HRNetV2p-W18 | pytorch |   2x    |   6.6    |       -        |  38.9  | [config]() | [model]( \| [log]( |
| HRNetV2p-W32 | pytorch |   1x    |   9.0    |      12.4      |  40.2  | [config]() |    [model]( \| [log](     |
| HRNetV2p-W32 | pytorch |   2x    |   9.0    |       -        |  41.4  | [config]() | [model]( \| [log]( |
| HRNetV2p-W40 | pytorch |   1x    |   10.4   |      10.5      |  41.2  | [config]() |    [model]( \| [log](     |
| HRNetV2p-W40 | pytorch |   2x    |   10.4   |       -        |  42.1  | [config]() | [model]( \| [log]( |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/hrnet/README.md)

---

### PVT

**Paper:** [PVT](https://arxiv.org/abs/2102.12122)

Although using convolutional neural networks (CNNs) as backbones achieves great successes in computer vision, this work investigates a simple backbone network useful for many dense prediction tasks wi...

|  Backbone  | Lr schd | Mem (GB) | box AP |                                                   Config                                                   |                                                                                                                                             Download                                                                                                                                             |
| :--------: | :-----: | :------: | :----: | :--------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  PVT-Tiny  |   12e   |   8.5    |  36.6  | [config]() | [model]( \| [log]( |
| PVT-Small  |   12e   |   14.5   |  40.4  | [config]() | [model]( \| [log]( |
| PVT-Medium |   12e   |   20.9   |  41.7  | [config]() | [model]( \| [log]( |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/pvt/README.md)

---

### RegNet

**Paper:** [RegNet](https://arxiv.org/abs/2003.13678)

In this work, we present a new network design paradigm. Our goal is to help advance the understanding of network design and discover design principles that generalize across settings. Instead of focus...

|                                       Backbone                                       |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP | mask AP |                                                               Config                                                               |                                                                                                                                                                                          Download                                                                                                                                                                                          |
| :----------------------------------------------------------------------------------: | :-----: | :-----: | :------: | :------------: | :----: | :-----: | :--------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                [R-50-FPN](../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py)                 | pytorch |   1x    |   4.4    |      12.0      |  38.2  |  34.7   |           [config]()           |                                               [model]( \| [log](                                                |
|            [RegNetX-3.2GF-FPN](./mask_rcnn_regnetx-3.2GF_fpn_1x_coco.py)             | pytorch |   1x    |   5.0    |                |  40.3  |  36.6   |       [config]()        |                           [model]( \| [log](                           |
|             [RegNetX-4.0GF-FPN](./mask_rcnn_regnetx-4GF_fpn_1x_coco.py)              | pytorch |   1x    |   5.5    |                |  41.5  |  37.4   |        [config]()         |                               [model]( \| [log](                               |
|               [R-101-FPN](../mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py)                | pytorch |   1x    |   6.4    |      10.3      |  40.0  |  36.1   |          [config]()           |                                             [model]( \| [log](                                              |
|            [RegNetX-6.4GF-FPN](./mask_rcnn_regnetx-6.4GF_fpn_1x_coco.py)             | pytorch |   1x    |   6.1    |                |  41.0  |  37.1   |       [config]()        |                           [model]( \| [log](                           |
|         [X-101-32x4d-FPN](../mask_rcnn/mask_rcnn_x101_32x4d_fpn_1x_coco.py)          | pytorch |   1x    |   7.6    |      9.4       |  41.9  |  37.5   |       [config]()        |                                 [model]( \| [log](                                  |
|             [RegNetX-8.0GF-FPN](./mask_rcnn_regnetx-8GF_fpn_1x_coco.py)              | pytorch |   1x    |   6.4    |                |  41.7  |  37.5   |        [config]()         |                               [model]( \| [log](                               |
|             [RegNetX-12GF-FPN](./mask_rcnn_regnetx-12GF_fpn_1x_coco.py)              | pytorch |   1x    |   7.4    |                |  42.2  |   38    |        [config]()        |                             [model]( \| [log](                             |
| [RegNetX-3.2GF-FPN-DCN-C3-C5](./mask_rcnn_regnetx-3.2GF_fpn_mdconv_c3-c5_1x_coco.py) | pytorch |   1x    |   5.0    |                |  40.3  |  36.6   | [config]() | [model]( \| [log]( |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/regnet/README.md)

---

### Res2Net

**Paper:** [Res2Net](https://arxiv.org/abs/1904.01169)

Representing features at multiple scales is of great importance for numerous vision tasks. Recent advances in backbone convolutional neural networks (CNNs) continually demonstrate stronger multi-scale...

|  Backbone  |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP |                                                      Config                                                       |                                                                                                                                               Download                                                                                                                                               |
| :--------: | :-----: | :-----: | :------: | :------------: | :----: | :---------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| R2-101-FPN | pytorch |   2x    |   7.4    |       -        |  43.0  | [config]() | [model]( \| [log]( |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/res2net/README.md)

---

### ResNeSt

**Paper:** [ResNeSt](https://arxiv.org/abs/2004.08955)

It is well known that featuremap attention and multi-path representation are important for visual recognition. In this paper, we present a modularized architecture, which applies the channel-wise atte...

| Backbone  |  Style  | Lr schd | Mem (GB) | Inf time (fps) | box AP |                                                                       Config                                                                       |                                                                                                                                                                                                                             Download                                                                                                                                                                                                                             |
| :-------: | :-----: | :-----: | :------: | :------------: | :----: | :------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| S-50-FPN  | pytorch |   1x    |   4.8    |       -        |  42.0  | [config]()  |   [model]( \| [log](   |
| S-101-FPN | pytorch |   1x    |   7.1    |       -        |  44.5  | [config]() | [model]( \| [log]( |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/resnest/README.md)

---

### Swin

**Paper:** [Swin](https://arxiv.org/abs/2103.14030)

This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformer from language to vision...

| Backbone |  Pretrain   | Lr schd | Multi-scale crop | FP16 | Mem (GB) | Inf time (fps) | box AP | mask AP |                             Config                             |                                                                                                                                                                                      Download                                                                                                                                                                                       |
| :------: | :---------: | :-----: | :--------------: | :--: | :------: | :------------: | :----: | :-----: | :------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  Swin-T  | ImageNet-1K |   1x    |        no        |  no  |   7.6    |                |  42.7  |  39.3   |       [config](./mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py)        |                           [model](  \| [log](                           |
|  Swin-T  | ImageNet-1K |   3x    |       yes        |  no  |   10.2   |                |  46.0  |  41.6   |   [config](./mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py)    |           [model](  \| [log](           |
|  Swin-T  | ImageNet-1K |   3x    |       yes        | yes  |   7.8    |                |  46.0  |  41.7   | [config](./mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py) | [model](  \| [log]( |
|  Swin-S  | ImageNet-1K |   3x    |       yes        | yes  |   11.9   |                |  48.2  |  43.2   | [config](./mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py) | [model](  \| [log]( |

[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/swin/README.md)

---

## Using Pre-trained Models

To use any pre-trained model from the model zoo:

```python
from mmdet.apis import init_detector, inference_detector

# Load model
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Run inference
result = inference_detector(model, 'demo/demo.jpg')
```

## Training Custom Models

See the [Training Guide](user-guide/training.md) for instructions on training models with your own data.

## Contributing New Models

We welcome contributions of new model implementations! Please see the [Contributing Guide](development/contributing.md) for details.
