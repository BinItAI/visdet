# Models API

This page documents the model APIs in VisDet.

## Detectors

### Base Detector

::: visdet.models.detectors.BaseDetector
    options:
      show_source: true
      members:
        - forward_train
        - simple_test
        - aug_test

### Two-Stage Detectors

::: visdet.models.detectors.TwoStageDetector
    options:
      show_source: true

### Mask R-CNN

::: visdet.models.detectors.MaskRCNN
    options:
      show_source: true

## Backbones

::: visdet.models.backbones
    options:
      show_source: false
      members: false

## Necks

::: visdet.models.necks
    options:
      show_source: false
      members: false

## Heads

::: visdet.models.dense_heads
    options:
      show_source: false
      members: false

## See Also

- [Core API](core.md)
- [Datasets API](datasets.md)
