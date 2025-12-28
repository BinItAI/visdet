# Quick Start

This guide provides quick examples to get you started with VisDet.

## Training with SimpleRunner

The easiest way to train a model is using the `SimpleRunner` API:

```python
from visdet import SimpleRunner

# Simple, string-based API - just like Hugging Face or Ultralytics
runner = SimpleRunner(
    model='mask_rcnn_swin_s',
    dataset='coco_instance_segmentation',
    epochs=12,
    batch_size=2
)

runner.train()
```

## Inference with Pre-trained Models

### Using YAML model presets (no repo clone)

VisDet ships with YAML model presets, so you can run inference without needing this repository’s Python config files.

```python
from visdet.apis import DetInferencer

# Uses a built-in preset name/alias; may download weights on first use
inferencer = DetInferencer(model="rtmdet-s", device="cuda:0")
result = inferencer("path/to/image.jpg")
print(result)
```

### Using explicit config + checkpoint (repo-style)

If you *are* working from a cloned repo (or you have your own configs/checkpoints), you can still use the classic APIs:

```python
from visdet.apis import init_detector, inference_detector, show_result_pyplot

config_file = "path/to/config.py"
checkpoint_file = "path/to/checkpoint.pth"

model = init_detector(config_file, checkpoint_file, device="cuda:0")
img = "path/to/image.jpg"
result = inference_detector(model, img)
show_result_pyplot(model, img, result)
```

## Training a Model

The training / testing entrypoints under `tools/` are part of this repository.
If you want to use them, clone the repo and run from the repo root.

### Train with a Single GPU

```bash
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example:

```bash
python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py
```

### Train with Multiple GPUs

```bash
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Example:

```bash
bash ./tools/dist_train.sh configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py 8
```

## Testing a Model

### Test with a Single GPU

```bash
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

### Test with Multiple GPUs

```bash
bash ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [optional arguments]
```

## Next Steps

- Learn about [Configuration](../tutorials/config.md) system
- Customize your [Datasets](../tutorials/customize_dataset.md)
- Build custom [Models](../tutorials/customize_models.md)
