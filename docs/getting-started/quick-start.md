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

### Using High-level APIs

You can use high-level APIs to perform inference on images:

```python
from visdet.apis import init_detector, inference_detector

# Specify the config file and checkpoint file
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco.pth'

# Build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Test a single image
img = 'demo/demo.jpg'
result = inference_detector(model, img)

# Show the results
from visdet.apis import show_result_pyplot
show_result_pyplot(model, img, result)
```

## Training a Model

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
