# Inference

This guide covers running inference with trained models.

## Quick Start

Run inference on a single image:

```python
from mmdet.apis import init_detector, inference_detector

config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco.pth'

# Build the model from config and checkpoint
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Run inference on an image
result = inference_detector(model, 'demo/demo.jpg')
```

## Visualization

Display results:

```python
from mmdet.apis import show_result_pyplot

# Show the results
show_result_pyplot(model, 'demo/demo.jpg', result, score_thr=0.3)
```

Save results to file:

```python
from mmdet.apis import show_result_pyplot

model.show_result('demo/demo.jpg', result, out_file='result.jpg', score_thr=0.3)
```

## Batch Inference

Process multiple images:

```python
import glob
from mmdet.apis import inference_detector

# Get all images in a directory
image_files = glob.glob('path/to/images/*.jpg')

for image_file in image_files:
    result = inference_detector(model, image_file)
    # Process result...
```

## Test Dataset

Evaluate model on a test dataset:

```bash
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval bbox segm
```

With multiple GPUs:

```bash
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} --eval bbox segm
```

## Advanced Usage

### Custom Inference Pipeline

Create a custom inference pipeline:

```python
from mmdet.apis import init_detector
from mmcv import Config

# Load config
cfg = Config.fromfile('config.py')

# Modify config if needed
cfg.model.test_cfg.score_thr = 0.5

# Initialize model
model = init_detector(cfg, 'checkpoint.pth', device='cuda:0')

# Run inference
result = inference_detector(model, 'image.jpg')
```

### Async Inference

For high-throughput scenarios:

```python
import asyncio
from mmdet.apis import init_detector, async_inference_detector

async def async_process():
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    tasks = [async_inference_detector(model, img) for img in images]
    results = await asyncio.gather(*tasks)
    return results
```

## Export Models

### ONNX Export

Export model to ONNX format:

```bash
python tools/deployment/pytorch2onnx.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --output-file model.onnx
```

### TensorRT

Convert to TensorRT for optimized inference:

```bash
python tools/deployment/onnx2tensorrt.py model.onnx --output model.trt
```

## See Also

- [Configuration Guide](configuration.md)
- [Training Guide](training.md)
- [API Reference](../api/core.md)
