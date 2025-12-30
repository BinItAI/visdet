# Inference

This guide covers running inference with trained models.

## Quick Start

Run inference on a single image:

```python
from visdet.apis import inference_detector, init_detector

config_file = "configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
checkpoint_file = "checkpoints/faster_rcnn_r50_fpn_1x_coco.pth"

# Build the model from config and checkpoint
model = init_detector(config_file, checkpoint_file, device="cuda:0")

# Run inference on an image
result = inference_detector(model, "demo/demo.jpg")
```

## Visualization

Save results to file:

```python
from visdet.cv import imread
from visdet.visualization import DetLocalVisualizer

# DetLocalVisualizer expects RGB images
image = imread("demo/demo.jpg", channel_order="rgb")

visualizer = DetLocalVisualizer()
visualizer.dataset_meta = model.dataset_meta
visualizer.add_datasample(
    "result",
    image,
    data_sample=result,
    draw_gt=False,
    out_file="result.jpg",
    pred_score_thr=0.3,
)
```

## Batch Inference

Process multiple images:

```python
import glob

from visdet.apis import inference_detector

# Get all images in a directory
image_files = glob.glob("path/to/images/*.jpg")

# Run as a batch for better performance
results = inference_detector(model, image_files)
```

`results` is a list of `DetDataSample` objects (one per input image).

### Multi-GPU Inference

To run inference on multiple GPUs in a single process, pass multiple CUDA devices to `init_detector` and then infer a batch (a list) of images:

```python
from visdet.apis import inference_detector, init_detector

model = init_detector(config_file, checkpoint_file, device="cuda:0,1")
results = inference_detector(model, image_files)  # list[DetDataSample]
```

## FiftyOne (Voxel51) Dataset

If you use [FiftyOne](https://voxel51.com/fiftyone/) for dataset management and visualization, you can run `visdet` inference over a `fiftyone.Dataset` and attach the predictions back onto each sample.

```python
import fiftyone as fo

from visdet.apis import inference_detector, init_detector
from visdet.utils import detections_to_fiftyone

# 1) Load your model
config_file = "configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
checkpoint_file = "checkpoints/faster_rcnn_r50_fpn_1x_coco.pth"
model = init_detector(config_file, checkpoint_file, device="cuda:0")

# 2) Create (or load) a FiftyOne dataset
#    You can also do: dataset = fo.load_dataset("my-dataset")
dataset = fo.Dataset.from_images_dir(
    "path/to/images",
    name="my-images",
    overwrite=True,
)
dataset.compute_metadata()  # populates sample.metadata.width/height

classes = model.dataset_meta.get("classes", [])
score_thr = 0.3

# 3) Run batched inference and attach detections
#    (Batching is important for performance, and required for multi-GPU inference)
filepaths = dataset.values("filepath")

batch_size = 8
predictions = []
for i in range(0, len(filepaths), batch_size):
    predictions.extend(inference_detector(model, filepaths[i : i + batch_size]))

for sample, data_sample in zip(dataset.iter_samples(progress=True), predictions, strict=True):
    pred = data_sample.pred_instances
    pred = pred[pred.scores > score_thr]

    width = sample.metadata.width
    height = sample.metadata.height

    dets = []
    for bbox, label_id, score in zip(
        pred.bboxes.cpu().numpy(),
        pred.labels.cpu().numpy(),
        pred.scores.cpu().numpy(),
        strict=True,
    ):
        x1, y1, x2, y2 = bbox.tolist()

        dets.append(
            {
                "label": classes[int(label_id)] if classes else str(int(label_id)),
                "bounding_box": [
                    x1 / width,
                    y1 / height,
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                ],
                "confidence": float(score),
            }
        )

    sample["predictions"] = detections_to_fiftyone(dets)
    sample.save()

# 4) Visualize in the FiftyOne App
session = fo.launch_app(dataset)
session.wait()
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
from visdet.apis import inference_detector, init_detector
from visdet.engine.config import Config

# Load config
cfg = Config.fromfile("config.py")

# Modify config if needed
cfg.model.test_cfg.score_thr = 0.5

# Initialize model
model = init_detector(cfg, "checkpoint.pth", device="cuda:0")

# Run inference
result = inference_detector(model, "image.jpg")
```

### Async Inference

For high-throughput scenarios:

```python
import asyncio

from visdet.apis import init_detector
from visdet.apis.inference import async_inference_detector

async def async_process():
    model = init_detector(config_file, checkpoint_file, device="cuda:0")
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
