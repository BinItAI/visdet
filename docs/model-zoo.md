# Model Zoo

visdet’s “supported models” are the YAML model presets under `configs/models/`.

At runtime these are discovered automatically (via `MODEL_PRESETS`), so adding a new YAML file under `configs/models/` makes it available everywhere that accepts a model preset name (e.g. `SimpleRunner(model=...)`, `DetInferencer(model=...)`).

## Using A Model Preset

### Inference

```python
from visdet.apis import DetInferencer

# Either use the preset name…
infer = DetInferencer(model="rtmdet_s")

# …or (when present) an alias from `preset_meta.aliases`
infer = DetInferencer(model="rtmdet-s")

result = infer("image.jpg")
```

### Training

```python
from visdet import SimpleRunner

runner = SimpleRunner(
    model="rtmdet_s",
    dataset="coco_detection",
    optimizer="adamw_default",
    scheduler="1cycle",
)
runner.train()
```

## Supported Models (from `configs/models`)

### Two-Stage Detectors

- `faster_rcnn_r50`
- `mask_rcnn_r50`
- `mask_rcnn_swin_s`

### RTMDet (Object Detection)

- `rtmdet_tiny` (aliases: `rtmdet-t`, `rtmdet-tiny`)
- `rtmdet_s` (alias: `rtmdet-s`)
- `rtmdet_m` (alias: `rtmdet-m`)
- `rtmdet_l` (alias: `rtmdet-l`)
- `rtmdet_x` (alias: `rtmdet-x`)

### RTMDet-Ins (Instance Segmentation)

- `rtmdet_ins_tiny` (aliases: `rtmdet-ins-t`, `rtmdet-ins-tiny`)
- `rtmdet_ins_s` (alias: `rtmdet-ins-s`)
- `rtmdet_ins_m` (alias: `rtmdet-ins-m`)
- `rtmdet_ins_l` (alias: `rtmdet-ins-l`)
- `rtmdet_ins_x` (alias: `rtmdet-ins-x`)

## Discoverability

```python
from visdet import SimpleRunner

print(SimpleRunner.list_models())
```

```python
from visdet.apis import DetInferencer

print(DetInferencer.list_models())
```
