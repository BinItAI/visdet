# visdet Scripts

This directory contains helpful scripts for common tasks with visdet.

## train_simple.py

A user-friendly training script that simplifies single-GPU training with sensible defaults.

### Features

- **Single-GPU training** with automatic device detection
- **Minimal CLI** for common use cases
- **Clear progress indicators** and error messages
- **Easy configuration** through both config files and CLI flags
- **Sensible defaults** for reproducibility (seed=42, deterministic mode available)

### Quick Start

The simplest way to get started:

```bash
# Train with a config file
python scripts/train_simple.py --config configs/examples/simple_faster_rcnn_coco.py
```

### Common Use Cases

#### 1. Train on Custom Dataset

```bash
python scripts/train_simple.py \
    --config configs/examples/simple_faster_rcnn_coco.py \
    --data-root /path/to/your/dataset \
    --epochs 24 \
    --work-dir work_dirs/my_custom_model
```

**Note**: Your dataset should be in COCO format. See the [documentation](https://binitai.github.io/visdet/tutorials/customize_dataset/) for details on custom datasets.

#### 2. Adjust Training Hyperparameters

```bash
python scripts/train_simple.py \
    --config configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    --batch-size 4 \
    --lr 0.01 \
    --epochs 36 \
    --seed 123
```

#### 3. Resume Training from Checkpoint

```bash
python scripts/train_simple.py \
    --config configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    --resume
```

This will automatically resume from the latest checkpoint in the work directory.

#### 4. Fine-tune from Pretrained Weights

```bash
python scripts/train_simple.py \
    --config configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    --pretrained checkpoints/faster_rcnn_r50_fpn_1x_coco.pth \
    --data-root /path/to/custom/data \
    --epochs 12 \
    --lr 0.001
```

### CLI Options

```
Required:
  --config CONFIG           Path to training config file

Common Overrides:
  --data-root PATH          Root directory for dataset
  --work-dir PATH           Directory to save logs and models
  --epochs N                Number of training epochs
  --batch-size N            Batch size per GPU
  --lr FLOAT                Learning rate

Training Control:
  --seed N                  Random seed (default: 42)
  --deterministic           Enable deterministic mode
  --device {cuda,cpu}       Device to use (default: cuda)
  --gpu-id N                GPU ID to use (default: 0)

Checkpointing:
  --resume                  Auto-resume from latest checkpoint
  --pretrained PATH         Path to pretrained model weights

Validation:
  --no-val                  Disable validation during training
```

### Understanding the Output

After training completes successfully, you'll find these files in the work directory:

```
work_dirs/my_model/
├── latest.pth              # Latest checkpoint
├── best_bbox_mAP_*.pth     # Best checkpoint based on validation mAP
├── epoch_*.pth             # Checkpoints at specific epochs
├── *.log                   # Training logs
└── *.log.json              # Structured log data
```

### Troubleshooting

#### CUDA Out of Memory

```bash
# Reduce batch size
python scripts/train_simple.py --config <config> --batch-size 1
```

#### Dataset Not Found

Make sure your dataset is in the correct location and format:
- For COCO: `data/coco/` with `annotations/` and `train2017/`, `val2017/` subdirectories
- Or use `--data-root` to specify a different location

See the [dataset documentation](https://binitai.github.io/visdet/tutorials/customize_dataset/) for more details.

#### Multiple GPUs Detected

This script is designed for single-GPU training. If you have multiple GPUs and want to use them all, use the main training script:

```bash
python tools/train.py <config> --launcher pytorch --gpus 4
```

### Advanced Usage

For more advanced use cases (multi-GPU, distributed training, custom hooks, etc.), use the main training script:

```bash
python tools/train.py <config> [options]
```

See `python tools/train.py --help` for all available options.

### Examples

The `configs/examples/` directory contains well-commented example configs:

- `simple_faster_rcnn_coco.py` - Minimal Faster R-CNN config with detailed comments

These are great starting points for understanding the config system and creating your own configurations.

### Tips

1. **Start Small**: Begin with a small dataset or fewer epochs to verify everything works
2. **Monitor Training**: Check the logs regularly to ensure training is progressing
3. **Experiment**: Try different learning rates, batch sizes, and augmentations
4. **Save Configs**: Keep track of configs that work well for your use case
5. **Use Checkpoints**: Resume training if interrupted to save time

### Next Steps

- Read the [Configuration Guide](https://binitai.github.io/visdet/tutorials/config/)
- Learn about [Custom Datasets](https://binitai.github.io/visdet/tutorials/customize_dataset/)
- Explore the [Model Zoo](https://binitai.github.io/visdet/model-zoo/)
- Check out [Advanced Training](https://binitai.github.io/visdet/tutorials/training/)
