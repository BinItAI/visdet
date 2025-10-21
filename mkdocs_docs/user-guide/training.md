# Training

This guide covers training models in visdet.

## Quick Start

To train a model with a single GPU:

```bash
python tools/train.py ${CONFIG_FILE}
```

## Multi-GPU Training

For distributed training across multiple GPUs:

```bash
bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
```

Example with 8 GPUs:

```bash
bash tools/dist_train.sh configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py 8
```

## Configuration

Training behavior is controlled through configuration files. See the [Configuration Guide](configuration.md) for details.

### Key Configuration Options

- **Learning Rate**: Adjust based on batch size and number of GPUs
- **Epochs**: Number of training epochs
- **Batch Size**: Samples per GPU
- **Optimizer**: Adam, SGD, AdamW, etc.

## Monitoring Training

### TensorBoard

Enable TensorBoard logging in your config:

```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
```

Then run:

```bash
tensorboard --logdir=work_dirs/
```

### Checkpoints

Model checkpoints are saved to `work_dirs/` by default. Configure checkpoint behavior:

```python
checkpoint_config = dict(interval=1)  # Save every epoch
```

## Advanced Topics

### Mixed Precision Training

Enable FP16 training for faster training and reduced memory usage:

```bash
python tools/train.py ${CONFIG_FILE} --fp16
```

### Resume Training

Resume from a checkpoint:

```bash
python tools/train.py ${CONFIG_FILE} --resume-from ${CHECKPOINT_FILE}
```

### Fine-tuning

Load pretrained weights and train:

```bash
python tools/train.py ${CONFIG_FILE} --load-from ${CHECKPOINT_FILE}
```

## See Also

- [Configuration Guide](configuration.md)
- [Inference Guide](inference.md)
- [Training Tutorial](../tutorials/training.md)
