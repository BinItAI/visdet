# Training

This page covers training workflows in MMDetection.

## Training Configuration

Training in MMDetection is controlled through configuration files. See the [Configuration](config.md) guide for details.

## Single GPU Training

To train a model on a single GPU:

```bash
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

### Optional Arguments

- `--work-dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume-from ${CHECKPOINT_FILE}`: Resume training from a previous checkpoint file.
- `--no-validate`: Whether not to evaluate the checkpoint during training.

## Multi-GPU Training

MMDetection supports distributed training with multiple GPUs using `torch.distributed.launch` or `slurm`.

### Using torch.distributed.launch

```bash
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Example with 8 GPUs:

```bash
bash ./tools/dist_train.sh configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py 8
```

### Using Slurm

If you run MMDetection on a cluster managed with slurm:

```bash
bash ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

## Training Tips

### Learning Rate

The default learning rate in config files is for 8 GPUs. If you use a different number of GPUs, you should scale the learning rate accordingly:

```
lr = base_lr * num_gpus / 8
```

### Mixed Precision Training

You can enable automatic mixed precision training by adding `--fp16` to the training command:

```bash
python tools/train.py ${CONFIG_FILE} --fp16
```

## Monitoring Training

### TensorBoard

MMDetection supports TensorBoard for monitoring training progress. To use it, add the following to your config file:

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

## See Also

- [Configuration System](config.md)
- [Customize Runtime Settings](customize_runtime.md)
- [Fine-tuning Models](finetune.md)
