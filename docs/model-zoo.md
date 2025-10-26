# Model Zoo

visdet is a minimal version of MMDetection focusing on a single, high-performance configuration: **Swin Transformer + Mask R-CNN**.

## Supported Models

### Mask R-CNN with Swin Transformer

**Paper:** [Swin Transformer](https://arxiv.org/abs/2103.14030)

This combination provides state-of-the-art performance for instance segmentation using the Swin Transformer backbone with Mask R-CNN detection head.

#### Architecture Details

- **Backbone**: Swin Transformer (window-based shifted attention)
- **Detector**: Mask R-CNN (two-stage detector with RPN)
- **Neck**: Feature Pyramid Network (FPN)
- **Head**: Box head + Mask head
- **Dataset**: COCO (80 classes)

#### Quick Start

```python
from visdet.apis import init_detector

# Initialize model
config_file = 'configs/mask_rcnn/mask_rcnn_swin_tiny_fpn_1x_coco.py'
checkpoint = 'path/to/checkpoint.pth'

model = init_detector(config_file, checkpoint, device='cuda:0')

# Run inference
from visdet.apis import inference_detector
result = inference_detector(model, 'image.jpg')
```

## Configuration Files

Available configurations in `configs/mask_rcnn/`:

- `mask_rcnn_swin_tiny_fpn_1x_coco.py` - Swin Tiny backbone
- `mask_rcnn_swin_small_fpn_1x_coco.py` - Swin Small backbone
- `mask_rcnn_swin_base_fpn_1x_coco.py` - Swin Base backbone

## Training

To train a model, see the [Training Guide](user-guide/training.md) for detailed instructions.

## Performance

The Swin Transformer backbone has shown excellent performance on COCO:

- High accuracy for instance segmentation
- Efficient computation with window-based attention
- Strong transfer learning capabilities

## See Also

- [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)
- [Mask R-CNN Paper](https://arxiv.org/abs/1703.06870)
- [Training Guide](user-guide/training.md)
- [Configuration System](user-guide/configuration.md)
