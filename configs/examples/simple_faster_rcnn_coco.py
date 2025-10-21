"""
Simple Faster R-CNN Configuration for COCO Detection

This is a minimal, well-commented configuration file that demonstrates
how to set up training for object detection using visdet.

This config is designed to be easy to understand and modify for beginners.
It inherits from base configs to keep things simple and maintainable.

Usage:
    python scripts/train_simple.py --config configs/examples/simple_faster_rcnn_coco.py
"""

# =============================================================================
# Base Configurations
# =============================================================================
# visdet uses a modular config system. We inherit from base configs to avoid
# repetition and make our config easier to maintain.

_base_ = [
    # Model architecture: Faster R-CNN with ResNet-50 backbone + FPN
    "../_base_/models/faster_rcnn_r50_fpn.py",
    # Dataset: COCO object detection dataset
    "../_base_/datasets/coco_detection.py",
    # Training schedule: 1x schedule (12 epochs for COCO)
    "../_base_/schedules/schedule_1x.py",
    # Runtime settings: logging, checkpointing, etc.
    "../_base_/default_runtime.py",
]

# =============================================================================
# Custom Settings
# =============================================================================
# You can override any settings from the base configs here.
# Common things to modify:

# --- Dataset Settings ---
# If your COCO dataset is in a different location, uncomment and modify:
# data_root = '/path/to/your/coco'

# If you want to change the batch size per GPU:
# data = dict(
#     samples_per_gpu=2,  # Default is usually 2
#     workers_per_gpu=2,  # Number of data loading workers
# )

# --- Training Settings ---
# If you want to change the number of training epochs:
# runner = dict(
#     type='EpochBasedRunner',
#     max_epochs=24,  # Train for 24 epochs instead of default 12
# )

# If you want to modify the learning rate:
# optimizer = dict(
#     type='SGD',
#     lr=0.02,  # Default is 0.02 for 2 GPUs with batch_size=2
#     momentum=0.9,
#     weight_decay=0.0001
# )

# --- Model Settings ---
# If you want to change the number of classes (e.g., for a custom dataset):
# model = dict(
#     roi_head=dict(
#         bbox_head=dict(
#             num_classes=80  # COCO has 80 classes
#         )
#     )
# )

# --- Checkpoint Settings ---
# Configure how often to save checkpoints:
# checkpoint_config = dict(
#     interval=1,  # Save checkpoint every epoch
#     max_keep_ckpts=3,  # Keep only the 3 most recent checkpoints
# )

# --- Logging Settings ---
# Configure logging interval:
# log_config = dict(
#     interval=50,  # Log every 50 iterations
#     hooks=[
#         dict(type='TextLoggerHook'),
#     ]
# )

# =============================================================================
# Tips for Customization
# =============================================================================
# 1. **Custom Dataset**: To train on your own dataset, you'll need to:
#    - Convert your annotations to COCO format, OR
#    - Create a custom dataset class
#    - Update `data_root` and possibly `dataset_type`
#
# 2. **Transfer Learning**: To fine-tune from pretrained weights:
#    - Use --pretrained flag with train_simple.py, OR
#    - Set load_from = 'path/to/checkpoint.pth' here
#
# 3. **Faster Training**: To speed up training:
#    - Increase batch_size (if GPU memory allows)
#    - Reduce input image size in the dataset config
#    - Use fewer training epochs
#
# 4. **Better Accuracy**: To improve detection accuracy:
#    - Train for more epochs
#    - Use multi-scale training
#    - Try data augmentation techniques
#    - Use a larger backbone (e.g., ResNet-101)
#
# For more details, see the documentation at:
# https://binitai.github.io/visdet/
