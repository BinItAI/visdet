# visdet justfile - common development commands

# Default recipe to list available commands
default:
    @just --list

# Run Mask R-CNN Swin training on COCO instance segmentation (default test)
train:
    uv run python scripts/train_simple.py \
        --model mask_rcnn_swin_s \
        --dataset coco_instance_segmentation \
        --epochs 12 \
        --work-dir ./work_dirs/mask_rcnn_swin_coco

# List available models
list-models:
    uv run python scripts/train_simple.py --list-models

# List available datasets
list-datasets:
    uv run python scripts/train_simple.py --list-datasets

# Show a preset configuration
show-preset preset category="model":
    uv run python scripts/train_simple.py --show-preset {{ preset }} --category {{ category }}
