# ruff: noqa
# type: ignore
_base_ = [
    "../_base_/models/cascade-mask-rcnn_r50_fpn.py",
    "../_base_/datasets/coco_instance.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]

pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth"

model = {
    "type": "CascadeRCNN",
    "backbone": {
        "_delete_": True,
        "type": "SwinTransformer",
        "embed_dims": 96,
        "depths": [2, 2, 18, 2],
        "num_heads": [3, 6, 12, 24],
        "window_size": 7,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "qk_scale": None,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.2,
        "patch_norm": True,
        "out_indices": (0, 1, 2, 3),
        "with_cp": False,
        "convert_weights": True,
        "init_cfg": {"type": "Pretrained", "checkpoint": pretrained},
    },
    "neck": {"in_channels": [96, 192, 384, 768]},
}

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    {"type": "LoadImageFromFile", "backend_args": {{_base_.backend_args}}},
    {"type": "LoadAnnotations", "with_bbox": True, "with_mask": True},
    {"type": "RandomFlip", "prob": 0.5},
    {
        "type": "RandomChoice",
        "transforms": [
            [
                {
                    "type": "RandomChoiceResize",
                    "scales": [
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    "keep_ratio": True,
                }
            ],
            [
                {
                    "type": "RandomChoiceResize",
                    "scales": [(400, 1333), (500, 1333), (600, 1333)],
                    "keep_ratio": True,
                },
                {
                    "type": "RandomCrop",
                    "crop_type": "absolute_range",
                    "crop_size": (384, 600),
                    "allow_negative_crop": True,
                },
                {
                    "type": "RandomChoiceResize",
                    "scales": [
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    "keep_ratio": True,
                },
            ],
        ],
    },
    {"type": "PackDetInputs"},
]
train_dataloader = {"dataset": {"pipeline": train_pipeline}}

max_epochs = 36
train_cfg = {"max_epochs": max_epochs}

# learning rate
param_scheduler = [
    {
        "type": "LinearLR",
        "start_factor": 0.001,
        "by_epoch": False,
        "begin": 0,
        "end": 1000,
    },
    {
        "type": "MultiStepLR",
        "begin": 0,
        "end": max_epochs,
        "by_epoch": True,
        "milestones": [27, 33],
        "gamma": 0.1,
    },
]

# optimizer
optim_wrapper = {
    "type": "OptimWrapper",
    "paramwise_cfg": {
        "custom_keys": {
            "absolute_pos_embed": {"decay_mult": 0.0},
            "relative_position_bias_table": {"decay_mult": 0.0},
            "norm": {"decay_mult": 0.0},
        }
    },
    "optimizer": {
        "_delete_": True,
        "type": "AdamW",
        "lr": 0.0001,
        "betas": (0.9, 0.999),
        "weight_decay": 0.05,
    },
}
