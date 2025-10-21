# ruff: noqa
# type: ignore
# dataset settings
dataset_type = "CocoDataset"
data_root = "data/coco/"

backend_args = None

train_pipeline = [
    {"type": "LoadImageFromFile", "backend_args": backend_args},
    {"type": "LoadAnnotations", "with_bbox": True, "with_mask": True},
    {"type": "Resize", "scale": (1333, 800), "keep_ratio": True},
    {"type": "RandomFlip", "prob": 0.5},
    {"type": "PackDetInputs"},
]
test_pipeline = [
    {"type": "LoadImageFromFile", "backend_args": backend_args},
    {"type": "Resize", "scale": (1333, 800), "keep_ratio": True},
    # If you don't have a gt annotation, delete the pipeline
    {"type": "LoadAnnotations", "with_bbox": True, "with_mask": True},
    {
        "type": "PackDetInputs",
        "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    },
]
train_dataloader = {
    "batch_size": 2,
    "num_workers": 2,
    "persistent_workers": True,
    "sampler": {"type": "DefaultSampler", "shuffle": True},
    "batch_sampler": {"type": "AspectRatioBatchSampler"},
    "dataset": {
        "type": dataset_type,
        "data_root": data_root,
        "ann_file": "annotations/instances_train2017.json",
        "data_prefix": {"img": "train2017/"},
        "filter_cfg": {"filter_empty_gt": True, "min_size": 32},
        "pipeline": train_pipeline,
        "backend_args": backend_args,
    },
}
val_dataloader = {
    "batch_size": 1,
    "num_workers": 2,
    "persistent_workers": True,
    "drop_last": False,
    "sampler": {"type": "DefaultSampler", "shuffle": False},
    "dataset": {
        "type": dataset_type,
        "data_root": data_root,
        "ann_file": "annotations/instances_val2017.json",
        "data_prefix": {"img": "val2017/"},
        "test_mode": True,
        "pipeline": test_pipeline,
        "backend_args": backend_args,
    },
}
test_dataloader = val_dataloader

val_evaluator = {
    "type": "CocoMetric",
    "ann_file": data_root + "annotations/instances_val2017.json",
    "metric": ["bbox", "segm"],
    "format_only": False,
    "backend_args": backend_args,
}
test_evaluator = val_evaluator
