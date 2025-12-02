_base_ = ["mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py"]

# Override the dataset settings
dataset_type = "CocoDataset"
classes = ("bus", "car")
data_root = "data/coco/"

# img_norm_cfg is moved to model.data_preprocessor

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]

data = dict(
    _delete_=True,
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file="annotations/instances_train2017.json",
        data_prefix=dict(img_path="train2017/"),
        pipeline=train_pipeline,
    ),
)

val_evaluator = None
evaluation = None

# Override the model head to match the number of classes
model = dict(
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    roi_head=dict(bbox_head=dict(num_classes=2), mask_head=dict(num_classes=2)),
)

runner = dict(max_epochs=1)
work_dir = "./tutorial_exps"
