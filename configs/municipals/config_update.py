classes = (
    "Aluminum Can",
    "Bottle Cap",
    "Brown Paper",
    "Cardboard",
    "Clear LDPE",
    "Clear PET Bottle",
    "Colored HDPE",
    "Colored LDPE",
    "Colored PET Bottle",
    "Colored Paper",
    "E-Waste",
    "Fabric",
    "Ferromagnetic Metals",
    "Glass",
    "Letter Paper",
    "Magazine",
    "Mixed/Other Metals",
    "Mixed/Other Paper",
    "Mixed/Other Plastics",
    "Molded Pulp",
    "Natural HDPE",
    "Newspaper",
    "Other Aluminum",
    "Other PET",
    "PVC",
    "Polypropylene",
    "Polystyrene",
    "Wood",
)


# The new config inherits a base config to highlight the necessary modification
_base_ = "../faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py"


# We also need to change the num_classes in head to match the dataset's annotation
model = dict(roi_head=dict(bbox_head=dict(num_classes=len(classes))))


data_root = "/home/ubuntu/vision-research/data/municipals/2023-03-22_13-21/"

# should be able to register all hooks here?
custom_hooks = [dict(type="TensorboardLoggerHook", log_dir="./output/four")]

optimizer = dict(type="SGD", lr=0.0001, momentum=0.9, weight_decay=0.0001)

data = dict(
    train=dict(
        img_prefix="data",
        classes=classes,
        ann_file="train_2023-03-31_09-23-33.118937.json",
        data_root=data_root,
    ),
    val=dict(
        img_prefix="data",
        classes=classes,
        ann_file="val.json",
        data_root=data_root,
    ),
    test=dict(
        img_prefix="data",
        classes=classes,
        ann_file="val.json",
        data_root=data_root,
    ),
)
