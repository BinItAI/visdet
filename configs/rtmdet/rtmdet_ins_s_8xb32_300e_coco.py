_base_ = ["./rtmdet_ins_l_8xb32_300e_coco.py"]

model = dict(
    backbone=dict(deepen_factor=0.33, widen_factor=0.5),
    neck=dict(in_channels=[128, 256, 512], out_channels=128, num_csp_blocks=1),
    bbox_head=dict(in_channels=128, feat_channels=128),
)
