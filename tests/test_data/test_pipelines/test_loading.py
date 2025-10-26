# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import numpy as np
import pytest

import visdet.cv as mmcv
from visdet.datasets.pipelines import (
    FilterAnnotations,
    LoadImageFromFile,
    LoadImageFromWebcam,
    LoadMultiChannelImageFromFiles,
)
from visdet.structures.mask.structures import BitmapMasks, PolygonMasks


class TestLoading:
    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.join(osp.dirname(__file__), "../../data")

    def test_load_img(self):
        # New API: use img_path directly
        img_path = osp.join(self.data_prefix, "color.jpg")
        results = dict(img_path=img_path)
        transform = LoadImageFromFile()
        results = transform(copy.deepcopy(results))
        assert results["img_path"] == img_path
        assert results["img"].shape == (288, 512, 3)
        assert results["img"].dtype == np.uint8
        assert results["img_shape"] == (288, 512)  # img_shape is H×W only
        assert results["ori_shape"] == (288, 512)  # ori_shape is H×W only
        # Check __repr__ contains expected values
        assert "LoadImageFromFile" in repr(transform)
        assert "to_float32=False" in repr(transform)
        assert "color_type='color'" in repr(transform)

        # to_float32
        transform = LoadImageFromFile(to_float32=True)
        results = dict(img_path=img_path)
        results = transform(copy.deepcopy(results))
        assert results["img"].dtype == np.float32

        # gray image
        gray_path = osp.join(self.data_prefix, "gray.jpg")
        results = dict(img_path=gray_path)
        transform = LoadImageFromFile()
        results = transform(copy.deepcopy(results))
        assert results["img"].shape == (288, 512, 3)
        assert results["img"].dtype == np.uint8

        transform = LoadImageFromFile(color_type="unchanged")
        results = dict(img_path=gray_path)
        results = transform(copy.deepcopy(results))
        assert results["img"].shape == (288, 512)
        assert results["img"].dtype == np.uint8

    def test_load_multi_channel_img(self):
        # New API: use img_path directly (can be list)
        img_paths = [
            osp.join(self.data_prefix, "color.jpg"),
            osp.join(self.data_prefix, "color.jpg"),
        ]
        results = dict(img_path=img_paths)
        transform = LoadMultiChannelImageFromFiles()
        results = transform(copy.deepcopy(results))
        assert results["img"].shape == (288, 512, 3, 2)
        assert results["img"].dtype == np.uint8
        assert results["img_shape"] == (288, 512, 3, 2)
        # ori_shape matches what get set in transform
        assert "ori_shape" in results
        # Check __repr__ contains expected values
        assert "LoadMultiChannelImageFromFiles" in repr(transform)
        assert "to_float32=False" in repr(transform)
        assert "color_type='unchanged'" in repr(transform)

    def test_load_webcam_img(self):
        img = mmcv.imread(osp.join(self.data_prefix, "color.jpg"))
        results = dict(img=img)
        transform = LoadImageFromWebcam()
        results = transform(copy.deepcopy(results))
        assert results["img"].shape == (288, 512, 3)
        assert results["img"].dtype == np.uint8
        assert results["img_shape"] == (288, 512, 3)
        assert results["ori_shape"] == (288, 512, 3)


def _build_filter_annotations_args():
    kwargs = (
        dict(min_gt_bbox_wh=(100, 100)),
        dict(min_gt_bbox_wh=(100, 100), keep_empty=False),
        dict(min_gt_bbox_wh=(1, 1)),
        dict(min_gt_bbox_wh=(0.01, 0.01)),
        dict(min_gt_bbox_wh=(0.01, 0.01), by_mask=True),
        dict(by_mask=True),
        dict(by_box=False, by_mask=True),
    )
    targets = (None, 0, 1, 2, 1, 1, 1)

    return list(zip(targets, kwargs))


@pytest.mark.parametrize("target, kwargs", _build_filter_annotations_args())
def test_filter_annotations(target, kwargs):
    filter_ann = FilterAnnotations(**kwargs)
    bboxes = np.array([[2.0, 10.0, 4.0, 14.0], [2.0, 10.0, 2.1, 10.1]])
    raw_masks = np.zeros((2, 24, 24))
    raw_masks[0, 10:14, 2:4] = 1
    bitmap_masks = BitmapMasks(raw_masks, 24, 24)
    results = dict(gt_bboxes=bboxes, gt_masks=bitmap_masks)
    results = filter_ann(results)
    if results is not None:
        results = results["gt_bboxes"].shape[0]
    assert results == target

    polygons = [
        [np.array([2.0, 10.0, 4.0, 10.0, 4.0, 14.0, 2.0, 14.0])],
        [np.array([2.0, 10.0, 2.1, 10.0, 2.1, 10.1, 2.0, 10.1])],
    ]
    polygon_masks = PolygonMasks(polygons, 24, 24)

    results = dict(gt_bboxes=bboxes, gt_masks=polygon_masks)
    results = filter_ann(results)

    if results is not None:
        results = len(results.get("gt_masks").masks)

    assert results == target
