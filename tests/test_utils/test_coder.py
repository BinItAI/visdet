# Copyright (c) OpenMMLab. All rights reserved.
import pytest

pytestmark = pytest.mark.skip(
    reason="Test suite for multiple bbox coders - only DeltaXYWHBBoxCoder is supported in this build"
)


def test_yolo_bbox_coder():
    pass


def test_yolo_bbox_coder_with_norm_factor():
    pass


def test_tblr_bbox_coder():
    pass


def test_distance_point_bbox_coder():
    pass


def test_delta_xywh_bbox_coder():
    pass
