# Copyright (c) OpenMMLab. All rights reserved.
"""Tests the Assigner objects.

CommandLine:
    pytest tests/test_utils/test_assigner.py
    xdoctest tests/test_utils/test_assigner.py zero
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="Test suite for multiple detector types - only Mask R-CNN is supported in this build"
)


def test_max_iou_assigner():
    pass


def test_max_iou_assigner_with_ignore():
    pass


def test_max_iou_assigner_with_empty_gt():
    pass


def test_max_iou_assigner_with_empty_pred():
    pass


def test_approx_max_iou_assigner():
    pass


def test_atss_assigner():
    pass


def test_hungarian_assigner():
    pass


def test_sim_ota_assigner():
    pass


def test_uniform_assigner():
    pass


def test_task_aligned_assigner():
    pass


def test_point_assigner():
    pass


def test_center_region_assigner():
    pass


def test_mask_hungarian_assigner():
    pass


def test_ascend_max_iou_assigner():
    pass
