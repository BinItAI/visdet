# Copyright (c) OpenMMLab. All rights reserved.

"""Downstream compatibility smoke test.

visdet no longer vendors MMDetection-style Python config inheritance, which the
upstream MMTrack configs relied on.

This test now only runs when `mmtrack` is installed and verifies the import
path, without depending on external `configs/_base_/*.py`.
"""

from __future__ import annotations

import pytest

mmtrack = pytest.importorskip("mmtrack")


def test_mmtrack_importable():
    assert mmtrack is not None
