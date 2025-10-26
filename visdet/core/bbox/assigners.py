# Copyright (c) OpenMMLab. All rights reserved.
"""Bbox assigners - backward compatibility wrapper."""

from visdet.models.task_modules.assigners import BaseAssigner, MaxIoUAssigner
from visdet.models.task_modules.assigners.assign_result import AssignResult

__all__ = ["MaxIoUAssigner", "BaseAssigner", "AssignResult"]
