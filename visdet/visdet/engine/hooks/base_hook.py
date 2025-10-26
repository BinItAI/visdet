# ruff: noqa
# Copyright (c) OpenMMLab. All rights reserved.
"""Base Hook class for visdet."""

from typing import Any, Dict


class Hook:
    """Stub Hook base class for visdet.

    This is a minimal implementation for type checking.
    In a full implementation, this would come from mmengine.
    """

    rule: str = ""

    def before_train(self, runner: Any) -> None:
        """Called before training starts."""
        pass

    def after_train(self, runner: Any) -> None:
        """Called after training finishes."""
        pass

    def before_train_epoch(self, runner: Any) -> None:
        """Called before each epoch."""
        pass

    def after_train_epoch(self, runner: Any) -> None:
        """Called after each epoch."""
        pass

    def before_train_iter(self, runner: Any, batch_idx: int, data_batch: Dict) -> None:
        """Called before each iteration."""
        pass

    def after_train_iter(self, runner: Any, batch_idx: int, data_batch: Dict, outputs: Any) -> None:
        """Called after each iteration."""
        pass
