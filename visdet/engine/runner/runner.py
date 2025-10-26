# ruff: noqa
"""
Runner class for visdet training.

This module provides the main training runner.
"""

from typing import Any, Dict, Optional


class Runner:
    """Stub runner class for visdet.

    This is a minimal implementation for type checking.
    In a full implementation, this would handle training loops, validation, etc.
    """

    def __init__(self) -> None:
        """Initialize runner."""
        self.iter = 0
        self.epoch = 0
        self.work_dir = ""
        self.timestamp = ""
        self._hooks = []

    def register_hook(self, hook: Any) -> None:
        """Register a hook.

        Args:
            hook: Hook instance to register
        """
        self._hooks.append(hook)


__all__ = ["Runner"]
