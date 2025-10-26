# ruff: noqa
"""
Logging module for visdet.

Provides logging utilities for training and inference.
"""


class MMLogger:
    """Stub logger class for visdet.

    This is a minimal implementation for type checking.
    In a full implementation, this would wrap mmengine logging.
    """

    def __init__(self, name: str = "visdet") -> None:
        """Initialize logger."""
        self.name = name

    def debug(self, msg: str) -> None:
        """Log debug message."""
        pass

    def info(self, msg: str) -> None:
        """Log info message."""
        pass

    def warning(self, msg: str) -> None:
        """Log warning message."""
        pass

    def error(self, msg: str) -> None:
        """Log error message."""
        pass


__all__ = ["MMLogger"]
