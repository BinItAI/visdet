# ruff: noqa
"""
Logging module for visdet.

Provides logging utilities for training and inference.
"""

import sys
from typing import Any, Dict, Optional


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


class MessageHub:
    """Stub MessageHub for visdet.

    Used to collect and manage messages during training.
    """

    def __init__(self) -> None:
        """Initialize message hub."""
        self.messages: Dict[str, Any] = {}

    def log_scalars(self, scalars: Dict[str, float], step: int = 0) -> None:
        """Log scalar values.

        Args:
            scalars: Dictionary of scalar values to log
            step: Step/iteration number
        """
        self.messages[step] = scalars

    def get_scalar(self, key: str, step: int = 0) -> Optional[float]:
        """Get a scalar value.

        Args:
            key: Key to retrieve
            step: Step/iteration number

        Returns:
            Scalar value if found
        """
        if step in self.messages:
            return self.messages[step].get(key)
        return None


def print_log(msg: str, logger: Optional[MMLogger] = None) -> None:
    """Print log message.

    Args:
        msg: Message to print
        logger: Optional logger instance
    """
    if logger is not None:
        logger.info(msg)
    else:
        print(msg, file=sys.stdout)


__all__ = ["MMLogger", "MessageHub", "print_log"]
