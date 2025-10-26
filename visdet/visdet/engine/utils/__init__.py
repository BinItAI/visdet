# ruff: noqa
"""
Utility functions for visdet.

Provides common utility functions for training and inference.
"""


def digit_version(version_str: str) -> tuple:
    """Convert version string to a tuple of digits.

    Args:
        version_str: Version string like "1.2.3"

    Returns:
        Tuple of integers representing the version
    """
    try:
        return tuple(int(d) for d in version_str.split("."))
    except (ValueError, AttributeError):
        return (0,)


def to_2tuple(x):
    """Convert input to a 2-tuple."""
    if isinstance(x, (tuple, list)):
        return tuple(x) if len(x) == 2 else (x[0], x[0])
    return (x, x)


__all__ = ["digit_version", "to_2tuple"]
