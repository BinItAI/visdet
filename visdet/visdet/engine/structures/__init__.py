# ruff: noqa
"""
Data structures module for visdet.

Provides data container classes for training and inference.
"""

from typing import Any, Dict


class BaseDataElement(dict):
    """Base data element class for visdet.

    Provides dict-like interface with attribute access.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize data element."""
        super().__init__(**kwargs)

    def __getattr__(self, name: str) -> Any:
        """Get attribute from data."""
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute in data."""
        self[name] = value

    def cpu(self) -> "BaseDataElement":
        """Move to CPU (stub implementation)."""
        return self

    def to(self, device: str) -> "BaseDataElement":
        """Move to device (stub implementation)."""
        return self


class InstanceData(BaseDataElement):
    """Instance data container for visdet.

    This is a minimal implementation for type checking.
    In a full implementation, this would provide structured data access.
    """

    pass


class PixelData(BaseDataElement):
    """Pixel-level data container for visdet.

    Used for mask, heatmap, and other pixel-level annotations.
    """

    pass


__all__ = ["BaseDataElement", "InstanceData", "PixelData"]
