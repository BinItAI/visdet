# ruff: noqa
"""
Data structures module for visdet.

Provides data container classes for training and inference.
"""

from typing import Any, Dict


class InstanceData:
    """Stub instance data container for visdet.

    This is a minimal implementation for type checking.
    In a full implementation, this would provide structured data access.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize instance data."""
        self._data: Dict[str, Any] = kwargs

    def __getattr__(self, name: str) -> Any:
        """Get attribute from data."""
        if name.startswith("_"):
            return super().__getattribute__(name)
        return self._data.get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute in data."""
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            if not hasattr(self, "_data"):
                super().__setattr__("_data", {})
            self._data[name] = value


__all__ = ["InstanceData"]
