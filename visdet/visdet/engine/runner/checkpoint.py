# ruff: noqa
"""
Checkpoint utilities for visdet.

This module provides checkpoint loading and saving utilities.
"""

from typing import Any, Dict, Optional
import torch


class CheckpointLoader:
    """Stub checkpoint loader for visdet.

    This is a minimal implementation for type checking.
    In a full implementation, this would handle various checkpoint formats.
    """

    @staticmethod
    def load_checkpoint(filename: str, map_location: Optional[str] = None) -> Dict[str, Any]:
        """Load checkpoint from file.

        Args:
            filename: Path to checkpoint file
            map_location: Device to load to

        Returns:
            Loaded checkpoint dict
        """
        return torch.load(filename, map_location=map_location)

    @staticmethod
    def save_checkpoint(model: Any, filename: str, optimizer: Optional[Any] = None, **kwargs: Any) -> None:
        """Save checkpoint to file.

        Args:
            model: Model to save
            filename: Path to save to
            optimizer: Optional optimizer to save
            **kwargs: Additional state to save
        """
        checkpoint = {"state_dict": model.state_dict()}
        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()
        checkpoint.update(kwargs)
        torch.save(checkpoint, filename)


__all__ = ["CheckpointLoader"]
