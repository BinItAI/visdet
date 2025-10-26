# ruff: noqa
"""Base module for visdet."""

from typing import Any, Dict, Optional

import torch.nn as nn


class BaseModule(nn.Module):
    """Base module class for visdet.

    This is a minimal implementation for type checking.
    In a full implementation, this would provide enhanced features like
    initialization control, profiling, etc.
    """

    def __init__(self, init_cfg: Optional[Dict] = None) -> None:
        """Initialize module.

        Args:
            init_cfg: Config dict for weight initialization
        """
        super().__init__()
        self.init_cfg = init_cfg

    def _init_weights(self) -> None:
        """Initialize weights."""
        pass


__all__ = ["BaseModule"]
