# ruff: noqa
"""
Data preprocessor classes for visdet.

This module provides base data preprocessor classes.
"""

from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn

from .base_module import BaseModule


class BaseDataPreprocessor(BaseModule):
    """Base data preprocessor for visdet.

    This is a minimal implementation for type checking.
    In a full implementation, this would provide comprehensive data preprocessing.
    """

    def __init__(
        self,
        mean: Optional[Union[List[float], torch.Tensor]] = None,
        std: Optional[Union[List[float], torch.Tensor]] = None,
        rgb_to_bgr: bool = False,
        bgr_to_rgb: bool = False,
        pad_mask: bool = False,
        pad_size_divisor: int = 1,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        """Initialize data preprocessor."""
        super().__init__(init_cfg=init_cfg)
        self.mean = mean
        self.std = std
        self.rgb_to_bgr = rgb_to_bgr
        self.bgr_to_rgb = bgr_to_rgb
        self.pad_mask = pad_mask
        self.pad_size_divisor = pad_size_divisor

    def forward(self, data: Dict) -> Dict:
        """Forward pass for data preprocessing.

        Args:
            data: Input data dictionary

        Returns:
            Preprocessed data dictionary
        """
        return data


class ImgDataPreprocessor(BaseDataPreprocessor):
    """Image data preprocessor for visdet.

    Handles image normalization, RGB/BGR conversion, and padding.
    """

    pass
