# ruff: noqa
"""
Image input/output utilities.

This module provides functions for reading and writing images.
"""

from typing import Optional, Union
import io
import numpy as np
import cv2


def imfrombytes(
    content: bytes,
    flag: str = "color",
    channel_order: str = "bgr",
) -> np.ndarray:
    """Read an image from bytes.

    Args:
        content: Image bytes
        flag: How to read the image ('color', 'grayscale', etc)
        channel_order: Channel order ('bgr' or 'rgb')

    Returns:
        Numpy array of the image
    """
    img_array = np.frombuffer(content, dtype=np.uint8)
    imread_flag = cv2.IMREAD_COLOR if flag == "color" else cv2.IMREAD_GRAYSCALE
    img = cv2.imdecode(img_array, imread_flag)

    if channel_order == "rgb" and flag == "color":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def imwrite(
    img: np.ndarray,
    file_path: str,
    params: Optional[list] = None,
) -> None:
    """Write an image to file.

    Args:
        img: Image array (BGR format)
        file_path: Path to save the image
        params: Encoding parameters for cv2.imwrite
    """
    cv2.imwrite(file_path, img, params)


__all__ = ["imfrombytes", "imwrite"]
