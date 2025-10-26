# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import torch


def imnormalize(img, mean, std, to_rgb=True):
    """Normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for image normalize.
        std (ndarray): The std to be used for image normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    img = img.copy().astype(np.float32)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    assert img.dtype != np.uint8
    mean = mean.reshape(1, -1)
    std = std.reshape(1, -1)

    if to_rgb:
        cv2 = None
        try:
            import cv2
        except ImportError:
            pass

        if cv2 is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = img[..., ::-1]

    img = (img - mean) / std
    return img


def imdenormalize(img, mean, std, to_bgr=True):
    """Denormalize an image with mean and std.

    Args:
        img (ndarray or Tensor): Image to be denormalized.
        mean (ndarray): The mean to be used for image denormalize.
        std (ndarray): The std to be used for image denormalize.
        to_bgr (bool): Whether to convert to bgr.

    Returns:
        ndarray or Tensor: The denormalized image.
    """
    if isinstance(img, torch.Tensor):
        mean = torch.tensor(mean, dtype=img.dtype, device=img.device)
        std = torch.tensor(std, dtype=img.dtype, device=img.device)
        if img.dim() == 4:  # (N, C, H, W)
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        elif img.dim() == 3:  # (C, H, W)
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
        img = img * std + mean
        if to_bgr and img.shape[-3] == 3:
            img = img.flip(-3)
    else:  # numpy array
        mean = np.array(mean, dtype=img.dtype)
        std = np.array(std, dtype=img.dtype)
        img = img * std + mean
        if to_bgr and img.shape[-1] == 3:
            cv2 = None
            try:
                import cv2
            except ImportError:
                pass

            if cv2 is not None:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img = img[..., ::-1].copy()

    return img


def hsv2bgr(img: np.ndarray) -> np.ndarray:
    """Convert a HSV image to BGR image.

    Args:
        img (ndarray): The input HSV image.

    Returns:
        ndarray: The converted BGR image.
    """
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
