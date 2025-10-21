import numpy as np
import torch
from torch import Tensor


def imrenormalize(img: Tensor | np.ndarray, img_norm_cfg: dict, new_img_norm_cfg: dict) -> Tensor | np.ndarray:
    """Re-normalize the image.

    Args:
        img (Tensor | ndarray): Input image. If the input is a Tensor, the
            shape is (1, C, H, W). If the input is a ndarray, the shape
            is (H, W, C).
        img_norm_cfg (dict): Original configuration for the normalization.
        new_img_norm_cfg (dict): New configuration for the normalization.

    Returns:
        Tensor | ndarray: Output image with the same type and shape of
        the input.
    """
    if isinstance(img, torch.Tensor):
        assert img.ndim == 4 and img.shape[0] == 1
        new_img = img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        new_img = _imrenormalize(new_img, img_norm_cfg, new_img_norm_cfg)
        new_img = new_img.transpose(2, 0, 1)[None]
        return torch.from_numpy(new_img).to(img)
    else:
        return _imrenormalize(img, img_norm_cfg, new_img_norm_cfg)


def _imrenormalize(img: Tensor | np.ndarray, img_norm_cfg: dict, new_img_norm_cfg: dict) -> Tensor | np.ndarray:
    """Re-normalize the image."""
    img_norm_cfg = img_norm_cfg.copy()
    new_img_norm_cfg = new_img_norm_cfg.copy()
    for k, v in img_norm_cfg.items():
        if (k == "mean" or k == "std") and not isinstance(v, np.ndarray):
            img_norm_cfg[k] = np.array(v, dtype=img.dtype)
    # reverse cfg
    if "bgr_to_rgb" in img_norm_cfg:
        img_norm_cfg["rgb_to_bgr"] = img_norm_cfg["bgr_to_rgb"]
        img_norm_cfg.pop("bgr_to_rgb")
    for k, v in new_img_norm_cfg.items():
        if (k == "mean" or k == "std") and not isinstance(v, np.ndarray):
            new_img_norm_cfg[k] = np.array(v, dtype=img.dtype)
    # Denormalize
    mean = img_norm_cfg.get("mean", [0, 0, 0])
    std = img_norm_cfg.get("std", [1, 1, 1])
    to_bgr = img_norm_cfg.get("rgb_to_bgr", False)

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
        img = img * std + mean
        if to_bgr and img.shape[-1] == 3:
            img = img[..., ::-1]

    # Normalize with new config
    mean = new_img_norm_cfg.get("mean", [0, 0, 0])
    std = new_img_norm_cfg.get("std", [1, 1, 1])
    to_rgb = new_img_norm_cfg.get("bgr_to_rgb", False)

    if isinstance(img, torch.Tensor):
        mean = torch.tensor(mean, dtype=img.dtype, device=img.device)
        std = torch.tensor(std, dtype=img.dtype, device=img.device)
        if img.dim() == 4:  # (N, C, H, W)
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        elif img.dim() == 3:  # (C, H, W)
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
        if to_rgb and img.shape[-3] == 3:
            img = img.flip(-3)
        img = (img - mean) / std
    else:  # numpy array
        if to_rgb and img.shape[-1] == 3:
            img = img[..., ::-1].copy()
        img = (img - mean) / std

    return img
