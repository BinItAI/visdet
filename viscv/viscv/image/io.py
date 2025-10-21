# Copyright (c) OpenMMLab. All rights reserved.
import io
import os.path as osp
import warnings
from typing import Optional

import cv2
import numpy as np
from cv2 import (
    IMREAD_COLOR,
    IMREAD_GRAYSCALE,
    IMREAD_IGNORE_ORIENTATION,
    IMREAD_UNCHANGED,
)

try:
    from turbojpeg import TJCS_RGB, TJPF_BGR, TJPF_GRAY, TurboJPEG
except ImportError:
    TJCS_RGB = TJPF_GRAY = TJPF_BGR = TurboJPEG = None

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None

try:
    import tifffile
except ImportError:
    tifffile = None

jpeg = None
supported_backends = ["cv2", "turbojpeg", "pillow", "tifffile"]

imread_flags = {
    "color": IMREAD_COLOR,
    "grayscale": IMREAD_GRAYSCALE,
    "unchanged": IMREAD_UNCHANGED,
    "color_ignore_orientation": IMREAD_IGNORE_ORIENTATION | IMREAD_COLOR,
    "grayscale_ignore_orientation": IMREAD_IGNORE_ORIENTATION | IMREAD_GRAYSCALE,
}

imread_backend = "cv2"


def imfrombytes(
    content: bytes,
    flag: str = "color",
    channel_order: str = "bgr",
    backend: str | None = None,
) -> np.ndarray:
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Same as :func:`imread`.
        channel_order (str): The channel order of the output, candidates
            are 'bgr' and 'rgb'. Default to 'bgr'.
        backend (str | None): The image decoding backend type. Options are
            `cv2`, `pillow`, `turbojpeg`, `tifffile`, `None`. If backend is
            None, the global imread_backend specified by ``use_backend()`` will
            be used. Default: None.

    Returns:
        ndarray: Loaded image array.
    """
    if backend is None:
        backend = imread_backend
    if backend not in supported_backends:
        raise ValueError(
            f"backend: {backend} is not supported. Supported backends are 'cv2', 'turbojpeg', 'pillow', 'tifffile'"
        )

    if backend == "turbojpeg":
        img = _jpegflag(flag, channel_order)
        if jpeg is None:
            raise ImportError("`PyTurboJPEG` is not installed")
        img = jpeg.decode(content, img)
        if channel_order == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    elif backend == "pillow":
        if Image is None:
            raise ImportError("`Pillow` is not installed")
        with io.BytesIO(content) as buff:
            img = Image.open(buff)
            img = _pillow2array(img, flag, channel_order)
        return img

    elif backend == "tifffile":
        if tifffile is None:
            raise ImportError("`tifffile` is not installed")
        with io.BytesIO(content) as buff:
            img = tifffile.imread(buff)
        return img

    else:
        # cv2 backend
        if len(content) == 0:
            return None
        img_np = np.frombuffer(content, np.uint8)
        flag = imread_flags[flag] if isinstance(flag, str) else flag
        img = cv2.imdecode(img_np, flag)
        if img is not None and flag == IMREAD_COLOR and channel_order == "rgb":
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        return img


def _jpegflag(flag: str = "color", channel_order: str = "bgr"):
    channel_order = channel_order.lower()
    if channel_order not in ["rgb", "bgr"]:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == "color":
        if channel_order == "bgr":
            return TJPF_BGR
        elif channel_order == "rgb":
            return TJCS_RGB
    elif flag == "grayscale":
        return TJPF_GRAY
    else:
        raise ValueError('flag must be "color" or "grayscale"')


def _pillow2array(img, flag: str = "color", channel_order: str = "bgr") -> np.ndarray:
    """Convert a pillow image to numpy array.

    Args:
        img (:obj:`PIL.Image.Image`): The image loaded using PIL
        flag (str): Flags specifying the color type of a loaded image,
            candidates are 'color', 'grayscale' and 'unchanged'.
            Default to 'color'.
        channel_order (str): The channel order of the output image array,
            candidates are 'bgr' and 'rgb'. Default to 'bgr'.

    Returns:
        np.ndarray: The converted numpy array
    """
    channel_order = channel_order.lower()
    if channel_order not in ["rgb", "bgr"]:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == "unchanged":
        array = np.array(img)
        if array.ndim >= 3 and array.shape[2] >= 3:  # color image
            array[:, :, :3] = array[:, :, (2, 1, 0)]  # RGB to BGR
    else:
        # Handle exif orientation tag
        if flag in ["color", "grayscale"]:
            if hasattr(img, "_getexif") and img._getexif() is not None:
                # Not all images have exif info
                exif = img._getexif()
                orientation = exif.get(274, 1)  # 274 is the orientation tag id
                if orientation > 1:
                    img = ImageOps.exif_transpose(img)

        # If the image mode is not 'RGB', convert it to 'RGB' first.
        if flag == "color":
            if img.mode != "RGB":
                if img.mode != "LA":
                    # Most formats except 'LA' can be directly converted to RGB
                    img = img.convert("RGB")
                else:
                    # When the mode is 'LA', the default conversion will fill in
                    #  the canvas with black, which sometimes shadows black objects
                    #  in the foreground.
                    # Therefore, a random color (124, 117, 104) is used for canvas
                    img_rgba = img.convert("RGBA")
                    img = Image.new("RGB", img_rgba.size, (124, 117, 104))
                    img.paste(img_rgba, mask=img_rgba.split()[3])  # 3 is alpha
            if channel_order == "bgr":
                array = np.array(img, dtype=np.uint8)
                array = array[:, :, ::-1]  # RGB to BGR
            else:
                array = np.array(img, dtype=np.uint8)
        elif flag == "grayscale":
            img = img.convert("L")
            array = np.array(img, dtype=np.uint8)

    return array


def imwrite(
    img: np.ndarray,
    file_path: str,
    params: list | None = None,
    auto_mkdir: bool | None = None,
) -> bool:
    """Write image to file.

    Warning:
        The parameter `auto_mkdir` will be deprecated in the future and every
        file clients will make directory automatically.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically. It will be deprecated.

    Returns:
        bool: Successful or not.

    Examples:
        >>> # write to hard disk client
        >>> ret = viscv.imwrite(img, '/path/to/img.jpg')
    """
    file_path = str(file_path)
    if auto_mkdir is not None:
        warnings.warn(
            "The parameter `auto_mkdir` will be deprecated in the future and "
            "every file clients will make directory automatically."
        )

    # Create directory if it doesn't exist
    dir_name = osp.dirname(file_path)
    if dir_name and not osp.exists(dir_name):
        import os

        os.makedirs(dir_name, exist_ok=True)

    img_ext = osp.splitext(file_path)[-1]
    # Encode image according to image suffix.
    # For example, if image path is '/path/your/img.jpg', the encode
    # format is '.jpg'.
    flag, img_buff = cv2.imencode(img_ext, img, params)

    if flag:
        with open(file_path, "wb") as f:
            f.write(img_buff.tobytes())

    return flag
