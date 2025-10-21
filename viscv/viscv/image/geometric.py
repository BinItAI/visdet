# Copyright (c) OpenMMLab. All rights reserved.
# type: ignore
import numbers

import cv2
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


def _scale_size(
    size: tuple[int, int],
    scale: float | int | tuple[float, float] | tuple[int, int],
) -> tuple[int, int]:
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float | int | tuple(float) | tuple(int)): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, float | int):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)


cv2_interp_codes = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}

cv2_border_modes = {
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT,
    "wrap": cv2.BORDER_WRAP,
    "reflect_101": cv2.BORDER_REFLECT_101,
    "transparent": cv2.BORDER_TRANSPARENT,
    "isolated": cv2.BORDER_ISOLATED,
}


def imresize(
    img: np.ndarray,
    size: tuple[int, int],
    return_scale: bool = False,
    interpolation: str = "bilinear",
    out: np.ndarray | None = None,
    backend: str | None = None,
) -> tuple[np.ndarray, float, float] | np.ndarray:
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend.
        out (ndarray): The output destination.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, `cv2` will be used. Default: None.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
        `resized_img`.
    """
    h, w = img.shape[:2]
    if backend is None:
        backend = "cv2"

    if backend == "cv2":
        resized_img = cv2.resize(img, size, dst=out, interpolation=cv2_interp_codes[interpolation])
    else:
        raise ValueError(f"backend: {backend} is not supported for resize.")

    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


def rescale_size(old_size: tuple, scale: float | int | tuple[int, int], return_scale: bool = False) -> tuple:
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | int | tuple[int]): The scaling factor or maximum size.
            If it is a float number or an integer, then the image will be
            rescaled by this factor, else if it is a tuple of 2 integers, then
            the image will be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, float | int):
        if scale <= 0:
            raise ValueError(f"Invalid scale {scale}, must be positive.")
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    else:
        raise TypeError(f"Scale must be a number or tuple of int, but got {type(scale)}")

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def imrescale(
    img: np.ndarray,
    scale: float | int | tuple[int, int],
    return_scale: bool = False,
    interpolation: str = "bilinear",
    backend: str | None = None,
) -> np.ndarray | tuple[np.ndarray, float]:  # type: ignore[return]
    """Resize image while keeping the aspect ratio.

    Args:
        img (ndarray): The input image.
        scale (float | int | tuple[int]): The scaling factor or maximum size.
            If it is a float number or an integer, then the image will be
            rescaled by this factor, else if it is a tuple of 2 integers, then
            the image will be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image.
        interpolation (str): Same as :func:`resize`.
        backend (str | None): Same as :func:`resize`.

    Returns:
        ndarray: The rescaled image.
    """
    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    rescaled_img = imresize(img, new_size, interpolation=interpolation, backend=backend)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img


def imflip(img: np.ndarray, direction: str = "horizontal") -> np.ndarray:
    """Flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or
            "vertical" or "diagonal".

    Returns:
        ndarray: The flipped image.
    """
    assert direction in ["horizontal", "vertical", "diagonal"]
    if direction == "horizontal":
        return np.flip(img, axis=1)
    elif direction == "vertical":
        return np.flip(img, axis=0)
    else:
        return np.flip(img, axis=(0, 1))


def imrotate(
    img: np.ndarray,
    angle: float,
    center: tuple[float, float] | None = None,
    scale: float = 1.0,
    border_value: int = 0,
    interpolation: str = "bilinear",
    auto_bound: bool = False,
    border_mode: str = "constant",
) -> np.ndarray:
    """Rotate an image.

    Args:
        img (np.ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used.
        scale (float): Isotropic scale factor.
        border_value (int): Border value used in case of a constant border.
            Defaults to 0.
        interpolation (str): Same as :func:`resize`.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image.
        border_mode (str): Pixel extrapolation method. Defaults to 'constant'.

    Returns:
        np.ndarray: The rotated image.
    """
    if center is not None and auto_bound:
        raise ValueError("`auto_bound` conflicts with `center`")
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2_interp_codes[interpolation],
        borderMode=cv2_border_modes[border_mode],
        borderValue=border_value,
    )
    return rotated


def impad(
    img: np.ndarray,
    *,
    shape: tuple[int, int] | None = None,
    padding: int | tuple | None = None,
    pad_val: float | list = 0,
    padding_mode: str = "constant",
) -> np.ndarray:
    """Pad the given image to a certain shape or pad on all sides with
    specified padding mode and padding value.

    Args:
        img (ndarray): Image to be padded.
        shape (tuple[int]): Expected padding shape (h, w). Default: None.
        padding (int or tuple[int]): Padding on each border. If a single int is
            provided this is used to pad all borders. If tuple of length 2 is
            provided this is the padding on left/right and top/bottom
            respectively. If a tuple of length 4 is provided this is the
            padding for the left, top, right and bottom borders respectively.
            Default: None. Note that `shape` and `padding` can not be both
            set.
        pad_val (Number | Sequence[Number]): Values to be filled in padding
            areas when padding_mode is 'constant'. Default: 0.
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Default: constant.

            - constant: pads with a constant value, this value is specified
              with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        ndarray: The padded image.
    """

    assert (shape is not None) ^ (padding is not None)
    if shape is not None:
        width = max(shape[1] - img.shape[1], 0)
        height = max(shape[0] - img.shape[0], 0)
        padding = (0, 0, width, height)

    # check pad_val
    if isinstance(pad_val, tuple):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError(f"pad_val must be a int or a tuple. But received {type(pad_val)}")

    # check padding
    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError(f"Padding must be a int or a 2, or 4 element tuple.But received {padding}")

    # check padding mode
    assert padding_mode in ["constant", "edge", "reflect", "symmetric"]

    border_type = {
        "constant": cv2.BORDER_CONSTANT,
        "edge": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT_101,
        "symmetric": cv2.BORDER_REFLECT,
    }
    img = cv2.copyMakeBorder(
        img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        border_type[padding_mode],
        value=pad_val,
    )

    return img


def _get_shear_matrix(magnitude: int | float, direction: str = "horizontal") -> np.ndarray:
    """Generate the shear matrix for transformation.

    Args:
        magnitude (int | float): The magnitude used for shear.
        direction (str): The flip direction, either "horizontal"
            or "vertical".

    Returns:
        ndarray: The shear matrix with dtype float32.
    """
    if direction == "horizontal":
        shear_matrix = np.float32([[1, magnitude, 0], [0, 1, 0]])
    elif direction == "vertical":
        shear_matrix = np.float32([[1, 0, 0], [magnitude, 1, 0]])
    return shear_matrix


def imshear(
    img: np.ndarray,
    magnitude: int | float,
    direction: str = "horizontal",
    border_value: int | tuple[int, int] = 0,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Shear an image.

    Args:
        img (ndarray): Image to be sheared with format (h, w)
            or (h, w, c).
        magnitude (int | float): The magnitude used for shear.
        direction (str): The flip direction, either "horizontal"
            or "vertical".
        border_value (int | tuple[int]): Value used in case of a
            constant border.
        interpolation (str): Same as :func:`resize`.

    Returns:
        ndarray: The sheared image.
    """
    assert direction in ["horizontal", "vertical"], f"Invalid direction: {direction}"
    height, width = img.shape[:2]
    if img.ndim == 2:
        channels = 1
    elif img.ndim == 3:
        channels = img.shape[-1]
    else:
        raise ValueError(f"Invalid image dimensions: {img.ndim}")
    if isinstance(border_value, int):
        border_value = tuple([border_value] * channels)  # type: ignore
    elif isinstance(border_value, tuple):
        assert len(border_value) == channels, (
            f"Expected the num of elements in tuple equals the channels of input image. Found {len(border_value)} vs {channels}"
        )
    else:
        raise ValueError(f"Invalid type {type(border_value)} for `border_value`")
    shear_matrix = _get_shear_matrix(magnitude, direction)
    sheared = cv2.warpAffine(
        img,
        shear_matrix,
        (width, height),
        # Note case when the number elements in `border_value`
        # greater than 3 (e.g. shearing masks whose channels large
        # than 3) will raise TypeError in `cv2.warpAffine`.
        # Here simply slice the first 3 values in `border_value`.
        borderValue=border_value[:3],  # type: ignore
        flags=cv2_interp_codes[interpolation],
    )
    return sheared


def _get_translate_matrix(offset: int | float, direction: str = "horizontal") -> np.ndarray:
    """Generate the translate matrix.

    Args:
        offset (int | float): The offset used for translate.
        direction (str): The translate direction, either
            "horizontal" or "vertical".

    Returns:
        ndarray: The translate matrix with dtype float32.
    """
    if direction == "horizontal":
        translate_matrix = np.float32([[1, 0, offset], [0, 1, 0]])
    elif direction == "vertical":
        translate_matrix = np.float32([[1, 0, 0], [0, 1, offset]])
    return translate_matrix


def imtranslate(
    img: np.ndarray,
    offset: int | float,
    direction: str = "horizontal",
    border_value: int | tuple = 0,
    interpolation: str = "bilinear",
) -> np.ndarray:
    """Translate an image.

    Args:
        img (ndarray): Image to be translated with format
            (h, w) or (h, w, c).
        offset (int | float): The offset used for translate.
        direction (str): The translate direction, either "horizontal"
            or "vertical".
        border_value (int | tuple[int]): Value used in case of a
            constant border.
        interpolation (str): Same as :func:`resize`.

    Returns:
        ndarray: The translated image.
    """
    assert direction in ["horizontal", "vertical"], f"Invalid direction: {direction}"
    height, width = img.shape[:2]
    if img.ndim == 2:
        channels = 1
    elif img.ndim == 3:
        channels = img.shape[-1]
    else:
        raise ValueError(f"Invalid image dimensions: {img.ndim}")
    if isinstance(border_value, int):
        border_value = tuple([border_value] * channels)
    elif isinstance(border_value, tuple):
        assert len(border_value) == channels, (
            f"Expected the num of elements in tuple equals the channels of input image. Found {len(border_value)} vs {channels}"
        )
    else:
        raise ValueError(f"Invalid type {type(border_value)} for `border_value`.")
    translate_matrix = _get_translate_matrix(offset, direction)
    translated = cv2.warpAffine(
        img,
        translate_matrix,
        (width, height),
        # Note case when the number elements in `border_value`
        # greater than 3 (e.g. translating masks whose channels
        # large than 3) will raise TypeError in `cv2.warpAffine`.
        # Here simply slice the first 3 values in `border_value`.
        borderValue=border_value[:3],
        flags=cv2_interp_codes[interpolation],
    )
    return translated


def bbox_clip(bboxes: np.ndarray, img_shape: tuple[int, int]) -> np.ndarray:
    """Clip bboxes to fit the image shape.

    Args:
        bboxes (ndarray): Shape (..., 4*k) in format (x1, y1, x2, y2, ...)
        img_shape (tuple[int]): (height, width) of the image.

    Returns:
        ndarray: Clipped bboxes.
    """
    assert bboxes.shape[-1] % 4 == 0
    clipped_bboxes = bboxes.copy()

    h, w = img_shape

    # Process each group of 4 coordinates (x1, y1, x2, y2)
    for i in range(0, bboxes.shape[-1], 4):
        # Clip x coordinates
        clipped_bboxes[..., i] = np.clip(clipped_bboxes[..., i], 0, w)  # x1
        clipped_bboxes[..., i + 2] = np.clip(clipped_bboxes[..., i + 2], 0, w)  # x2

        # Clip y coordinates
        clipped_bboxes[..., i + 1] = np.clip(clipped_bboxes[..., i + 1], 0, h)  # y1
        clipped_bboxes[..., i + 3] = np.clip(clipped_bboxes[..., i + 3], 0, h)  # y2

    return clipped_bboxes


def bbox_scaling(bboxes: np.ndarray, scale: float, clip_shape=None) -> np.ndarray:
    """Scaling bboxes w.r.t the box center.

    Args:
        bboxes (ndarray): Shape(..., 4).
        scale (float): Scaling factor.
        clip_shape (tuple[int], optional): If specified, bboxes that exceed the
            boundary will be clipped according to the given shape (h, w).

    Returns:
        ndarray: Scaled bboxes.
    """
    if float(scale) == 1.0:
        scaled_bboxes = bboxes.copy()
    else:
        w = bboxes[..., 2] - bboxes[..., 0] + 1
        h = bboxes[..., 3] - bboxes[..., 1] + 1
        dw = (w * (scale - 1)) * 0.5
        dh = (h * (scale - 1)) * 0.5
        scaled_bboxes = bboxes + np.stack((-dw, -dh, dw, dh), axis=-1)
    if clip_shape is not None:
        return bbox_clip(scaled_bboxes, clip_shape)
    else:
        return scaled_bboxes


def imcrop(
    img: np.ndarray,
    bboxes: np.ndarray,
    scale: float = 1.0,
    pad_fill: float | list | None = None,
) -> np.ndarray | list[np.ndarray]:
    """Crop image patches.

    3 steps: scale the bboxes -> clip bboxes -> crop and pad.

    Args:
        img (ndarray): Image to be cropped.
        bboxes (ndarray): Shape (k, 4) or (4, ), location of cropped bboxes.
        scale (float, optional): Scale ratio of bboxes, the default value
            1.0 means no scaling.
        pad_fill (Number | list[Number]): Value to be filled for padding.
            Default: None, which means no padding.

    Returns:
        list[ndarray] | ndarray: The cropped image patches.
    """
    chn = 1 if img.ndim == 2 else img.shape[2]
    if pad_fill is not None:
        if isinstance(pad_fill, (int, float)):
            pad_fill = [pad_fill for _ in range(chn)]
        assert len(pad_fill) == chn

    _bboxes = bboxes[None, ...] if bboxes.ndim == 1 else bboxes
    scaled_bboxes = bbox_scaling(_bboxes, scale).astype(np.int32)
    clipped_bbox = bbox_clip(scaled_bboxes, img.shape)

    patches = []
    for i in range(clipped_bbox.shape[0]):
        x1, y1, x2, y2 = tuple(clipped_bbox[i, :])
        if pad_fill is None:
            patch = img[y1 : y2 + 1, x1 : x2 + 1, ...]
        else:
            _x1, _y1, _x2, _y2 = tuple(scaled_bboxes[i, :])
            patch_h = _y2 - _y1 + 1
            patch_w = _x2 - _x1 + 1
            if chn == 1:
                patch_shape = (patch_h, patch_w)
            else:
                patch_shape = (patch_h, patch_w, chn)  # type: ignore
            patch = np.array(pad_fill, dtype=img.dtype) * np.ones(patch_shape, dtype=img.dtype)
            x_start = 0 if _x1 >= 0 else -_x1
            y_start = 0 if _y1 >= 0 else -_y1
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            patch[y_start : y_start + h, x_start : x_start + w, ...] = img[y1 : y1 + h, x1 : x1 + w, ...]
        patches.append(patch)

    if bboxes.ndim == 1:
        return patches[0]
    else:
        return patches
