# ruff: noqa
# Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from visdet.engine.utils import is_str

ColorTuple = tuple[int, int, int]


def _as_color_tuple(color: Any) -> ColorTuple | None:
    if isinstance(color, np.ndarray):
        color = color.tolist()
    if isinstance(color, Sequence):
        ints = [int(c) for c in color[:3]]
        if not ints:
            return None
        while len(ints) < 3:
            ints.append(0)
        return tuple(ints)
    if isinstance(color, tuple):
        return tuple(int(c) for c in color[:3])
    return None


def palette_val(palette: Sequence[Sequence[int]] | Sequence[int]) -> list[ColorTuple]:
    """Convert palette to matplotlib palette.

    Args:
        palette (List[tuple]): A list of color tuples.

    Returns:
        List[tuple[float]]: A list of RGB matplotlib color tuples.
    """
    new_palette = []
    for color in palette:
        color_tuple = _as_color_tuple(color)
        if color_tuple is None:
            continue
        color = [c / 255 for c in color_tuple]
        new_palette.append(tuple(color))
    return new_palette


def get_palette(
    palette: Sequence[Sequence[int]] | Sequence[int] | np.ndarray | str | ColorTuple | None,
    num_classes: int,
) -> list[ColorTuple]:
    """Get palette from various inputs.

    Args:
        palette (list[tuple] | str | tuple | None): palette inputs.
        num_classes (int): the number of classes.

    Returns:
        list[tuple[int, ...]]: A list of color tuples.
    """
    assert isinstance(num_classes, int)

    dataset_palette: list[ColorTuple] | None = None
    if isinstance(palette, list):
        colors: list[ColorTuple] = []
        for color in palette:
            color_tuple = _as_color_tuple(color)
            if color_tuple is not None:
                colors.append(color_tuple)
        if colors:
            dataset_palette = colors
    elif isinstance(palette, np.ndarray):
        if palette.ndim == 1:
            color_tuple = _as_color_tuple(palette)
            if color_tuple is not None:
                dataset_palette = [color_tuple] * num_classes
        else:
            colors = []
            for row in palette:
                color_tuple = _as_color_tuple(row)
                if color_tuple is not None:
                    colors.append(color_tuple)
            if colors:
                dataset_palette = colors
    else:
        color_tuple = _as_color_tuple(palette)
        if color_tuple is not None:
            dataset_palette = [color_tuple] * num_classes
    if dataset_palette is None and (palette == "random" or palette is None):
        state = np.random.get_state()
        # random color
        np.random.seed(42)
        palette = np.random.randint(0, 256, size=(num_classes, 3))
        np.random.set_state(state)
        dataset_palette = [tuple(c) for c in palette]
    elif dataset_palette is None and palette == "coco":
        # For now, we'll use a predefined COCO palette
        # This avoids circular imports from datasets
        coco_palette = [
            (220, 20, 60),
            (119, 11, 32),
            (0, 0, 142),
            (0, 0, 230),
            (106, 0, 228),
            (0, 60, 100),
            (0, 80, 100),
            (0, 0, 70),
            (0, 0, 192),
            (250, 170, 30),
            (100, 170, 30),
            (220, 220, 0),
            (175, 116, 175),
            (250, 0, 30),
            (165, 42, 42),
            (255, 77, 255),
            (0, 226, 252),
            (182, 182, 255),
            (0, 82, 0),
            (120, 166, 157),
            (110, 76, 0),
            (174, 57, 255),
            (199, 100, 0),
            (72, 0, 118),
            (255, 179, 240),
            (0, 125, 92),
            (209, 0, 151),
            (188, 208, 182),
            (0, 220, 176),
            (255, 99, 164),
            (92, 0, 73),
            (133, 129, 255),
            (78, 180, 255),
            (0, 228, 0),
            (174, 255, 243),
            (45, 89, 255),
            (134, 134, 103),
            (145, 148, 174),
            (255, 208, 186),
            (197, 226, 255),
            (171, 134, 1),
            (109, 63, 54),
            (207, 138, 255),
            (151, 0, 95),
            (9, 80, 61),
            (84, 105, 51),
            (74, 65, 105),
            (166, 196, 102),
            (208, 195, 210),
            (255, 109, 65),
            (0, 143, 149),
            (179, 0, 194),
            (209, 99, 106),
            (5, 121, 0),
            (227, 255, 205),
            (147, 186, 208),
            (153, 69, 1),
            (3, 95, 161),
            (163, 255, 0),
            (119, 0, 170),
            (0, 182, 199),
            (0, 165, 120),
            (183, 130, 88),
            (95, 32, 0),
            (130, 114, 135),
            (110, 129, 133),
            (166, 74, 118),
            (219, 142, 185),
            (79, 210, 114),
            (178, 90, 62),
            (65, 70, 15),
            (127, 167, 115),
            (59, 105, 106),
            (142, 108, 45),
            (196, 172, 0),
            (95, 54, 80),
            (128, 76, 255),
            (201, 57, 1),
            (246, 0, 122),
            (191, 162, 208),
        ]
        dataset_palette = coco_palette[:num_classes]
        if len(dataset_palette) < num_classes:
            # Generate additional colors if needed
            np.random.seed(42)
            extra_colors = np.random.randint(0, 256, size=(num_classes - len(dataset_palette), 3))
            dataset_palette.extend([tuple(c) for c in extra_colors])
    elif dataset_palette is None and palette == "citys":
        # Cityscapes palette - simplified version
        citys_palette = [
            (128, 64, 128),
            (244, 35, 232),
            (70, 70, 70),
            (102, 102, 156),
            (190, 153, 153),
            (153, 153, 153),
            (250, 170, 30),
            (220, 220, 0),
            (107, 142, 35),
            (152, 251, 152),
            (70, 130, 180),
            (220, 20, 60),
            (255, 0, 0),
            (0, 0, 142),
            (0, 0, 70),
            (0, 60, 100),
            (0, 80, 100),
            (0, 0, 230),
            (119, 11, 32),
        ]
        dataset_palette = citys_palette[:num_classes]
        if len(dataset_palette) < num_classes:
            np.random.seed(42)
            extra_colors = np.random.randint(0, 256, size=(num_classes - len(dataset_palette), 3))
            dataset_palette.extend([tuple(c) for c in extra_colors])
    elif dataset_palette is None and palette == "voc":
        # VOC palette
        voc_palette = [
            (0, 0, 0),
            (128, 0, 0),
            (0, 128, 0),
            (128, 128, 0),
            (0, 0, 128),
            (128, 0, 128),
            (0, 128, 128),
            (128, 128, 128),
            (64, 0, 0),
            (192, 0, 0),
            (64, 128, 0),
            (192, 128, 0),
            (64, 0, 128),
            (192, 0, 128),
            (64, 128, 128),
            (192, 128, 128),
            (0, 64, 0),
            (128, 64, 0),
            (0, 192, 0),
            (128, 192, 0),
            (0, 64, 128),
        ]
        dataset_palette = voc_palette[:num_classes]
        if len(dataset_palette) < num_classes:
            np.random.seed(42)
            extra_colors = np.random.randint(0, 256, size=(num_classes - len(dataset_palette), 3))
            dataset_palette.extend([tuple(c) for c in extra_colors])
    elif dataset_palette is None and is_str(palette):
        # Convert color string to RGB tuple
        # Simple color name to RGB mapping
        color_map = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
        }
        rgb = color_map.get(palette.lower(), (128, 128, 128))
        dataset_palette = [rgb] * num_classes
    if dataset_palette is None:
        raise TypeError(f"Invalid type for palette: {type(palette)}")

    if len(dataset_palette) < num_classes:
        last_color = dataset_palette[-1] if dataset_palette else (0, 0, 0)
        dataset_palette = dataset_palette + [last_color] * (num_classes - len(dataset_palette))

    return dataset_palette[:num_classes]


def _get_adaptive_scales(areas: np.ndarray | float, min_area: int = 800, max_area: int = 30000) -> np.ndarray:
    """Get adaptive scales according to areas.

    The scale range is [0.5, 1.0]. When the area is less than
    ``min_area``, the scale is 0.5 while the area is larger than
    ``max_area``, the scale is 1.0.

    Args:
        areas (ndarray | float): The areas of bboxes or masks with the
            shape of (n, ) or a single float value.
        min_area (int): Lower bound areas for adaptive scales.
            Defaults to 800.
        max_area (int): Upper bound areas for adaptive scales.
            Defaults to 30000.

    Returns:
        ndarray: The adaotive scales with the shape of (n, ) or (1,).
    """
    if isinstance(areas, np.ndarray):
        scales = 0.5 + (areas - min_area) // (max_area - min_area)
        scales = np.clip(scales, 0.5, 1.0)
        return scales
    else:
        # Handle scalar case - convert to array
        scale = 0.5 + (areas - min_area) / (max_area - min_area)
        return np.array([np.clip(scale, 0.5, 1.0)])


def jitter_color(color: tuple) -> tuple:
    """Randomly jitter the given color in order to better distinguish instances
    with the same class.

    Args:
        color (tuple): The RGB color tuple. Each value is between [0, 255].

    Returns:
        tuple: The jittered color tuple.
    """
    jitter = np.random.rand(3)
    jitter = (jitter / np.linalg.norm(jitter) - 0.5) * 0.5 * 255
    color = np.clip(jitter + color, 0, 255).astype(np.uint8)
    return tuple(color)
