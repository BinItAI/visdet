# ruff: noqa
"""
Utility functions for visdet.

Provides common utility functions for training and inference.
"""

import os
import os.path as osp
from pathlib import Path
from typing import Any, Iterable, Sequence, Union


def digit_version(version_str: str) -> tuple:
    """Convert version string to a tuple of digits.

    Args:
        version_str: Version string like "1.2.3"

    Returns:
        Tuple of integers representing the version
    """
    try:
        return tuple(int(d) for d in version_str.split("."))
    except (ValueError, AttributeError):
        return (0,)


def to_2tuple(x: Union[int, float, Sequence]) -> tuple:
    """Convert input to a 2-tuple."""
    if isinstance(x, (tuple, list)):
        return tuple(x) if len(x) == 2 else (x[0], x[0])
    return (x, x)


def is_str(x: Any) -> bool:
    """Whether the input is a string instance."""
    return isinstance(x, str)


def is_seq_of(seq: Any, expected_type: type, seq_type: type = None) -> bool:
    """Check whether it is a sequence of some type.

    Args:
        seq: The sequence to be checked
        expected_type: Expected type of sequence items
        seq_type: Expected sequence type, defaults to (tuple, list)

    Returns:
        True if all items have expected_type
    """
    if seq_type is None:
        seq_type = (tuple, list)
    if not isinstance(seq, seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_tuple_of(seq: Any, expected_type: type) -> bool:
    """Check whether it is a tuple of some type.

    Args:
        seq: The sequence to be checked
        expected_type: Expected type of sequence items

    Returns:
        True if it is a tuple and all items have expected_type
    """
    return is_seq_of(seq, expected_type, seq_type=tuple)


def scandir(path: str) -> Iterable[str]:
    """Scan a directory to find the interested files.

    Args:
        path: Path to the directory

    Yields:
        Path to each file in the directory
    """
    for entry in os.scandir(path):
        yield entry.path


def slice_list(in_list: list, lens: Iterable) -> list:
    """Slice a list into several sub lists by the given length.

    Args:
        in_list: The list to slice
        lens: The slice length

    Returns:
        List of sliced lists
    """
    if not isinstance(lens, Iterable):
        raise TypeError(f"lens must be an iterable, but got {type(lens)}")

    if isinstance(lens, int):
        raise TypeError(f"lens must be an iterable, but got {type(lens)}")

    lens = list(lens)
    if sum(lens) != len(in_list):
        raise ValueError("sum(lens) and the length of in_list do not match")

    result = []
    idx = 0
    for length in lens:
        result.append(in_list[idx : idx + length])
        idx += length
    return result


def mkdir_or_exist(dir_name: str) -> None:
    """Make a directory or check if it exists.

    Args:
        dir_name: Directory name
    """
    if not osp.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def is_abs(path: str) -> bool:
    """Check whether the path is absolute.

    Args:
        path: Path string

    Returns:
        True if path is absolute
    """
    return osp.isabs(path) or Path(path).is_absolute()


__all__ = [
    "digit_version",
    "to_2tuple",
    "is_str",
    "is_seq_of",
    "is_tuple_of",
    "scandir",
    "slice_list",
    "mkdir_or_exist",
    "is_abs",
]
