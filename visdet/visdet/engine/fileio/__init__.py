# ruff: noqa
"""
File I/O utilities for visdet.

This module provides functions for reading and writing files.
"""

from typing import Optional
from pathlib import Path
import os


def get(filepath: str, backend_args: Optional[dict] = None) -> bytes:
    """Get file content.

    Args:
        filepath: Path to the file
        backend_args: Backend arguments (unused in this stub)

    Returns:
        File content as bytes
    """
    with open(filepath, "rb") as f:
        return f.read()


def get_local_path(filepath: str, backend_args: Optional[dict] = None) -> str:
    """Get local path of a file.

    For local files, returns the file path as-is.
    For remote files, would download and return local path.

    Args:
        filepath: Path to the file (local or remote)
        backend_args: Backend arguments (unused in this stub)

    Returns:
        Local file path
    """
    # In a full implementation, this would handle remote file downloads
    # For now, just return the path as-is
    return filepath


def load(filepath: str, backend_args: Optional[dict] = None) -> any:
    """Load data from a file.

    Args:
        filepath: Path to the file
        backend_args: Backend arguments (unused in this stub)

    Returns:
        Loaded data (could be any type based on file format)
    """
    # This is a stub that would handle JSON, pickle, yaml, etc.
    # For now, just read the file content
    import json

    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Fall back to reading as bytes
        with open(filepath, "rb") as f:
            return f.read()


__all__ = ["get", "get_local_path", "load"]
