# Copyright (c) OpenMMLab. All rights reserved.
"""Simple file I/O utilities for viscv."""

from pathlib import Path


def get(filepath: str | Path, backend_args: dict | None = None) -> bytes:
    """Read bytes from a given filepath.

    Args:
        filepath: Path to read from.
        backend_args: Backend-specific arguments (currently unused).

    Returns:
        bytes: File contents as bytes.
    """
    filepath = Path(filepath)
    with open(filepath, "rb") as f:
        return f.read()


class FileClient:
    """Simple file client for reading files."""

    def __init__(self, backend="disk", **kwargs):
        self.backend = backend

    @classmethod
    def infer_client(cls, file_client_args, filename):
        """Infer the file client from arguments."""
        return cls(**file_client_args)

    def get(self, filepath: str | Path) -> bytes:
        """Read bytes from a given filepath."""
        return get(filepath)
