from __future__ import annotations

import importlib
import importlib.util


_DEMO_EXT_SPEC = importlib.util.find_spec("visdet._ext_demo._demo_ext")
HAS_EXT = _DEMO_EXT_SPEC is not None

if HAS_EXT:
    _demo_ext = importlib.import_module("visdet._ext_demo._demo_ext")
else:
    _demo_ext = None


def add(a: int, b: int) -> int:
    """Add two integers.

    Uses the compiled extension when available, otherwise falls back to Python.
    """

    if _demo_ext is None:
        return int(a) + int(b)

    return int(_demo_ext.add(int(a), int(b)))
