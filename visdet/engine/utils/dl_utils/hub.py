# ruff: noqa
# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any

from ..path import mkdir_or_exist
from torch.hub import load_state_dict_from_url as _load_state_dict_from_url

__all__ = ["mkdir_or_exist", "load_url"]


def load_url(
    url: str,
    model_dir: str | None = None,
    map_location: Any | None = None,
    progress: bool = True,
    check_hash: bool = False,
    file_name: str | None = None,
    **kwargs: Any,
):
    """Compat shim that delegates to ``torch.hub.load_state_dict_from_url``.

    Accepts the legacy ``torch.utils.model_zoo.load_url`` signature so existing
    call sites remain unchanged while benefiting from the maintained API.
    """

    return _load_state_dict_from_url(
        url,
        model_dir=model_dir,
        map_location=map_location,
        progress=progress,
        check_hash=check_hash,
        file_name=file_name,
        **kwargs,
    )
