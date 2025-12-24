import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest


if os.environ.get("VISDET_RUN_MODAL_INTEGRATION_TESTS") != "1":
    pytest.skip(
        "Set VISDET_RUN_MODAL_INTEGRATION_TESTS=1 to run Modal integration tests (may incur cloud costs).",
        allow_module_level=True,
    )

if find_spec("modal") is None:
    pytest.fail(
        "Modal integration tests requested but `modal` is not installed. "
        "Install it (e.g. `uv sync --group dev` or `uv pip install modal`) and retry.",
        pytrace=False,
    )


def test_mask_rcnn_coco_smoke_on_modal() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.setdefault("VISDET_COCO_VOLUME", "visdet-coco")

    subprocess.run(
        [sys.executable, "-m", "modal", "run", "tools/modal/train_mask_rcnn_coco_smoke.py"],
        cwd=repo_root,
        env=env,
        check=True,
        timeout=60 * 60,
    )
