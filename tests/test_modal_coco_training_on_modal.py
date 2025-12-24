import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest


_RUN_MODAL_INTEGRATION_TESTS = os.environ.get("VISDET_RUN_MODAL_INTEGRATION_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not _RUN_MODAL_INTEGRATION_TESTS,
    reason="Set VISDET_RUN_MODAL_INTEGRATION_TESTS=1 to run Modal integration tests (may incur cloud costs).",
)

if _RUN_MODAL_INTEGRATION_TESTS and find_spec("modal") is None:
    pytest.fail(
        "Modal integration tests requested but `modal` is not installed. "
        "Install it (e.g. `uv sync --group dev` or `uv pip install modal`) and retry.",
        pytrace=False,
    )


def test_mask_rcnn_coco_smoke_on_modal() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.setdefault("VISDET_COCO_VOLUME", "visdet-coco")

    required_tokens = ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"]
    missing_tokens = [name for name in required_tokens if not env.get(name)]
    if missing_tokens:
        pytest.fail(
            "Modal integration tests require Modal auth via env vars. "
            f"Missing: {', '.join(missing_tokens)}. "
            "Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET and retry.",
            pytrace=False,
        )

    subprocess.run(
        [sys.executable, "-m", "modal", "run", "tools/modal/train_mask_rcnn_coco_smoke.py"],
        cwd=repo_root,
        env=env,
        check=True,
        timeout=60 * 60,
    )
