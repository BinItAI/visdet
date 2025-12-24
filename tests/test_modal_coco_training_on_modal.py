import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

if find_spec("modal") is None:
    pytest.skip(
        "`modal` is not installed. Install it (e.g. `uv sync --group dev`) to run Modal integration tests.",
        allow_module_level=True,
    )


def test_mask_rcnn_coco_smoke_on_modal() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.setdefault("VISDET_COCO_VOLUME", "visdet-coco")

    required_tokens = ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"]
    missing_tokens = [name for name in required_tokens if not env.get(name)]
    if missing_tokens:
        pytest.skip(
            f"Modal integration tests require Modal auth via env vars. Missing: {', '.join(missing_tokens)}.",
        )

    subprocess.run(
        [sys.executable, "-m", "modal", "run", "tools/modal/train_mask_rcnn_coco_smoke.py"],
        cwd=repo_root,
        env=env,
        check=True,
        timeout=60 * 60,
    )
