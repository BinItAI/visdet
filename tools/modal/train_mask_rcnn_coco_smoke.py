"""Run a tiny Mask R-CNN training smoke test on COCO inside Modal.

This is intended as an integration test / infrastructure smoke test, not a
benchmark. It:
  - mounts a persistent Modal Volume containing COCO 2017 under `/root/data`
  - generates a 1-image COCO annotation file (val split) for fast iteration
  - runs 1 epoch of `mask_rcnn_swin_s` with small images and no checkpointing

Prereqs:
  1) `modal` installed + authenticated (`python -m modal token new` etc.)
  2) COCO volume populated:
       VISDET_COCO_VOLUME=visdet-coco python -m modal run tools/modal/download_coco2017_to_volume.py

Run:
  VISDET_COCO_VOLUME=visdet-coco python -m modal run tools/modal/train_mask_rcnn_coco_smoke.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import modal

DEFAULT_VOLUME_NAME = os.environ.get("VISDET_COCO_VOLUME", "visdet-coco")
DATA_MOUNT_PATH = "/root/data"
COCO_DIRNAME = "coco"
REMOTE_REPO_PATH = "/root/visdet"


def _ignore_repo_path(path: Path) -> bool:
    parts = path.parts
    if ".git" in parts:
        return True
    if ".venv" in parts:
        return True
    if "__pycache__" in parts:
        return True
    if "work_dirs" in parts:
        return True
    if "data" in parts:
        return True
    if "archive" in parts:
        return True
    return False


def _find_repo_root() -> Path:
    """Best-effort repo root resolution.

    `modal run` copies this file into the container as `/root/<name>.py`, so
    `Path(__file__).parents[2]` is not stable. Prefer a marker-based search.
    """

    this_file = Path(__file__).resolve()
    for parent in [this_file.parent, *this_file.parents]:
        if (parent / "pyproject.toml").is_file() and (parent / "visdet").is_dir() and (parent / "configs").is_dir():
            return parent

    # Inside the Modal image, the repo is baked into `/root/visdet`.
    modal_repo = Path(REMOTE_REPO_PATH)
    if (modal_repo / "visdet").is_dir() and (modal_repo / "configs").is_dir():
        return modal_repo

    # Fall back to something safe (may disable add_local_dir on CI misconfigs).
    return this_file.parent


REPO_ROOT = _find_repo_root()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        # opencv-python runtime deps
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install(
        # Minimal runtime deps needed for a 1-iter training run.
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "pycocotools",
        "pydantic>=2.0.0",
        "pyyaml",
        "rich",
        "scipy",
        "shapely",
        "addict",
        "six",
        "termcolor",
        "terminaltables",
        "tqdm",
        "yapf",
    )
    .add_local_dir(REPO_ROOT / "visdet", remote_path=f"{REMOTE_REPO_PATH}/visdet", copy=True, ignore=_ignore_repo_path)
    .add_local_dir(
        REPO_ROOT / "configs", remote_path=f"{REMOTE_REPO_PATH}/configs", copy=True, ignore=_ignore_repo_path
    )
)

app = modal.App("visdet-train-maskrcnn-coco-smoke", image=image)
volume = modal.Volume.from_name(DEFAULT_VOLUME_NAME, create_if_missing=False)


def _make_one_image_coco_subset(*, source_ann: Path, coco_images_dir: Path, out_ann: Path) -> None:
    data = json.loads(source_ann.read_text())
    annotations = data.get("annotations", [])
    images = data.get("images", [])
    categories = data.get("categories", [])

    ann_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in annotations:
        image_id = ann.get("image_id")
        if not isinstance(image_id, int):
            continue
        ann_by_image.setdefault(image_id, []).append(ann)

    chosen_image: dict[str, Any] | None = None
    for img in images:
        image_id = img.get("id")
        file_name = img.get("file_name")
        if not isinstance(image_id, int) or not isinstance(file_name, str):
            continue
        if image_id not in ann_by_image:
            continue
        if not (coco_images_dir / file_name).is_file():
            continue
        chosen_image = img
        break

    if chosen_image is None:
        raise RuntimeError(f"Could not find a COCO image with annotations in {source_ann}")

    chosen_id = chosen_image["id"]
    subset = {
        "images": [chosen_image],
        "annotations": [ann for ann in annotations if ann.get("image_id") == chosen_id],
        "categories": categories,
    }
    out_ann.parent.mkdir(parents=True, exist_ok=True)
    out_ann.write_text(json.dumps(subset))


@app.function(
    timeout=60 * 60,  # 1 hour
    cpu=4,
    memory=16_000,
    volumes={DATA_MOUNT_PATH: volume},
)
def train_mask_rcnn_coco_smoke(*, force_regen_ann: bool = False) -> dict[str, Any]:
    import sys

    sys.path.insert(0, REMOTE_REPO_PATH)

    coco_root = Path(DATA_MOUNT_PATH) / COCO_DIRNAME
    source_ann = coco_root / "annotations" / "instances_val2017.json"
    coco_images_dir = coco_root / "val2017"

    if not source_ann.is_file():
        raise RuntimeError(
            "COCO annotations not found in the Modal volume. Populate it first:\n"
            f"  VISDET_COCO_VOLUME={DEFAULT_VOLUME_NAME} python -m modal run tools/modal/download_coco2017_to_volume.py\n"
        )
    if not coco_images_dir.is_dir():
        raise RuntimeError(f"COCO image dir missing: {coco_images_dir}")

    smoke_ann = coco_root / "annotations" / "instances_val2017_smoke_1img.json"
    if force_regen_ann or not smoke_ann.is_file():
        _make_one_image_coco_subset(source_ann=source_ann, coco_images_dir=coco_images_dir, out_ann=smoke_ann)

    # Run a tiny training loop.
    from visdet import SimpleRunner

    dataset_override: dict[str, Any] = {
        "_base_": "coco_instance_segmentation",
        # Use absolute paths to avoid depending on the CWD.
        "data_root": f"{coco_root}/",
        "ann_file": "annotations/instances_val2017_smoke_1img.json",
        "data_prefix": {"img_path": "val2017/"},
        # Smaller images for faster CPU iteration.
        "train_pipeline": [
            {"type": "LoadImageFromFile"},
            {"type": "LoadAnnotations", "with_bbox": True, "with_mask": True},
            {"type": "Resize", "scale": [320, 240], "keep_ratio": True},
            {"type": "RandomFlip", "prob": 0.0},
            {"type": "PackDetInputs"},
        ],
    }

    work_dir = "/tmp/work_dirs/mask_rcnn_coco_smoke"
    start = time.time()
    runner = SimpleRunner(
        model="mask_rcnn_swin_s",
        dataset=dataset_override,
        epochs=1,
        val_interval=999999,  # safety, though validation is disabled below
        batch_size=1,
        num_workers=1,  # SimpleRunner currently hard-codes persistent_workers=True
        work_dir=work_dir,
        # Disable validation entirely to keep this as a pure training smoke test.
        val_cfg=None,
        val_dataloader=None,
        val_evaluator=None,
        # Avoid writing huge checkpoints.
        default_hooks={
            "timer": {"type": "IterTimerHook"},
            "logger": {"type": "LoggerHook", "interval": 1},
            "param_scheduler": {"type": "ParamSchedulerHook"},
            "checkpoint": {"type": "CheckpointHook", "interval": 999999999},
            "sampler_seed": {"type": "DistSamplerSeedHook"},
        },
    )
    runner.train()
    return {"status": "ok", "seconds": time.time() - start, "work_dir": work_dir, "ann_file": str(smoke_ann)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a tiny Mask R-CNN COCO training smoke test on Modal")
    parser.add_argument(
        "--volume",
        default=DEFAULT_VOLUME_NAME,
        help=(f"Modal volume name (must match VISDET_COCO_VOLUME at import time). Default: {DEFAULT_VOLUME_NAME!r}"),
    )
    parser.add_argument("--force-regen-ann", action="store_true", help="Regenerate the 1-image COCO annotation file")
    args = parser.parse_args()

    if args.volume != DEFAULT_VOLUME_NAME:
        raise SystemExit(
            "Volume name is fixed at import time. Re-run with env var, e.g.:\n\n"
            f"  VISDET_COCO_VOLUME={args.volume} python -m modal run tools/modal/train_mask_rcnn_coco_smoke.py\n"
        )

    out = train_mask_rcnn_coco_smoke.remote(force_regen_ann=args.force_regen_ann)
    print(out)
