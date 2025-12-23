"""Download COCO 2017 (train/val + annotations) into a Modal Volume.

Usage:
  VISDET_COCO_VOLUME=visdet-coco modal run tools/modal/download_coco2017_to_volume.py

This populates the volume so training jobs can mount it at `/root/data` and use the
existing `data/coco/` paths in configs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import Literal, TypedDict

import modal

DEFAULT_VOLUME_NAME = os.environ.get("VISDET_COCO_VOLUME", "visdet-coco")
DATA_MOUNT_PATH = "/root/data"
COCO_DIRNAME = "coco"

COCO_URLS: dict[str, str] = {
    "train": "https://images.cocodataset.org/zips/train2017.zip",
    "val": "https://images.cocodataset.org/zips/val2017.zip",
    "annotations": "https://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

image = modal.Image.debian_slim(python_version="3.12").apt_install("wget", "unzip")
app = modal.App("visdet-coco-download", image=image)
volume = modal.Volume.from_name(DEFAULT_VOLUME_NAME, create_if_missing=True)


class SplitResult(TypedDict):
    split: Literal["train", "val", "annotations"]
    status: Literal["skipped", "downloaded"]
    seconds: float


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=None if cwd is None else str(cwd), check=True)


def _is_nonempty_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        next(path.iterdir())
    except StopIteration:
        return False
    return True


def _download_zip(url: str, *, download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    filename = url.rsplit("/", 1)[-1]
    zip_path = download_dir / filename

    # Resume partial downloads with `-c`.
    _run(["wget", "-c", "--no-check-certificate", "--progress=dot:giga", url], cwd=download_dir)
    if not zip_path.exists():
        raise FileNotFoundError(f"Expected {zip_path} to exist after download")
    return zip_path


def _unzip(zip_path: Path, *, extract_dir: Path, overwrite: bool) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    flags = ["-q"]
    flags.append("-o" if overwrite else "-n")
    _run(["unzip", *flags, str(zip_path), "-d", str(extract_dir)])


@app.function(
    timeout=60 * 60 * 8,  # 8 hours
    volumes={DATA_MOUNT_PATH: volume},
)
def download_coco2017(
    *,
    include_train: bool = True,
    include_val: bool = True,
    include_annotations: bool = True,
    delete_zips: bool = True,
    force: bool = False,
) -> list[SplitResult]:
    """Populate a Modal volume with COCO 2017 under `/root/data/coco/`."""

    coco_root = Path(DATA_MOUNT_PATH) / COCO_DIRNAME
    coco_root.mkdir(parents=True, exist_ok=True)

    wanted: list[Literal["train", "val", "annotations"]] = []
    if include_train:
        wanted.append("train")
    if include_val:
        wanted.append("val")
    if include_annotations:
        wanted.append("annotations")

    annotation_markers = [
        coco_root / "annotations" / "instances_train2017.json",
        coco_root / "annotations" / "instances_val2017.json",
    ]
    markers: dict[str, Path] = {
        "train": coco_root / "train2017",
        "val": coco_root / "val2017",
        "annotations": annotation_markers[0],
    }

    results: list[SplitResult] = []
    for split in wanted:
        marker = markers[split]
        already_present = False
        if split in {"train", "val"}:
            already_present = marker.is_dir() and _is_nonempty_dir(marker)
        elif split == "annotations":
            already_present = all(p.is_file() for p in annotation_markers)

        if not force and already_present:
            results.append({"split": split, "status": "skipped", "seconds": 0.0})
            continue

        url = COCO_URLS[split]
        start = time.time()
        zip_path = _download_zip(url, download_dir=coco_root)
        _unzip(zip_path, extract_dir=coco_root, overwrite=force)
        if delete_zips:
            zip_path.unlink(missing_ok=True)

        if split in {"train", "val"} and not _is_nonempty_dir(markers[split]):
            raise RuntimeError(f"Expected extracted directory to exist and be non-empty: {markers[split]}")
        if split == "annotations" and not all(p.is_file() for p in annotation_markers):
            raise RuntimeError(f"Expected annotation files to exist after unzip: {annotation_markers}")

        results.append({"split": split, "status": "downloaded", "seconds": time.time() - start})

    # Make sure writes are durable for subsequent runs.
    volume.commit()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download COCO 2017 into a Modal volume")
    parser.add_argument(
        "--volume",
        default=DEFAULT_VOLUME_NAME,
        help=(f"Modal volume name (must match VISDET_COCO_VOLUME at import time). Default: {DEFAULT_VOLUME_NAME!r}"),
    )
    parser.add_argument("--train", dest="include_train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val", dest="include_val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--annotations",
        dest="include_annotations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--delete-zips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="Re-download/re-extract even if files already exist")
    args = parser.parse_args()

    if args.volume != DEFAULT_VOLUME_NAME:
        raise SystemExit(
            "Volume name is fixed at import time. Re-run with env var, e.g.:\n\n"
            f"  VISDET_COCO_VOLUME={args.volume} modal run tools/modal/download_coco2017_to_volume.py\n"
        )

    out = download_coco2017.remote(
        include_train=args.include_train,
        include_val=args.include_val,
        include_annotations=args.include_annotations,
        delete_zips=args.delete_zips,
        force=args.force,
    )
    for item in out:
        print(item)
