"""Train Mask R-CNN on COCO using Modal GPU.

Usage:
  modal run tools/modal/train_mask_rcnn.py
  modal run tools/modal/train_mask_rcnn.py --epochs 12 --batch-size 4

Requires:
  - Modal volume 'visdet-coco' with COCO 2017 data (run download_coco2017_to_volume.py first)
  - Optional: Modal volume 'visdet-checkpoints' for persistent checkpoints
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Literal, Optional, TypedDict

import modal

# Configuration
DEFAULT_COCO_VOLUME = os.environ.get("VISDET_COCO_VOLUME", "visdet-coco")
DEFAULT_CHECKPOINT_VOLUME = os.environ.get("VISDET_CHECKPOINTS_VOLUME", "visdet-checkpoints")
DATA_MOUNT_PATH = "/root/data"
CHECKPOINT_MOUNT_PATH = "/root/checkpoints"

# Get the visdet package root directory (2 levels up from this file)
VISDET_ROOT = Path(__file__).parent.parent.parent

# Modal image with PyTorch and visdet dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        # PyTorch with CUDA support
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        # Core dependencies from pyproject.toml
        "addict",
        "matplotlib",
        "numpy",
        "opencv-python",
        "pycocotools",
        "pydantic>=2.0.0",
        "pyyaml",
        "rich",
        "scipy",
        "shapely",
        "six",
        "termcolor",
        "terminaltables",
        "tqdm>=4.67.1",
        "yapf",
    )
    # Add full visdet package including data files (.json, etc)
    .add_local_dir(str(VISDET_ROOT / "visdet"), remote_path="/root/visdet")
    .add_local_dir(str(VISDET_ROOT / "configs"), remote_path="/root/configs")
)

app = modal.App("visdet-train-mask-rcnn", image=image)
coco_volume = modal.Volume.from_name(DEFAULT_COCO_VOLUME)
checkpoint_volume = modal.Volume.from_name(DEFAULT_CHECKPOINT_VOLUME, create_if_missing=True)


class TrainingResult(TypedDict):
    status: Literal["success", "error"]
    epochs_completed: int
    final_metrics: dict
    work_dir: str
    duration_seconds: float
    error_message: Optional[str]


def _extract_metrics(work_dir: str) -> dict:
    """Extract final metrics from training logs."""
    log_dir = Path(work_dir)
    metrics = {}

    json_logs = list(log_dir.glob("*.log.json"))
    if json_logs:
        with open(sorted(json_logs)[-1]) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if "coco/bbox_mAP" in entry:
                        metrics["bbox_mAP"] = entry["coco/bbox_mAP"]
                    if "coco/segm_mAP" in entry:
                        metrics["segm_mAP"] = entry["coco/segm_mAP"]
                except json.JSONDecodeError:
                    continue

    return metrics


@app.function(
    gpu="A10G",
    timeout=60 * 60 * 12,  # 12 hours max
    volumes={
        DATA_MOUNT_PATH: coco_volume,
        CHECKPOINT_MOUNT_PATH: checkpoint_volume,
    },
)
def train_mask_rcnn(
    *,
    model: str = "mask_rcnn_r50",
    dataset: str = "coco_instance_segmentation",
    epochs: int = 1,
    batch_size: int = 2,
    num_workers: int = 4,
    val_interval: int = 1,
    save_checkpoints: bool = True,
) -> TrainingResult:
    """Train Mask R-CNN on COCO dataset using Modal GPU.

    Args:
        model: Model preset name (e.g., "mask_rcnn_r50", "mask_rcnn_swin_s")
        dataset: Dataset preset name (default: "coco_instance_segmentation")
        epochs: Number of training epochs
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        val_interval: Validation interval in epochs
        save_checkpoints: Whether to persist checkpoints to volume

    Returns:
        TrainingResult with status, metrics, and training info
    """
    import os

    start_time = time.time()

    # Set working directory to /root so data/coco paths resolve correctly
    os.chdir("/root")

    if save_checkpoints:
        work_dir = f"{CHECKPOINT_MOUNT_PATH}/{model}_{epochs}ep_{int(time.time())}"
    else:
        work_dir = "/tmp/work_dirs"

    try:
        from visdet import SimpleRunner

        print(f"Starting training: model={model}, dataset={dataset}, epochs={epochs}")
        print(f"Work directory: {work_dir}")
        print(f"COCO data path: {DATA_MOUNT_PATH}/coco")

        # Verify COCO data exists
        coco_path = Path(DATA_MOUNT_PATH) / "coco"
        if not coco_path.exists():
            raise FileNotFoundError(f"COCO data not found at {coco_path}. Run download_coco2017_to_volume.py first.")

        train_images = coco_path / "train2017"
        val_images = coco_path / "val2017"
        annotations = coco_path / "annotations"

        print(f"Train images: {train_images} (exists: {train_images.exists()})")
        print(f"Val images: {val_images} (exists: {val_images.exists()})")
        print(f"Annotations: {annotations} (exists: {annotations.exists()})")

        # Initialize runner with presets
        runner = SimpleRunner(
            model=model,
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            val_interval=val_interval,
            work_dir=work_dir,
        )

        # Start training
        runner.train()

        # Commit checkpoint volume if saving
        if save_checkpoints:
            checkpoint_volume.commit()

        metrics = _extract_metrics(work_dir)

        return TrainingResult(
            status="success",
            epochs_completed=epochs,
            final_metrics=metrics,
            work_dir=work_dir,
            duration_seconds=time.time() - start_time,
            error_message=None,
        )

    except Exception as e:
        import traceback

        return TrainingResult(
            status="error",
            epochs_completed=0,
            final_metrics={},
            work_dir=work_dir,
            duration_seconds=time.time() - start_time,
            error_message=f"{e!s}\n{traceback.format_exc()}",
        )


@app.local_entrypoint()
def main(
    model: str = "mask_rcnn_r50",
    epochs: int = 1,
    batch_size: int = 2,
    num_workers: int = 4,
    val_interval: int = 1,
    no_checkpoints: bool = False,
):
    """Local entrypoint for Modal training."""
    print(f"\n{'=' * 60}")
    print("Modal Mask R-CNN Training")
    print(f"{'=' * 60}")
    print(f"Model:         {model}")
    print(f"Epochs:        {epochs}")
    print(f"Batch Size:    {batch_size}")
    print(f"Num Workers:   {num_workers}")
    print(f"Val Interval:  {val_interval}")
    print(f"Save Checkpoints: {not no_checkpoints}")
    print(f"{'=' * 60}\n")

    result = train_mask_rcnn.remote(
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        val_interval=val_interval,
        save_checkpoints=not no_checkpoints,
    )

    print(f"\n{'=' * 60}")
    print("Training Result")
    print(f"{'=' * 60}")
    print(f"Status:    {result['status']}")
    print(f"Duration:  {result['duration_seconds']:.1f}s")
    print(f"Work Dir:  {result['work_dir']}")
    if result["final_metrics"]:
        print(f"Metrics:   {result['final_metrics']}")
    if result["error_message"]:
        print(f"Error:\n{result['error_message']}")
    print(f"{'=' * 60}")
