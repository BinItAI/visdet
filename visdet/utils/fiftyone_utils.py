# ruff: noqa
# type: ignore
"""FiftyOne integration utilities for visdet.

This module provides functions to convert visdet predictions to FiftyOne format
and load them into FiftyOne datasets for visualization.

Example usage:
    >>> import fiftyone as fo
    >>> from visdet.utils.fiftyone_utils import load_inference_results
    >>>
    >>> # Load results from Modal inference
    >>> dataset = load_inference_results("inference_results.json", name="visdet-demo")
    >>>
    >>> # Launch FiftyOne app
    >>> session = fo.launch_app(dataset)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# FiftyOne is optional - only import when needed
try:
    import fiftyone as fo

    FIFTYONE_AVAILABLE = True
except ImportError:
    FIFTYONE_AVAILABLE = False
    fo = None


def _check_fiftyone():
    """Check if FiftyOne is installed."""
    if not FIFTYONE_AVAILABLE:
        raise ImportError("FiftyOne is not installed. Install it with: pip install fiftyone")


def detections_to_fiftyone(
    detections: list[dict[str, Any]],
) -> "fo.Detections":
    """Convert a list of detection dicts to FiftyOne Detections.

    Args:
        detections: List of detection dicts with keys:
            - label: str - class name
            - bounding_box: [x, y, w, h] - normalized coords
            - confidence: float - detection score

    Returns:
        FiftyOne Detections object
    """
    _check_fiftyone()

    fo_detections = []
    for det in detections:
        fo_det = fo.Detection(
            label=det["label"],
            bounding_box=det["bounding_box"],
            confidence=det.get("confidence"),
        )
        fo_detections.append(fo_det)

    return fo.Detections(detections=fo_detections)


def load_inference_results(
    results_path: str | Path,
    name: str = "visdet-inference",
    image_base_path: str | Path | None = None,
    field_name: str = "predictions",
) -> "fo.Dataset":
    """Load inference results from JSON into a FiftyOne dataset.

    Args:
        results_path: Path to JSON file from Modal inference
        name: Name for the FiftyOne dataset
        image_base_path: Base path to prepend to image file names.
            If None, uses paths from the results file.
        field_name: Name of the detections field in the dataset

    Returns:
        FiftyOne Dataset with loaded predictions
    """
    _check_fiftyone()

    results_path = Path(results_path)
    with open(results_path) as f:
        data = json.load(f)

    # Create dataset
    dataset = fo.Dataset(name=name, overwrite=True)

    # Get metadata
    class_names = data.get("class_names", [])
    backbone_key = data.get("backbone_key", "unknown")

    dataset.info = {
        "backbone": backbone_key,
        "config": data.get("config", ""),
        "num_classes": len(class_names),
    }

    # Add samples
    samples = []
    for result in data.get("results", []):
        # Determine image path
        if image_base_path:
            image_path = Path(image_base_path) / result["file_name"]
        else:
            image_path = result.get("image_path", result["file_name"])

        # Create sample
        sample = fo.Sample(filepath=str(image_path))

        # Add image metadata
        sample["image_id"] = result.get("image_id")
        sample["width"] = result.get("width")
        sample["height"] = result.get("height")

        # Add detections
        detections = result.get("detections", [])
        if detections:
            sample[field_name] = detections_to_fiftyone(detections)

        samples.append(sample)

    dataset.add_samples(samples)

    return dataset


def add_predictions_to_dataset(
    dataset: "fo.Dataset",
    results: dict[str, Any] | list[dict[str, Any]],
    field_name: str = "predictions",
    match_by: str = "image_id",
) -> None:
    """Add predictions to an existing FiftyOne dataset.

    Args:
        dataset: Existing FiftyOne dataset
        results: Either a dict with "results" key, or list of result dicts
        field_name: Name of the detections field to add
        match_by: Field to match samples by ("image_id" or "filepath")
    """
    _check_fiftyone()

    # Handle both formats
    if isinstance(results, dict):
        result_list = results.get("results", [])
    else:
        result_list = results

    # Build lookup
    if match_by == "image_id":
        sample_lookup = {s["image_id"]: s for s in dataset if "image_id" in s}
    else:
        sample_lookup = {Path(s.filepath).name: s for s in dataset}

    # Add predictions
    for result in result_list:
        if match_by == "image_id":
            key = result.get("image_id")
        else:
            key = result.get("file_name")

        sample = sample_lookup.get(key)
        if sample is None:
            continue

        detections = result.get("detections", [])
        if detections:
            sample[field_name] = detections_to_fiftyone(detections)
            sample.save()


def create_coco_dataset(
    coco_root: str | Path,
    split: str = "val2017",
    name: str | None = None,
    max_samples: int | None = None,
) -> "fo.Dataset":
    """Create a FiftyOne dataset from COCO images.

    Args:
        coco_root: Path to COCO dataset root (containing val2017/, annotations/)
        split: Image split to load (e.g., "val2017", "train2017")
        name: Dataset name (defaults to f"coco-{split}")
        max_samples: Maximum number of samples to load

    Returns:
        FiftyOne Dataset with COCO images (no annotations)
    """
    _check_fiftyone()

    coco_root = Path(coco_root)
    images_dir = coco_root / split
    ann_file = coco_root / "annotations" / f"instances_{split}.json"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Load annotations for image metadata
    with open(ann_file) as f:
        coco_data = json.load(f)

    name = name or f"coco-{split}"
    dataset = fo.Dataset(name=name, overwrite=True)

    samples = []
    for i, img_info in enumerate(coco_data.get("images", [])):
        if max_samples and i >= max_samples:
            break

        image_path = images_dir / img_info["file_name"]
        if not image_path.exists():
            continue

        sample = fo.Sample(filepath=str(image_path))
        sample["image_id"] = img_info["id"]
        sample["width"] = img_info["width"]
        sample["height"] = img_info["height"]
        samples.append(sample)

    dataset.add_samples(samples)

    return dataset


def visualize_results(
    results_path: str | Path,
    image_base_path: str | Path | None = None,
    name: str = "visdet-inference",
    port: int = 5151,
) -> "fo.Session":
    """Quick visualization of inference results.

    Args:
        results_path: Path to JSON file from Modal inference
        image_base_path: Base path to images (if different from results)
        name: Dataset name
        port: Port for FiftyOne app

    Returns:
        FiftyOne Session object
    """
    _check_fiftyone()

    dataset = load_inference_results(
        results_path,
        name=name,
        image_base_path=image_base_path,
    )

    session = fo.launch_app(dataset, port=port)
    return session
