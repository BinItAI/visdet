"""Run batch inference on Modal with any backbone from HuggingFace weights.

This script loads a detector with pretrained backbone weights from HuggingFace
and runs inference on COCO val2017 images, returning results in FiftyOne-compatible format.

Prereqs:
  1) `modal` installed + authenticated
  2) COCO volume populated:
       VISDET_COCO_VOLUME=visdet-coco python -m modal run tools/modal/download_coco2017_to_volume.py

Run:
  VISDET_COCO_VOLUME=visdet-coco python -m modal run tools/modal/inference_batch.py --backbone resnet50 --num-images 10
"""

from __future__ import annotations

import argparse
import json
import os
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
    this_file = Path(__file__).resolve()
    for parent in [this_file.parent, *this_file.parents]:
        if (parent / "pyproject.toml").is_file() and (parent / "visdet").is_dir():
            return parent

    modal_repo = Path(REMOTE_REPO_PATH)
    if (modal_repo / "visdet").is_dir() and (modal_repo / "configs").is_dir():
        return modal_repo

    return this_file.parent


REPO_ROOT = _find_repo_root()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "pycocotools",
        "pydantic>=2.0.0",
        "pyyaml",
        "matplotlib",
        "rich",
        "scipy",
        "shapely",
        "addict",
        "packaging",
        "six",
        "termcolor",
        "terminaltables",
        "tqdm",
        "yapf",
        "huggingface_hub",
    )
    .add_local_dir(
        REPO_ROOT / "visdet",
        remote_path=f"{REMOTE_REPO_PATH}/visdet",
        copy=True,
        ignore=_ignore_repo_path,
    )
    .add_local_dir(
        REPO_ROOT / "configs",
        remote_path=f"{REMOTE_REPO_PATH}/configs",
        copy=True,
        ignore=_ignore_repo_path,
    )
)

app = modal.App("visdet-inference-batch", image=image)
volume = modal.Volume.from_name(DEFAULT_VOLUME_NAME, create_if_missing=False)


# Mapping from backbone keys to detector configs
# These configs use the corresponding backbone architecture
BACKBONE_TO_CONFIG = {
    # ResNet variants
    "resnet50": "mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py",
    "resnet101": "mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py",
    "detectron/resnet50_caffe": "mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_coco.py",
    "detectron/resnet101_caffe": "mask_rcnn/mask_rcnn_r101_caffe_fpn_1x_coco.py",
    "detectron2/resnet50_caffe": "mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_coco.py",
    "detectron2/resnet101_caffe": "mask_rcnn/mask_rcnn_r101_caffe_fpn_1x_coco.py",
    # Swin Transformer
    "swin_tiny": "mask_rcnn/mask_rcnn_swin_tiny_fpn_1x_coco.py",
    "swin_small": "swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py",
    # ResNeXt
    "resnext50_32x4d": "resnext/mask_rcnn_x50_32x4d_fpn_1x_coco.py",
    "resnext101_32x4d": "resnext/mask_rcnn_x101_32x4d_fpn_1x_coco.py",
    "resnext101_64x4d": "resnext/mask_rcnn_x101_64x4d_fpn_1x_coco.py",
    # ResNeSt
    "resnest50": "resnest/mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco.py",
    "resnest101": "resnest/mask_rcnn_s101_fpn_syncbn-backbone+head_mstrain_1x_coco.py",
    # HRNet
    "msra/hrnetv2_w18": "hrnet/mask_rcnn_hrnetv2p_w18_1x_coco.py",
    "msra/hrnetv2_w32": "hrnet/mask_rcnn_hrnetv2p_w32_1x_coco.py",
    "msra/hrnetv2_w40": "hrnet/mask_rcnn_hrnetv2p_w40_1x_coco.py",
    # RegNet
    "regnetx_400mf": "regnet/mask_rcnn_regnetx-400MF_fpn_1x_coco.py",
    "regnetx_800mf": "regnet/mask_rcnn_regnetx-800MF_fpn_1x_coco.py",
    "regnetx_1.6gf": "regnet/mask_rcnn_regnetx-1.6GF_fpn_1x_coco.py",
    "regnetx_3.2gf": "regnet/mask_rcnn_regnetx-3.2GF_fpn_1x_coco.py",
    "regnetx_4.0gf": "regnet/mask_rcnn_regnetx-4.0GF_fpn_1x_coco.py",
}


def prediction_to_fiftyone_format(
    pred_instances: Any,
    image_width: int,
    image_height: int,
    class_names: list[str],
    score_threshold: float = 0.3,
) -> list[dict[str, Any]]:
    """Convert visdet predictions to FiftyOne-compatible detection format.

    FiftyOne expects bboxes as [x, y, width, height] normalized to [0, 1].
    visdet uses [x1, y1, x2, y2] in pixel coordinates.
    """
    detections = []

    if not hasattr(pred_instances, "bboxes"):
        return detections

    bboxes = pred_instances.bboxes.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    labels = pred_instances.labels.cpu().numpy()

    # Check for masks
    has_masks = hasattr(pred_instances, "masks") and pred_instances.masks is not None

    for i in range(len(bboxes)):
        score = float(scores[i])
        if score < score_threshold:
            continue

        # Convert [x1, y1, x2, y2] to normalized [x, y, w, h]
        x1, y1, x2, y2 = bboxes[i]
        x = float(x1) / image_width
        y = float(y1) / image_height
        w = float(x2 - x1) / image_width
        h = float(y2 - y1) / image_height

        label_idx = int(labels[i])
        label = class_names[label_idx] if label_idx < len(class_names) else str(label_idx)

        detection = {
            "label": label,
            "bounding_box": [x, y, w, h],
            "confidence": score,
        }

        # Add mask if available (RLE encoded for FiftyOne)
        if has_masks:
            # For now, skip masks to keep output size manageable
            # Can add RLE encoding later if needed
            pass

        detections.append(detection)

    return detections


@app.function(
    timeout=60 * 60,  # 1 hour
    gpu="T4",
    memory=16_000,
    volumes={DATA_MOUNT_PATH: volume},
)
def run_inference(
    backbone_key: str,
    num_images: int = 10,
    score_threshold: float = 0.3,
) -> dict[str, Any]:
    """Run inference with specified backbone on COCO val images.

    Args:
        backbone_key: Key from huggingface.json (e.g., "resnet50", "swin_tiny")
        num_images: Number of COCO val images to process
        score_threshold: Minimum confidence score for detections

    Returns:
        Dict with predictions in FiftyOne-compatible format
    """
    import sys

    sys.path.insert(0, REMOTE_REPO_PATH)

    from visdet.apis.inference import inference_detector, init_detector
    from visdet.engine.hub import get_huggingface_url

    # Get config path
    if backbone_key not in BACKBONE_TO_CONFIG:
        available = list(BACKBONE_TO_CONFIG.keys())
        raise ValueError(f"Unknown backbone key: {backbone_key}. Available: {available}")

    config_path = Path(REMOTE_REPO_PATH) / "configs" / BACKBONE_TO_CONFIG[backbone_key]
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Get HuggingFace checkpoint URL
    checkpoint_url = get_huggingface_url(backbone_key)
    if checkpoint_url is None:
        raise ValueError(f"No HuggingFace URL found for backbone: {backbone_key}")

    print(f"VISDET_INFERENCE: Loading model with config={config_path}")
    print(f"VISDET_INFERENCE: Checkpoint URL={checkpoint_url}")

    # Initialize detector
    # Note: For backbone-only weights, we load without checkpoint and rely on
    # the backbone init_cfg in the config. For full detector weights, we'd pass
    # the checkpoint directly.
    model = init_detector(
        config=str(config_path),
        checkpoint=None,  # Using pretrained backbone from config
        device="cuda:0",
    )

    # Get COCO val images
    coco_root = Path(DATA_MOUNT_PATH) / COCO_DIRNAME
    coco_images_dir = coco_root / "val2017"
    ann_file = coco_root / "annotations" / "instances_val2017.json"

    if not coco_images_dir.is_dir():
        raise RuntimeError(f"COCO image dir missing: {coco_images_dir}")

    # Load annotations to get image info
    with open(ann_file) as f:
        coco_ann = json.load(f)

    # Get image paths
    image_infos = coco_ann["images"][:num_images]
    image_paths = [str(coco_images_dir / img["file_name"]) for img in image_infos]

    print(f"VISDET_INFERENCE: Running inference on {len(image_paths)} images")

    # Get class names
    class_names = list(model.dataset_meta.get("classes", []))

    # Run inference
    results = []
    for i, (img_path, img_info) in enumerate(zip(image_paths, image_infos)):
        print(f"VISDET_INFERENCE: Processing image {i + 1}/{len(image_paths)}: {img_info['file_name']}")

        result = inference_detector(model, img_path)

        # Convert to FiftyOne format
        detections = prediction_to_fiftyone_format(
            result.pred_instances,
            image_width=img_info["width"],
            image_height=img_info["height"],
            class_names=class_names,
            score_threshold=score_threshold,
        )

        results.append(
            {
                "image_id": img_info["id"],
                "file_name": img_info["file_name"],
                "image_path": img_path,
                "width": img_info["width"],
                "height": img_info["height"],
                "detections": detections,
            }
        )

    print(f"VISDET_INFERENCE: Completed inference on {len(results)} images")

    return {
        "backbone_key": backbone_key,
        "config": str(BACKBONE_TO_CONFIG[backbone_key]),
        "num_images": len(results),
        "class_names": class_names,
        "results": results,
    }


@app.local_entrypoint()
def main(
    backbone: str = "resnet50",
    num_images: int = 10,
    score_threshold: float = 0.3,
    output_file: str | None = None,
):
    """Run inference and optionally save results."""
    print(f"Running inference with backbone={backbone}, num_images={num_images}")

    result = run_inference.remote(
        backbone_key=backbone,
        num_images=num_images,
        score_threshold=score_threshold,
    )

    print(f"\nCompleted! Processed {result['num_images']} images")
    print(f"Config used: {result['config']}")

    # Save results if output file specified
    if output_file:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {output_file}")
    else:
        # Print summary
        total_detections = sum(len(r["detections"]) for r in result["results"])
        print(f"Total detections: {total_detections}")
        print("\nTo visualize in FiftyOne, save results with --output-file and use:")
        print("  from visdet.utils.fiftyone_utils import load_predictions_to_dataset")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch inference on Modal")
    parser.add_argument(
        "--backbone",
        default="resnet50",
        help=f"Backbone key from huggingface.json. Available: {list(BACKBONE_TO_CONFIG.keys())}",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Number of COCO val images to process",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Minimum confidence score for detections",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save JSON results",
    )
    args = parser.parse_args()

    main(
        backbone=args.backbone,
        num_images=args.num_images,
        score_threshold=args.score_threshold,
        output_file=args.output_file,
    )
