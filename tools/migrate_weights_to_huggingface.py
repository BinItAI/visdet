#!/usr/bin/env python
"""Migrate model weights from OpenMMLab to Hugging Face Hub.

This script downloads all weights from the model zoo JSON files and uploads them
to Hugging Face Hub. It generates updated JSON mapping files with hf:// URLs.

Usage:
    python tools/migrate_weights_to_huggingface.py \
        --repo-id your-org/visdet-weights \
        --token YOUR_HF_TOKEN

Requirements:
    pip install huggingface_hub requests tqdm
"""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


def get_model_zoo_files():
    """Get paths to all model zoo JSON files."""
    hub_dir = Path(__file__).parent.parent / "visdet" / "engine" / "hub"
    return {
        "openmmlab": hub_dir / "openmmlab.json",
        "mmcls": hub_dir / "mmcls.json",
    }


def load_model_urls(json_path: Path) -> dict:
    """Load model URLs from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def download_file(url: str, dest_path: str, chunk_size: int = 8192) -> str:
    """Download a file from URL to destination path."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))

    return dest_path


def compute_sha256(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()[:8]


def get_filename_from_url(url: str) -> str:
    """Extract filename from URL."""
    return url.split("/")[-1]


def migrate_weights(
    repo_id: str,
    token: str,
    dry_run: bool = False,
    skip_existing: bool = True,
):
    """Migrate all weights to Hugging Face Hub.

    Args:
        repo_id: HuggingFace repository ID (e.g., 'your-org/visdet-weights')
        token: HuggingFace API token
        dry_run: If True, only print what would be done without uploading
        skip_existing: If True, skip files that already exist in the repo
    """
    api = HfApi(token=token)

    # Create repo if it doesn't exist
    if not dry_run:
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            print(f"Repository {repo_id} ready")
        except Exception as e:
            print(f"Note: {e}")

    # Get existing files in repo
    existing_files = set()
    if skip_existing and not dry_run:
        try:
            files = api.list_repo_files(repo_id=repo_id, repo_type="model")
            existing_files = set(files)
            print(f"Found {len(existing_files)} existing files in repo")
        except Exception:
            pass

    # Load all model zoo files
    zoo_files = get_model_zoo_files()
    hf_mappings = {}

    for zoo_name, json_path in zoo_files.items():
        if not json_path.exists():
            print(f"Skipping {zoo_name}: {json_path} not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing {zoo_name} ({json_path})")
        print(f"{'=' * 60}")

        urls = load_model_urls(json_path)
        hf_mappings[zoo_name] = {}

        for model_name, url in urls.items():
            filename = get_filename_from_url(url)
            # Organize by source (openmmlab, mmcls, etc.)
            hf_path = f"{zoo_name}/{filename}"

            if hf_path in existing_files:
                print(f"[SKIP] {model_name}: {hf_path} already exists")
                hf_mappings[zoo_name][model_name] = f"hf://{repo_id}/{hf_path}"
                continue

            if dry_run:
                print(f"[DRY-RUN] Would upload {model_name}: {url} -> {hf_path}")
                hf_mappings[zoo_name][model_name] = f"hf://{repo_id}/{hf_path}"
                continue

            # Download and upload
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = os.path.join(tmpdir, filename)

                try:
                    print(f"\n[DOWNLOAD] {model_name}")
                    download_file(url, local_path)

                    # Verify file was downloaded
                    if not os.path.exists(local_path):
                        print(f"[ERROR] Failed to download {model_name}")
                        continue

                    file_size = os.path.getsize(local_path)
                    sha = compute_sha256(local_path)
                    print(f"  Size: {file_size / 1024 / 1024:.2f} MB, SHA256: {sha}")

                    # Upload to HuggingFace
                    print(f"[UPLOAD] {hf_path}")
                    api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=hf_path,
                        repo_id=repo_id,
                        repo_type="model",
                    )

                    hf_mappings[zoo_name][model_name] = f"hf://{repo_id}/{hf_path}"
                    print(f"[DONE] {model_name}")

                except requests.exceptions.HTTPError as e:
                    print(f"[ERROR] Failed to download {model_name}: {e}")
                    continue
                except Exception as e:
                    print(f"[ERROR] Failed to process {model_name}: {e}")
                    continue

    # Generate HuggingFace mapping file
    output_dir = Path(__file__).parent.parent / "visdet" / "engine" / "hub"

    # Combine all mappings into one file
    combined_mapping = {}
    for zoo_name, mapping in hf_mappings.items():
        combined_mapping.update(mapping)

    hf_json_path = output_dir / "huggingface.json"

    if not dry_run:
        with open(hf_json_path, "w") as f:
            json.dump(combined_mapping, f, indent=2)
        print(f"\n{'=' * 60}")
        print(f"Generated HuggingFace mapping: {hf_json_path}")
        print(f"Total models: {len(combined_mapping)}")
    else:
        print(f"\n[DRY-RUN] Would write {len(combined_mapping)} mappings to {hf_json_path}")

    # Also upload a README to the HF repo
    if not dry_run:
        readme_content = f"""---
license: apache-2.0
tags:
- object-detection
- instance-segmentation
- visdet
- pytorch
---

# VisDet Model Weights

This repository contains pretrained weights for [VisDet](https://github.com/BinItAI/visdet),
a streamlined object detection and instance segmentation library.

## Models

The weights are organized by source:
- `openmmlab/`: Backbone weights from OpenMMLab's model zoo
- `mmcls/`: Classification backbone weights from MMClassification

## Usage

```python
from visdet.apis import init_detector

# Use HuggingFace-hosted weights
model = init_detector(
    config="configs/presets/models/mask_rcnn_r50.yaml",
    checkpoint="hf://{repo_id}/openmmlab/resnet50_msra-5891d200.pth"
)
```

Or in config files:
```python
model = dict(
    backbone=dict(
        init_cfg=dict(
            type="Pretrained",
            checkpoint="hf://{repo_id}/openmmlab/resnet50_msra-5891d200.pth"
        )
    )
)
```

## License

These weights are provided under the Apache 2.0 license, consistent with their original sources.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(readme_content)
            readme_path = f.name

        try:
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
            )
            print("Uploaded README.md to repo")
        finally:
            os.unlink(readme_path)

    return hf_mappings


def main():
    parser = argparse.ArgumentParser(description="Migrate model weights to Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., 'your-org/visdet-weights')",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually uploading",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-upload files even if they already exist in the repo",
    )

    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        parser.error("--token or HF_TOKEN environment variable required")

    migrate_weights(
        repo_id=args.repo_id,
        token=token,
        dry_run=args.dry_run,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
