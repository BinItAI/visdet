"""Generate comprehensive model zoo documentation from config READMEs."""

import re
from pathlib import Path
from typing import Dict, List, Tuple


def extract_model_family_info(readme_path: Path) -> Dict[str, str]:
    """Extract model family information from README."""
    content = readme_path.read_text()

    info = {
        "name": "",
        "title": "",
        "paper_link": "",
        "abstract": "",
        "image_url": "",
    }

    # Extract title (first h1)
    title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
    if title_match:
        info["title"] = title_match.group(1).strip()
        info["name"] = info["title"]

    # Extract paper link
    paper_match = re.search(r"\[([^\]]+)\]\((https://arxiv.org/abs/[^\)]+)\)", content)
    if paper_match:
        info["paper_link"] = paper_match.group(2)

    # Extract abstract
    abstract_match = re.search(r"## Abstract\s+(.+?)(?=\n##|\n<div|\Z)", content, re.DOTALL)
    if abstract_match:
        # Take first 200 chars
        abstract = abstract_match.group(1).strip()
        info["abstract"] = abstract[:200] + "..." if len(abstract) > 200 else abstract

    # Extract image URL
    img_match = re.search(r'<img src="(https://[^"]+)"', content)
    if img_match:
        info["image_url"] = img_match.group(1)

    return info


def extract_results_table(readme_path: Path) -> List[str]:
    """Extract the results table from README."""
    content = readme_path.read_text()

    # Find the Results and Models section
    results_match = re.search(r"## Results and Models\s+(.*?)(?=\n##|\Z)", content, re.DOTALL)

    if not results_match:
        return []

    results_section = results_match.group(1)

    # Extract markdown table
    table_lines = []
    in_table = False
    for line in results_section.split("\n"):
        if line.strip().startswith("|"):
            in_table = True
            table_lines.append(line)
        elif in_table and not line.strip():
            break

    return table_lines if len(table_lines) > 2 else []


def generate_model_zoo_page() -> str:
    """Generate the complete model zoo page."""
    configs_dir = Path("configs")

    # Find all README files
    readme_files = sorted(configs_dir.glob("*/README.md"))

    # Group models by category
    categories = {
        "Two-Stage Detectors": [
            "faster_rcnn",
            "mask_rcnn",
            "cascade_rcnn",
            "cascade_mask_rcnn",
            "htc",
            "ms_rcnn",
            "grid_rcnn",
            "sparse_rcnn",
            "queryinst",
        ],
        "One-Stage Detectors": ["retinanet", "fcos", "atss", "gfl", "vfnet", "paa", "tood", "autoassign", "ddod"],
        "YOLO Series": ["yolo", "yolof", "yolox", "yolact"],
        "Transformer-Based": ["detr", "deformable_detr", "mask2former", "maskformer"],
        "Instance Segmentation": ["solo", "solov2", "mask_rcnn"],
        "Panoptic Segmentation": ["panoptic_fpn", "mask2former"],
        "Specialized Backbones": ["hrnet", "swin", "pvt", "convnext", "regnet", "resnest", "res2net", "efficientnet"],
        "Other Methods": [],
    }

    # Build model family map
    model_families = {}
    for readme in readme_files:
        family_name = readme.parent.name
        info = extract_model_family_info(readme)
        table = extract_results_table(readme)

        model_families[family_name] = {"info": info, "table": table, "path": readme}

    # Categorize models
    categorized = {cat: [] for cat in categories}
    uncategorized = []

    for family_name in model_families:
        added = False
        for category, families in categories.items():
            if family_name in families:
                categorized[category].append(family_name)
                added = True
                break
        if not added and category != "Other Methods":
            uncategorized.append(family_name)

    categorized["Other Methods"] = uncategorized

    # Generate markdown
    lines = [
        "# Model Zoo",
        "",
        "This page provides a comprehensive overview of all available models in visdet.",
        "Each model family includes performance metrics, configuration files, and pre-trained weights.",
        "",
        "## Overview",
        "",
        f"visdet includes **{len(model_families)}** model families covering various object detection and instance segmentation architectures.",
        "",
        "## Table of Contents",
        "",
    ]

    # Add TOC
    for category, families in categorized.items():
        if families:
            lines.append(f"- [{category}](#{category.lower().replace(' ', '-').replace('-', '')})")

    lines.append("")

    # Add each category
    for category, families in categorized.items():
        if not families:
            continue

        lines.extend([f"## {category}", ""])

        for family_name in sorted(families):
            if family_name not in model_families:
                continue

            family_data = model_families[family_name]
            info = family_data["info"]
            table = family_data["table"]

            lines.extend([f"### {info['title'] or family_name}", ""])

            if info["paper_link"]:
                lines.append(f"**Paper:** [{info['title']}]({info['paper_link']})")
                lines.append("")

            if info["abstract"]:
                lines.append(info["abstract"])
                lines.append("")

            if table:
                lines.extend(table)
                lines.append("")

            lines.append(
                f"[**Full Documentation →**](https://github.com/BinItAI/visdet/tree/master/configs/{family_name}/README.md)"
            )
            lines.append("")
            lines.append("---")
            lines.append("")

    # Add footer
    lines.extend(
        [
            "## Using Pre-trained Models",
            "",
            "To use any pre-trained model from the model zoo:",
            "",
            "```python",
            "from mmdet.apis import init_detector, inference_detector",
            "",
            "# Load model",
            "config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'",
            "checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco.pth'",
            "model = init_detector(config_file, checkpoint_file, device='cuda:0')",
            "",
            "# Run inference",
            "result = inference_detector(model, 'demo/demo.jpg')",
            "```",
            "",
            "## Training Custom Models",
            "",
            "See the [Training Guide](../user-guide/training.md) for instructions on training models with your own data.",
            "",
            "## Contributing New Models",
            "",
            "We welcome contributions of new model implementations! Please see the [Contributing Guide](../development/contributing.md) for details.",
            "",
        ]
    )

    return "\n".join(lines)


if __name__ == "__main__":
    # Generate the page
    content = generate_model_zoo_page()

    # Write to docs
    output_path = Path("mkdocs_docs/model-zoo.md")
    output_path.write_text(content)

    print(f"✓ Generated model zoo documentation: {output_path}")
    print(f"  Total length: {len(content)} characters")
