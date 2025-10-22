#!/usr/bin/env python3
"""Script to fix old imports in test files."""

import re
from pathlib import Path
from typing import Dict, List, Tuple

# Define import mappings - order matters (more specific first)
IMPORT_MAPPINGS: List[Tuple[re.Pattern, str]] = [
    # mmcv -> visdet.cv
    (re.compile(r"^(\s*)import mmcv$"), r"\1import visdet.cv as mmcv"),
    (re.compile(r"^(\s*)from mmcv\.(.+)"), r"\1from visdet.cv.\2"),
    (re.compile(r"^(\s*)from mmcv import (.+)"), r"\1from visdet.cv import \2"),
    # mmdet -> visdet
    (re.compile(r"^(\s*)import mmdet$"), r"\1import visdet as mmdet"),
    (re.compile(r"^(\s*)from mmdet\.(.+)"), r"\1from visdet.\2"),
    (re.compile(r"^(\s*)from mmdet import (.+)"), r"\1from visdet import \2"),
    # mmengine -> visdet.engine (just in case)
    (re.compile(r"^(\s*)import mmengine$"), r"\1import visdet.engine as mmengine"),
    (re.compile(r"^(\s*)from mmengine\.(.+)"), r"\1from visdet.engine.\2"),
    (re.compile(r"^(\s*)from mmengine import (.+)"), r"\1from visdet.engine import \2"),
]


def fix_imports_in_file(file_path: Path) -> Tuple[bool, int]:
    """
    Fix imports in a single file.

    Returns:
        Tuple of (was_modified, num_changes)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False, 0

    original_content = content
    lines = content.split("\n")
    modified_lines: List[str] = []
    num_changes = 0

    for line in lines:
        modified_line = line
        for pattern, replacement in IMPORT_MAPPINGS:
            match = pattern.match(line)
            if match:
                modified_line = pattern.sub(replacement, line)
                if modified_line != line:
                    num_changes += 1
                    print(f"  {file_path.name}: {line.strip()} -> {modified_line.strip()}")
                break

        modified_lines.append(modified_line)

    new_content = "\n".join(modified_lines)

    if new_content != original_content:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return True, num_changes
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            return False, 0

    return False, 0


def main():
    """Main function to process all test files."""
    tests_dir = Path(__file__).parent.parent / "tests"

    if not tests_dir.exists():
        print(f"Tests directory not found: {tests_dir}")
        return

    print(f"Scanning {tests_dir} for Python files...")

    # Find all Python files in tests directory
    py_files = list(tests_dir.rglob("*.py"))
    print(f"Found {len(py_files)} Python files")

    total_modified = 0
    total_changes = 0

    for py_file in py_files:
        was_modified, num_changes = fix_imports_in_file(py_file)
        if was_modified:
            total_modified += 1
            total_changes += num_changes

    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Files modified: {total_modified}")
    print(f"  Total changes: {total_changes}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
