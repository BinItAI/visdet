#!/usr/bin/env python3
"""
LibCST-based codemod to refactor viscv/visengine imports to visdet.cv/visdet.engine.

This script performs AST-based transformations to safely update import statements
without touching strings, comments, or other non-import code.

Usage:
    # Refactor a single file
    python scripts/refactor_imports.py path/to/file.py

    # Refactor multiple files
    python scripts/refactor_imports.py path/to/file1.py path/to/file2.py

    # Refactor a directory recursively
    python scripts/refactor_imports.py visdet/visdet/models/

Transformations (alias-preserving):
    import viscv                    → import visdet.cv as viscv
    import visengine                → import visdet.engine as visengine
    from visdet.cv.X import Y           → from visdet.cv.X import Y
    from visdet.engine.X import Y       → from visdet.engine.X import Y
"""

import argparse
import sys
from pathlib import Path
from typing import List, Union

import libcst as cst
from libcst import matchers as m


class ImportRefactorTransformer(cst.CSTTransformer):
    """LibCST transformer to refactor viscv/visengine imports to visdet.cv/visdet.engine."""

    def __init__(self):
        self.changes_made = 0

    def leave_Import(self, original_node: cst.Import, updated_node: cst.Import) -> Union[cst.Import, cst.ImportFrom]:
        """
        Transform:
            import viscv → import visdet.cv as viscv
            import visengine → import visdet.engine as visengine
        """
        # Handle import statements with ImportAlias nodes
        new_names = []
        made_change = False

        for name in updated_node.names:
            if isinstance(name, cst.ImportAlias):
                module_name = name.name

                # Check if it's a simple name (not dotted)
                if isinstance(module_name, cst.Name):
                    if module_name.value == "viscv":
                        # import viscv → import visdet.cv as viscv
                        new_name = cst.ImportAlias(
                            name=cst.Attribute(value=cst.Name("visdet"), attr=cst.Name("cv")),
                            asname=cst.AsName(
                                name=cst.Name("viscv"),
                                whitespace_before_as=cst.SimpleWhitespace(" "),
                                whitespace_after_as=cst.SimpleWhitespace(" "),
                            ),
                        )
                        new_names.append(new_name)
                        made_change = True
                        self.changes_made += 1
                        continue
                    elif module_name.value == "visengine":
                        # import visengine → import visdet.engine as visengine
                        new_name = cst.ImportAlias(
                            name=cst.Attribute(value=cst.Name("visdet"), attr=cst.Name("engine")),
                            asname=cst.AsName(
                                name=cst.Name("visengine"),
                                whitespace_before_as=cst.SimpleWhitespace(" "),
                                whitespace_after_as=cst.SimpleWhitespace(" "),
                            ),
                        )
                        new_names.append(new_name)
                        made_change = True
                        self.changes_made += 1
                        continue

            # Keep other imports unchanged
            new_names.append(name)

        if made_change:
            return updated_node.with_changes(names=new_names)

        return updated_node

    def leave_ImportFrom(self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom) -> cst.ImportFrom:
        """
        Transform:
            from visdet.cv.X import Y → from visdet.cv.X import Y
            from visdet.engine.X import Y → from visdet.engine.X import Y
        """
        module = updated_node.module
        if module is None:
            return updated_node

        # Check if import is from viscv or visengine
        new_module = self._transform_module_path(module)

        if new_module != module:
            self.changes_made += 1
            return updated_node.with_changes(module=new_module)

        return updated_node

    def _transform_module_path(self, module: cst.BaseExpression) -> cst.BaseExpression:
        """Transform viscv.X → visdet.cv.X and visengine.X → visdet.engine.X"""

        if isinstance(module, cst.Name):
            # Handle simple case: from visdet.cv import X
            if module.value == "viscv":
                return cst.Attribute(value=cst.Name("visdet"), attr=cst.Name("cv"))
            elif module.value == "visengine":
                return cst.Attribute(value=cst.Name("visdet"), attr=cst.Name("engine"))

        elif isinstance(module, cst.Attribute):
            # Handle dotted imports: from visdet.cv.transforms import X
            # Recursively transform the left side
            new_value = self._transform_module_path(module.value)

            if new_value != module.value:
                return module.with_changes(value=new_value)

        return module


def refactor_file(file_path: Path, dry_run: bool = False) -> bool:
    """
    Refactor imports in a single file.

    Args:
        file_path: Path to the Python file
        dry_run: If True, don't write changes, just report what would change

    Returns:
        True if changes were made, False otherwise
    """
    try:
        source_code = file_path.read_text()
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return False

    # Parse the source code
    try:
        source_tree = cst.parse_module(source_code)
    except Exception as e:
        print(f"❌ Error parsing {file_path}: {e}")
        return False

    # Apply transformations
    transformer = ImportRefactorTransformer()
    modified_tree = source_tree.visit(transformer)

    if transformer.changes_made == 0:
        return False

    # Generate modified code
    modified_code = modified_tree.code

    if dry_run:
        print(f"🔍 {file_path}: {transformer.changes_made} import(s) would be changed")
        return True

    # Write changes
    try:
        file_path.write_text(modified_code)
        print(f"✅ {file_path}: {transformer.changes_made} import(s) refactored")
        return True
    except Exception as e:
        print(f"❌ Error writing {file_path}: {e}")
        return False


def refactor_directory(directory: Path, dry_run: bool = False) -> tuple[int, int]:
    """
    Refactor all Python files in a directory recursively.

    Args:
        directory: Path to the directory
        dry_run: If True, don't write changes

    Returns:
        Tuple of (files_changed, total_files)
    """
    python_files = list(directory.rglob("*.py"))
    files_changed = 0

    for file_path in python_files:
        if refactor_file(file_path, dry_run=dry_run):
            files_changed += 1

    return files_changed, len(python_files)


def main():
    parser = argparse.ArgumentParser(
        description="Refactor viscv/visengine imports to visdet.cv/visdet.engine using LibCST"
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Files or directories to refactor")
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes, just report what would change")

    args = parser.parse_args()

    print("=" * 70)
    print("IMPORT REFACTORING" + (" (DRY RUN)" if args.dry_run else ""))
    print("=" * 70)

    total_files_changed = 0
    total_files_processed = 0

    for path in args.paths:
        if not path.exists():
            print(f"❌ Path not found: {path}")
            continue

        if path.is_file():
            if path.suffix == ".py":
                if refactor_file(path, dry_run=args.dry_run):
                    total_files_changed += 1
                total_files_processed += 1
            else:
                print(f"⚠️  Skipping non-Python file: {path}")
        elif path.is_dir():
            files_changed, files_processed = refactor_directory(path, dry_run=args.dry_run)
            total_files_changed += files_changed
            total_files_processed += files_processed

    print("\n" + "=" * 70)
    print(f"SUMMARY: {total_files_changed}/{total_files_processed} files modified")
    print("=" * 70)

    return 0 if not args.dry_run or total_files_changed > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
