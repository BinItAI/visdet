"""
Command-line interface for the code comparison tool.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Set

from codediff.comparator import ChangeType, CodeComparator
from codediff.report import ComparisonReport


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Semantic code comparison tool - compare Python code ignoring whitespace/formatting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two directories
  codediff archive/mmdet/tests/ tests/ --output report.json

  # Generate markdown report
  codediff archive/mmdet/tests/ tests/ --format markdown --output report.md

  # Show only missing tests
  codediff tests/ visdet/tests/ --missing-only

  # Compare specific files
  codediff archive/mmdet/file.py visdet/file.py
        """,
    )

    parser.add_argument(
        "source",
        help="Source directory or file to compare",
    )

    parser.add_argument(
        "target",
        help="Target directory or file to compare",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (default: stdout)",
    )

    parser.add_argument(
        "-f",
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format (default: json)",
    )

    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Show only missing items (removed from source, not in target)",
    )

    parser.add_argument(
        "--added-only",
        action="store_true",
        help="Show only added items (in target, not in source)",
    )

    parser.add_argument(
        "--renamed-only",
        action="store_true",
        help="Show only renamed items",
    )

    parser.add_argument(
        "--modified-only",
        action="store_true",
        help="Show only modified items",
    )

    parser.add_argument(
        "--function",
        action="append",
        dest="functions",
        help="Show only changes for specific function(s) (can be used multiple times)",
    )

    parser.add_argument(
        "--class",
        action="append",
        dest="classes",
        help="Show only changes for specific class(es) (can be used multiple times)",
    )

    parser.add_argument(
        "--pattern",
        help="Filter by name pattern (regex)",
    )

    parser.add_argument(
        "--show-diff",
        action="store_true",
        help="Show detailed code differences (for modified items)",
    )

    parser.add_argument(
        "--context-lines",
        type=int,
        default=3,
        help="Number of context lines to show in diffs (default: 3)",
    )

    parser.add_argument(
        "--normalize-imports",
        action="store_true",
        help="Normalize imports for feature parity checking: replace mmcv→visdet.cv and mmengine→visdet.engine",
    )

    args = parser.parse_args()

    try:
        comparator = CodeComparator(normalize_imports=args.normalize_imports)

        source_path = Path(args.source)
        target_path = Path(args.target)

        if not source_path.exists():
            print(f"Error: Source path does not exist: {args.source}", file=sys.stderr)
            sys.exit(1)

        if not target_path.exists():
            print(f"Error: Target path does not exist: {args.target}", file=sys.stderr)
            sys.exit(1)

        # Determine if we're comparing files or directories
        if source_path.is_file() and target_path.is_file():
            # Compare files
            file_changes = comparator.compare_files(source_path, target_path)
            changes = {"_": file_changes}
        else:
            # Compare directories
            changes = comparator.compare_directories(source_path, target_path)

        # Filter changes if requested
        if any(
            [
                args.missing_only,
                args.added_only,
                args.renamed_only,
                args.modified_only,
                args.functions,
                args.classes,
                args.pattern,
            ]
        ):
            changes = _filter_changes(changes, args)

        # Create report
        report = ComparisonReport.from_comparison(changes, show_diff=args.show_diff, context_lines=args.context_lines)

        # Generate output
        if args.format == "json":
            output = report.to_json()
        else:
            output = report.to_markdown()

        # Write output
        if args.output:
            Path(args.output).write_text(output)
            print(f"Report written to {args.output}")
        else:
            print(output)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def _filter_changes(changes: dict, args) -> dict:
    """Filter changes based on command-line arguments."""
    filtered = {}

    # Compile pattern if provided
    pattern = None
    if args.pattern:
        try:
            pattern = re.compile(args.pattern)
        except re.error as e:
            print(f"Error: Invalid regex pattern '{args.pattern}': {e}", file=sys.stderr)
            sys.exit(1)

    # Get sets of names to filter
    target_functions: Set[str] = set(args.functions or [])
    target_classes: Set[str] = set(args.classes or [])

    for path, file_change in changes.items():
        # Filter based on change type
        if args.missing_only and file_change.change_type != ChangeType.REMOVED:
            continue
        if args.added_only and file_change.change_type != ChangeType.ADDED:
            continue
        if args.modified_only and file_change.change_type != ChangeType.MODIFIED:
            continue

        # Filter classes and functions
        if args.renamed_only:
            file_change.class_changes = {
                k: v for k, v in file_change.class_changes.items() if v.change_type == ChangeType.RENAMED
            }
            file_change.function_changes = {
                k: v for k, v in file_change.function_changes.items() if v.change_type == ChangeType.RENAMED
            }

        # Filter by function name
        if target_functions:
            file_change.function_changes = {
                k: v
                for k, v in file_change.function_changes.items()
                if k in target_functions or _matches_pattern(k, target_functions)
            }

        # Filter by class name
        if target_classes:
            file_change.class_changes = {
                k: v
                for k, v in file_change.class_changes.items()
                if k in target_classes or _matches_pattern(k, target_classes)
            }
            # Also filter methods within classes
            for class_change in file_change.class_changes.values():
                class_change.method_changes = {
                    k: v
                    for k, v in class_change.method_changes.items()
                    if k in target_functions or _matches_pattern(k, target_functions)
                }

        # Filter by pattern (regex)
        if pattern:
            file_change.function_changes = {k: v for k, v in file_change.function_changes.items() if pattern.search(k)}
            file_change.class_changes = {k: v for k, v in file_change.class_changes.items() if pattern.search(k)}
            for class_change in file_change.class_changes.values():
                class_change.method_changes = {
                    k: v for k, v in class_change.method_changes.items() if pattern.search(k)
                }

        # Only include if there are changes after filtering
        if (
            file_change.class_changes
            or file_change.function_changes
            or file_change.change_type in [ChangeType.ADDED, ChangeType.REMOVED]
        ):
            filtered[path] = file_change

    return filtered


def _matches_pattern(name: str, target_names: Set[str]) -> bool:
    """Check if name matches any pattern in target_names using wildcards."""
    for target in target_names:
        # Simple wildcard matching
        pattern = target.replace("*", ".*").replace("?", ".")
        if re.match(f"^{pattern}$", name):
            return True
    return False


if __name__ == "__main__":
    main()
