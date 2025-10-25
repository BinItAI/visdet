"""
Report generation for code comparison results.

Generates JSON and markdown output from comparison results.
"""

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Union

from codediff.comparator import ChangeType, ClassChange, FileChange, FunctionChange


@dataclass
class ComparisonReport:
    """Summary report of code comparison."""

    file_changes: Dict[str, FileChange]
    stats: Dict[str, Any]

    @classmethod
    def from_comparison(cls, file_changes: Dict[str, FileChange]) -> "ComparisonReport":
        """Create a report from comparison results.

        Args:
            file_changes: Dictionary of FileChange objects

        Returns:
            ComparisonReport instance
        """
        # Calculate statistics
        stats = cls._calculate_stats(file_changes)

        return cls(
            file_changes=file_changes,
            stats=stats,
        )

    @staticmethod
    def _calculate_stats(file_changes: Dict[str, FileChange]) -> Dict[str, Any]:
        """Calculate summary statistics from changes."""
        file_stats = {
            ChangeType.ADDED: 0,
            ChangeType.REMOVED: 0,
            ChangeType.MODIFIED: 0,
            ChangeType.RENAMED: 0,
            ChangeType.UNCHANGED: 0,
        }

        class_stats = {
            ChangeType.ADDED: 0,
            ChangeType.REMOVED: 0,
            ChangeType.MODIFIED: 0,
            ChangeType.RENAMED: 0,
        }

        function_stats = {
            ChangeType.ADDED: 0,
            ChangeType.REMOVED: 0,
            ChangeType.MODIFIED: 0,
            ChangeType.RENAMED: 0,
        }

        # Count changes
        for file_change in file_changes.values():
            file_stats[file_change.change_type] += 1

            for class_change in file_change.class_changes.values():
                class_stats[class_change.change_type] += 1

            for func_change in file_change.function_changes.values():
                function_stats[func_change.change_type] += 1

            # Count method changes
            for class_change in file_change.class_changes.values():
                for method_change in class_change.method_changes.values():
                    function_stats[method_change.change_type] += 1

        return {
            "total_files": len(file_changes),
            "files": dict(file_stats),
            "total_classes": sum(class_stats.values()),
            "classes": dict(class_stats),
            "total_functions": sum(function_stats.values()),
            "functions": dict(function_stats),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        output = {
            "stats": self.stats,
            "files": {},
        }

        for path, file_change in self.file_changes.items():
            output["files"][path] = self._serialize_file_change(file_change)

        return json.dumps(output, indent=indent, default=str)

    def to_markdown(self) -> str:
        """Convert report to markdown string.

        Returns:
            Markdown string representation
        """
        lines = []

        # Header and stats
        lines.append("# Code Comparison Report\n")

        lines.append("## Summary\n")
        lines.append(f"- **Total Files:** {self.stats['total_files']}\n")
        lines.append(f"- **Total Classes:** {self.stats['total_classes']}\n")
        lines.append(f"- **Total Functions:** {self.stats['total_functions']}\n")

        # File-level changes
        lines.append("\n## File Changes\n")
        for change_type in [ChangeType.REMOVED, ChangeType.ADDED, ChangeType.MODIFIED]:
            count = self.stats["files"].get(change_type.value, 0)
            if count > 0:
                lines.append(f"\n### {change_type.value.upper()} ({count})\n")
                for path, file_change in self.file_changes.items():
                    if file_change.change_type == change_type:
                        lines.append(f"- `{path}`\n")

        # Class-level changes
        if self.stats["total_classes"] > 0:
            lines.append("\n## Class Changes\n")
            for file_path, file_change in self.file_changes.items():
                if file_change.class_changes:
                    lines.append(f"\n### {file_path}\n")
                    for class_name, class_change in file_change.class_changes.items():
                        lines.append(
                            f"- **{class_change.change_type.value.upper()}**: "
                            f"`{class_change.name}` (line {class_change.source_line or class_change.target_line})\n"
                        )
                        if class_change.method_changes:
                            for method_name, method_change in class_change.method_changes.items():
                                lines.append(f"  - {method_change.change_type.value}: `{method_change.name}`\n")

        # Function-level changes
        if self.stats["total_functions"] > 0:
            lines.append("\n## Function Changes\n")
            for file_path, file_change in self.file_changes.items():
                if file_change.function_changes:
                    lines.append(f"\n### {file_path}\n")
                    for func_name, func_change in file_change.function_changes.items():
                        sig = func_change.new_signature or func_change.old_signature or ""
                        lines.append(f"- **{func_change.change_type.value.upper()}**: `{func_change.name}` {sig}\n")

        return "".join(lines)

    def save_json(self, path: Union[str, Path]) -> None:
        """Save report as JSON file.

        Args:
            path: Output file path
        """
        Path(path).write_text(self.to_json())

    def save_markdown(self, path: Union[str, Path]) -> None:
        """Save report as markdown file.

        Args:
            path: Output file path
        """
        Path(path).write_text(self.to_markdown())

    @staticmethod
    def _serialize_file_change(file_change: FileChange) -> Dict[str, Any]:
        """Serialize a FileChange object for JSON output."""
        return {
            "status": file_change.change_type.value,
            "classes": {
                name: {
                    "status": change.change_type.value,
                    "line": change.source_line or change.target_line,
                    "methods": {
                        method_name: {
                            "status": method_change.change_type.value,
                            "old_signature": method_change.old_signature,
                            "new_signature": method_change.new_signature,
                        }
                        for method_name, method_change in change.method_changes.items()
                    },
                }
                for name, change in file_change.class_changes.items()
            },
            "functions": {
                name: {
                    "status": change.change_type.value,
                    "line": change.source_line or change.target_line,
                    "old_signature": change.old_signature,
                    "new_signature": change.new_signature,
                }
                for name, change in file_change.function_changes.items()
            },
        }
