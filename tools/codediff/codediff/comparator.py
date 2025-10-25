"""
Code comparison engine implementing two-pass comparison strategy.

Pass 1: Match by name
Pass 2: Match unmatched items by semantic hash
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from codediff.hasher import HashMatcher, SemanticHasher
from codediff.parser import ClassInfo, CodeParser, FileInfo, FunctionInfo


class ChangeType(str, Enum):
    """Type of change detected."""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    RENAMED = "renamed"
    UNCHANGED = "unchanged"


@dataclass
class FunctionChange:
    """Information about a function change."""

    name: str
    change_type: ChangeType
    source_file: Optional[str] = None
    target_file: Optional[str] = None
    source_line: Optional[int] = None
    target_line: Optional[int] = None
    old_signature: Optional[str] = None
    new_signature: Optional[str] = None
    similarity: Optional[float] = None


@dataclass
class ClassChange:
    """Information about a class change."""

    name: str
    change_type: ChangeType
    source_file: Optional[str] = None
    target_file: Optional[str] = None
    source_line: Optional[int] = None
    target_line: Optional[int] = None
    method_changes: Dict[str, FunctionChange] = field(default_factory=dict)


@dataclass
class FileChange:
    """Information about a file change."""

    path: str
    change_type: ChangeType
    class_changes: Dict[str, ClassChange] = field(default_factory=dict)
    function_changes: Dict[str, FunctionChange] = field(default_factory=dict)


class CodeComparator:
    """Compare two code trees and identify differences."""

    def __init__(self):
        """Initialize the comparator."""
        self.parser = CodeParser()

    def compare_directories(
        self,
        source_dir: Union[str, Path],
        target_dir: Union[str, Path],
    ) -> Dict[str, FileChange]:
        """Compare two directories recursively.

        Args:
            source_dir: Source directory to compare
            target_dir: Target directory to compare

        Returns:
            Dictionary mapping file paths to FileChange objects
        """
        source_dir = Path(source_dir)
        target_dir = Path(target_dir)

        # Find all Python files
        source_files = {f.relative_to(source_dir): f for f in source_dir.rglob("*.py")}
        target_files = {f.relative_to(target_dir): f for f in target_dir.rglob("*.py")}

        file_changes: Dict[str, FileChange] = {}

        # Compare files that exist in both directories
        common_paths = set(source_files.keys()) & set(target_files.keys())
        for rel_path in sorted(common_paths):
            changes = self.compare_files(source_files[rel_path], target_files[rel_path])
            if changes.class_changes or changes.function_changes:
                file_changes[str(rel_path)] = changes

        # Track added files
        added_paths = set(target_files.keys()) - set(source_files.keys())
        for rel_path in sorted(added_paths):
            file_changes[str(rel_path)] = FileChange(
                path=str(rel_path),
                change_type=ChangeType.ADDED,
            )

        # Track removed files
        removed_paths = set(source_files.keys()) - set(target_files.keys())
        for rel_path in sorted(removed_paths):
            file_changes[str(rel_path)] = FileChange(
                path=str(rel_path),
                change_type=ChangeType.REMOVED,
            )

        return file_changes

    def compare_files(self, source_file: Union[str, Path], target_file: Union[str, Path]) -> FileChange:
        """Compare two Python files.

        Args:
            source_file: Source file to compare
            target_file: Target file to compare

        Returns:
            FileChange object describing differences
        """
        source_file = Path(source_file)
        target_file = Path(target_file)

        # Parse both files
        try:
            source_info = self.parser.parse_file(source_file)
        except SyntaxError:
            # If source doesn't parse, skip it
            return FileChange(path=str(target_file), change_type=ChangeType.ADDED)

        try:
            target_info = self.parser.parse_file(target_file)
        except SyntaxError:
            # If target doesn't parse, mark as removed
            return FileChange(path=str(source_file), change_type=ChangeType.REMOVED)

        # Compare classes and functions
        class_changes = self._compare_classes(source_info, target_info)
        function_changes = self._compare_functions(source_info, target_info)

        change_type = ChangeType.UNCHANGED
        if class_changes or function_changes:
            change_type = ChangeType.MODIFIED

        return FileChange(
            path=str(target_file),
            change_type=change_type,
            class_changes=class_changes,
            function_changes=function_changes,
        )

    def _compare_classes(
        self,
        source_info: FileInfo,
        target_info: FileInfo,
    ) -> Dict[str, ClassChange]:
        """Compare classes between two files (pass 1: by name)."""
        changes: Dict[str, ClassChange] = {}

        source_classes = source_info.classes
        target_classes = target_info.classes
        source_names = set(source_classes.keys())
        target_names = set(target_classes.keys())

        # Pass 1: Match by name
        matched: Set[str] = set()

        for name in source_names & target_names:
            source_class = source_classes[name]
            target_class = target_classes[name]

            method_changes = self._compare_methods(source_class, target_class)

            if method_changes:
                changes[name] = ClassChange(
                    name=name,
                    change_type=ChangeType.MODIFIED,
                    source_file=str(source_info.path),
                    target_file=str(target_info.path),
                    source_line=source_class.line_number,
                    target_line=target_class.line_number,
                    method_changes=method_changes,
                )

            matched.add(name)

        # Pass 2: Try to match unmatched by semantic hash
        unmatched_source = {k: v for k, v in source_classes.items() if k not in matched}
        unmatched_target = {k: v for k, v in target_classes.items() if k not in matched}

        rename_matches = HashMatcher.find_matches(unmatched_source, unmatched_target)
        for old_name, (new_name, similarity) in rename_matches.items():
            source_class = unmatched_source[old_name]
            target_class = unmatched_target[new_name]

            changes[old_name] = ClassChange(
                name=f"{old_name} -> {new_name}",
                change_type=ChangeType.RENAMED,
                source_file=str(source_info.path),
                target_file=str(target_info.path),
                source_line=source_class.line_number,
                target_line=target_class.line_number,
                method_changes=self._compare_methods(source_class, target_class),
            )

            matched.add(old_name)

        # Removed classes
        for name in source_names - matched:
            source_class = source_classes[name]
            changes[name] = ClassChange(
                name=name,
                change_type=ChangeType.REMOVED,
                source_file=str(source_info.path),
                source_line=source_class.line_number,
            )

        # Added classes
        for name in target_names - matched:
            target_class = target_classes[name]
            changes[name] = ClassChange(
                name=name,
                change_type=ChangeType.ADDED,
                target_file=str(target_info.path),
                target_line=target_class.line_number,
            )

        return changes

    def _compare_functions(
        self,
        source_info: FileInfo,
        target_info: FileInfo,
    ) -> Dict[str, FunctionChange]:
        """Compare top-level functions between two files."""
        changes: Dict[str, FunctionChange] = {}

        source_funcs = source_info.functions
        target_funcs = target_info.functions
        source_names = set(source_funcs.keys())
        target_names = set(target_funcs.keys())

        # Pass 1: Match by name
        matched: Set[str] = set()

        for name in source_names & target_names:
            source_func = source_funcs[name]
            target_func = target_funcs[name]

            if self._functions_different(source_func, target_func):
                changes[name] = FunctionChange(
                    name=name,
                    change_type=ChangeType.MODIFIED,
                    source_file=str(source_info.path),
                    target_file=str(target_info.path),
                    source_line=source_func.line_number,
                    target_line=target_func.line_number,
                    old_signature=source_func.signature,
                    new_signature=target_func.signature,
                )

            matched.add(name)

        # Pass 2: Try to match by hash
        unmatched_source = {k: v for k, v in source_funcs.items() if k not in matched}
        unmatched_target = {k: v for k, v in target_funcs.items() if k not in matched}

        rename_matches = HashMatcher.find_matches(unmatched_source, unmatched_target)
        for old_name, (new_name, similarity) in rename_matches.items():
            source_func = unmatched_source[old_name]
            target_func = unmatched_target[new_name]

            changes[old_name] = FunctionChange(
                name=f"{old_name} -> {new_name}",
                change_type=ChangeType.RENAMED,
                source_file=str(source_info.path),
                target_file=str(target_info.path),
                source_line=source_func.line_number,
                target_line=target_func.line_number,
                old_signature=source_func.signature,
                new_signature=target_func.signature,
                similarity=similarity,
            )

            matched.add(old_name)

        # Removed functions
        for name in source_names - matched:
            source_func = source_funcs[name]
            changes[name] = FunctionChange(
                name=name,
                change_type=ChangeType.REMOVED,
                source_file=str(source_info.path),
                source_line=source_func.line_number,
                old_signature=source_func.signature,
            )

        # Added functions
        for name in target_names - matched:
            target_func = target_funcs[name]
            changes[name] = FunctionChange(
                name=name,
                change_type=ChangeType.ADDED,
                target_file=str(target_info.path),
                target_line=target_func.line_number,
                new_signature=target_func.signature,
            )

        return changes

    def _compare_methods(
        self,
        source_class: ClassInfo,
        target_class: ClassInfo,
    ) -> Dict[str, FunctionChange]:
        """Compare methods within a class."""
        changes: Dict[str, FunctionChange] = {}

        source_methods = source_class.methods
        target_methods = target_class.methods

        for name in set(source_methods.keys()) & set(target_methods.keys()):
            source_method = source_methods[name]
            target_method = target_methods[name]

            if self._functions_different(source_method, target_method):
                changes[name] = FunctionChange(
                    name=name,
                    change_type=ChangeType.MODIFIED,
                    old_signature=source_method.signature,
                    new_signature=target_method.signature,
                )

        # Added/removed methods
        for name in set(source_methods.keys()) - set(target_methods.keys()):
            changes[name] = FunctionChange(
                name=name,
                change_type=ChangeType.REMOVED,
                old_signature=source_methods[name].signature,
            )

        for name in set(target_methods.keys()) - set(source_methods.keys()):
            changes[name] = FunctionChange(
                name=name,
                change_type=ChangeType.ADDED,
                new_signature=target_methods[name].signature,
            )

        return changes

    @staticmethod
    def _functions_different(func1: FunctionInfo, func2: FunctionInfo) -> bool:
        """Check if two functions are semantically different."""
        # Compare semantic hashes
        hash1 = SemanticHasher.hash_function(func1)
        hash2 = SemanticHasher.hash_function(func2)

        return hash1 != hash2
