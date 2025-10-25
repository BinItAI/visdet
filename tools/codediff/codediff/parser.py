"""
AST parsing and code extraction module.

Parses Python files and extracts classes, functions, and their signatures.
Handles normalization for semantic comparison.
"""

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class FunctionInfo:
    """Information about a function or method."""

    name: str
    line_number: int
    end_line_number: Optional[int]
    signature: str
    docstring: Optional[str]
    source_code: str
    is_method: bool = False
    decorators: List[str] = None

    def __post_init__(self):
        if self.decorators is None:
            self.decorators = []


@dataclass
class ClassInfo:
    """Information about a class."""

    name: str
    line_number: int
    end_line_number: Optional[int]
    docstring: Optional[str]
    bases: List[str]
    methods: Dict[str, FunctionInfo]

    @property
    def source_code(self) -> str:
        """Return concatenated source of all methods."""
        return "\n".join(m.source_code for m in self.methods.values())


@dataclass
class FileInfo:
    """Information about a Python file."""

    path: str
    classes: Dict[str, ClassInfo]
    functions: Dict[str, FunctionInfo]
    source_code: str

    @property
    def line_count(self) -> int:
        """Get total line count of file."""
        return len(self.source_code.split("\n"))


class CodeParser:
    """Parser for Python source code using AST."""

    def __init__(self):
        """Initialize the code parser."""
        self.source_lines: List[str] = []
        self.tree: Optional[ast.AST] = None

    def parse_file(self, file_path: Union[str, Path]) -> FileInfo:
        """Parse a Python file and extract code structure.

        Args:
            file_path: Path to Python file to parse

        Returns:
            FileInfo object containing parsed information
        """
        file_path = Path(file_path)
        source_code = file_path.read_text(encoding="utf-8")
        self.source_lines = source_code.split("\n")

        try:
            self.tree = ast.parse(source_code, filename=str(file_path))
        except SyntaxError as e:
            raise SyntaxError(f"Failed to parse {file_path}: {e}")

        classes: Dict[str, ClassInfo] = {}
        functions: Dict[str, FunctionInfo] = {}

        # Extract top-level classes and functions (only from module body, not nested)
        for node in self.tree.body:
            if isinstance(node, ast.ClassDef):
                classes[node.name] = self._extract_class(node)
            elif isinstance(node, ast.FunctionDef):
                functions[node.name] = self._extract_function(node)

        return FileInfo(
            path=str(file_path),
            classes=classes,
            functions=functions,
            source_code=source_code,
        )

    def _extract_class(self, node: ast.ClassDef) -> ClassInfo:
        """Extract class information from AST node."""
        methods: Dict[str, FunctionInfo] = {}

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods[item.name] = self._extract_function(item, is_method=True)

        bases = [self._get_name(base) for base in node.bases]
        docstring = ast.get_docstring(node)

        return ClassInfo(
            name=node.name,
            line_number=node.lineno,
            end_line_number=node.end_lineno,
            docstring=docstring,
            bases=bases,
            methods=methods,
        )

    def _extract_function(
        self,
        node: ast.FunctionDef,
        is_method: bool = False,
    ) -> FunctionInfo:
        """Extract function information from AST node."""
        signature = self._get_function_signature(node)
        docstring = ast.get_docstring(node)
        source_code = self._get_source_lines(node)
        decorators = [self._get_name(d) for d in node.decorator_list]

        return FunctionInfo(
            name=node.name,
            line_number=node.lineno,
            end_line_number=node.end_lineno,
            signature=signature,
            docstring=docstring,
            source_code=source_code,
            is_method=is_method,
            decorators=decorators,
        )

    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Get function signature from AST node."""
        args = node.args

        # Build parameter list with type annotations
        params = []

        # Regular arguments
        for arg in args.args:
            param = arg.arg
            if arg.annotation:
                param = f"{param}: {ast.unparse(arg.annotation)}"
            params.append(param)

        # *args
        if args.vararg:
            param = f"*{args.vararg.arg}"
            if args.vararg.annotation:
                param = f"*{args.vararg.arg}: {ast.unparse(args.vararg.annotation)}"
            params.append(param)

        # Keyword-only arguments
        for arg in args.kwonlyargs:
            param = arg.arg
            if arg.annotation:
                param = f"{param}: {ast.unparse(arg.annotation)}"
            params.append(param)

        # **kwargs
        if args.kwarg:
            param = f"**{args.kwarg.arg}"
            if args.kwarg.annotation:
                param = f"**{args.kwarg.arg}: {ast.unparse(args.kwarg.annotation)}"
            params.append(param)

        # Return type annotation if present
        return_type = ""
        if node.returns:
            return_type = f" -> {ast.unparse(node.returns)}"

        return f"def {node.name}({', '.join(params)}){return_type}:"

    def _get_source_lines(self, node: ast.AST) -> str:
        """Get source code lines for an AST node."""
        if node.lineno is None or node.end_lineno is None:
            return ""

        # ast uses 1-based indexing
        start_idx = node.lineno - 1
        end_idx = node.end_lineno

        lines = self.source_lines[start_idx:end_idx]
        return "\n".join(lines)

    def _get_name(self, node: ast.expr) -> str:
        """Get name from various AST node types."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._get_name(node.value)}[...]"
        else:
            try:
                return ast.unparse(node)
            except Exception:
                return "<unknown>"

    def _is_method(self, node: ast.FunctionDef) -> bool:
        """Check if a function is a method (inside a class)."""
        # This is a simple heuristic - we only call this on top-level items
        return False


class NormalizationVisitor(ast.NodeTransformer):
    """AST visitor that normalizes code for comparison.

    Removes/standardizes elements that don't affect logic:
    - Docstrings
    - Comments (via source code normalization)
    - Whitespace
    - Variable names (optionally)
    """

    def visit_Expr(self, node: ast.Expr) -> Optional[ast.AST]:
        """Remove docstring expressions from the AST."""
        # Check if this is a docstring (string constant at statement level)
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    @staticmethod
    def normalize_source(source_code: str) -> str:
        """Normalize source code for semantic comparison.

        Removes comments and extra whitespace while preserving structure.
        """
        lines = source_code.split("\n")
        normalized_lines = []

        for line in lines:
            # Remove inline comments
            if "#" in line:
                code_part = line[: line.index("#")]
            else:
                code_part = line

            # Strip trailing whitespace but preserve leading for indentation
            code_part = code_part.rstrip()

            # Skip blank lines
            if code_part.strip():
                normalized_lines.append(code_part)

        return "\n".join(normalized_lines)
