"""Tests for the parser module."""

import tempfile
from pathlib import Path

import pytest
from codediff.parser import ClassInfo, CodeParser, FunctionInfo


@pytest.fixture
def temp_python_file():
    """Create a temporary Python file for testing."""
    code = '''"""Module docstring."""

def simple_function(x: int) -> str:
    """A simple function."""
    return str(x)

class MyClass:
    """A test class."""

    def __init__(self, name: str):
        """Initialize the class."""
        self.name = name

    def get_name(self) -> str:
        """Get the name."""
        return self.name

    def set_name(self, name: str) -> None:
        """Set the name."""
        self.name = name

def another_function(a, b):
    # This is a comment
    return a + b
'''
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        yield f.name

    Path(f.name).unlink()


class TestCodeParser:
    """Tests for CodeParser class."""

    def test_parse_file_success(self, temp_python_file):
        """Test parsing a valid Python file."""
        parser = CodeParser()
        file_info = parser.parse_file(temp_python_file)

        assert file_info.path == temp_python_file
        assert len(file_info.functions) == 2
        assert len(file_info.classes) == 1

    def test_extract_functions(self, temp_python_file):
        """Test that functions are extracted correctly."""
        parser = CodeParser()
        file_info = parser.parse_file(temp_python_file)

        assert "simple_function" in file_info.functions
        assert "another_function" in file_info.functions

        simple_func = file_info.functions["simple_function"]
        assert simple_func.name == "simple_function"
        assert simple_func.signature == "def simple_function(x: int) -> str:"
        assert "A simple function" in simple_func.docstring

    def test_extract_classes(self, temp_python_file):
        """Test that classes are extracted correctly."""
        parser = CodeParser()
        file_info = parser.parse_file(temp_python_file)

        assert "MyClass" in file_info.classes

        my_class = file_info.classes["MyClass"]
        assert my_class.name == "MyClass"
        assert len(my_class.methods) == 3
        assert "__init__" in my_class.methods
        assert "get_name" in my_class.methods
        assert "set_name" in my_class.methods

    def test_extract_methods(self, temp_python_file):
        """Test that methods are extracted from classes."""
        parser = CodeParser()
        file_info = parser.parse_file(temp_python_file)

        my_class = file_info.classes["MyClass"]
        init_method = my_class.methods["__init__"]

        assert init_method.name == "__init__"
        assert init_method.is_method is True
        assert "Initialize the class" in init_method.docstring

    def test_syntax_error_handling(self):
        """Test that syntax errors are handled gracefully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("this is not valid python )(")
            f.flush()

            parser = CodeParser()
            with pytest.raises(SyntaxError):
                parser.parse_file(f.name)

        Path(f.name).unlink()

    def test_source_code_extraction(self, temp_python_file):
        """Test that source code is correctly extracted."""
        parser = CodeParser()
        file_info = parser.parse_file(temp_python_file)

        # Check that source code is extracted
        assert "simple_function" in file_info.source_code
        assert "MyClass" in file_info.source_code

    def test_line_numbers(self, temp_python_file):
        """Test that line numbers are correctly tracked."""
        parser = CodeParser()
        file_info = parser.parse_file(temp_python_file)

        simple_func = file_info.functions["simple_function"]
        assert simple_func.line_number > 0
        assert simple_func.end_line_number is not None
        assert simple_func.end_line_number >= simple_func.line_number


class TestNormalization:
    """Tests for code normalization."""

    def test_normalize_source_removes_comments(self):
        """Test that comments are removed during normalization."""
        from codediff.parser import NormalizationVisitor

        code = """def foo():
    x = 1  # inline comment
    # full line comment
    y = 2
"""
        normalized = NormalizationVisitor.normalize_source(code)

        assert "inline comment" not in normalized
        assert "full line comment" not in normalized
        assert "x = 1" in normalized
        assert "y = 2" in normalized

    def test_normalize_source_preserves_indentation(self):
        """Test that indentation structure is preserved."""
        from codediff.parser import NormalizationVisitor

        code = """def foo():
    if True:
        return 1
    return 2
"""
        normalized = NormalizationVisitor.normalize_source(code)

        # Check that indentation is preserved
        lines = normalized.split("\n")
        assert any(line.startswith("    if") for line in lines)
