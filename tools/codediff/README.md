# codediff - Semantic Code Comparison Tool

Compare Python code **semantically** - ignoring whitespace, formatting, and comments while detecting code that has been moved, renamed, or modified.

## Overview

`codediff` is a Python tool designed to help developers and LLMs identify where implementations have diverged between two codebases. Unlike traditional text-based diff tools, it:

- **Ignores whitespace & formatting**: Only cares about semantic changes
- **Detects renames**: Uses semantic hashing to find functions/classes that have been renamed
- **Multi-level analysis**: Compares at file, class, and function levels
- **LLM-friendly output**: Generates structured JSON and markdown reports

## Installation

```bash
cd tools/codediff
pip install -e .
```

Or with uv:

```bash
uv pip install -e tools/codediff/
```

## Usage

### Basic Comparison

Compare two directories:

```bash
codediff archive/mmdet/tests/ tests/ --output report.json
```

Compare two files:

```bash
codediff archive/mmdet/file.py visdet/file.py
```

### Output Formats

**JSON output** (default - optimal for LLMs):

```bash
codediff source/ target/ --format json --output report.json
```

**Markdown output** (human-readable):

```bash
codediff source/ target/ --format markdown --output report.md
```

### Filtering Results

Show only missing tests:

```bash
codediff tests/ visdet/tests/ --missing-only --output missing.json
```

Show only added items:

```bash
codediff tests/ visdet/tests/ --added-only
```

Show only renamed items:

```bash
codediff tests/ visdet/tests/ --renamed-only
```

Show only modified items:

```bash
codediff tests/ visdet/tests/ --modified-only
```

## Output Format

### JSON Structure

```json
{
  "stats": {
    "total_files": 5,
    "files": {
      "added": 2,
      "removed": 1,
      "modified": 2
    },
    "total_classes": 8,
    "classes": {
      "added": 2,
      "removed": 1,
      "modified": 5
    },
    "total_functions": 24,
    "functions": {
      "added": 4,
      "removed": 2,
      "modified": 18
    }
  },
  "files": {
    "path/to/file.py": {
      "status": "modified",
      "classes": {
        "MyClass": {
          "status": "modified",
          "line": 42,
          "methods": {
            "my_method": {
              "status": "modified",
              "old_signature": "def my_method(self, x: int) -> str:",
              "new_signature": "def my_method(self, x: int, y: int) -> str:"
            }
          }
        }
      },
      "functions": {
        "helper_func": {
          "status": "renamed",
          "line": 100,
          "old_signature": "def helper_func(x):",
          "new_signature": "def helper_func_v2(x):"
        }
      }
    }
  }
}
```

## How It Works

### Two-Pass Comparison

1. **Pass 1 (Name Matching)**: Match classes/functions by name
2. **Pass 2 (Semantic Hashing)**: For unmatched items, generate semantic hashes and find similar implementations that may have been renamed

### Semantic Hashing

The tool normalizes code and generates semantic fingerprints:

- Removes comments and docstrings
- Normalizes whitespace
- Converts code to canonical AST form
- Generates SHA256 hash for comparison

Items with hashes within 15% similarity are considered potential renames.

## Use Cases

### Test Coverage Validation

Ensure all tests are migrated when refactoring:

```bash
codediff archive/tests/ new_tests/ --missing-only
```

### Implementation Drift Detection

Identify where implementations have changed between branches:

```bash
codediff main/src/ feature/src/ --modified-only
```

### API Migration Tracking

Find what APIs need updating during major refactors:

```bash
codediff old_api.py new_api.py --format markdown
```

## Architecture

```
codediff/
├── parser.py      # AST parsing and code extraction
├── hasher.py      # Semantic hash generation and matching
├── comparator.py  # Two-pass comparison logic
├── report.py      # JSON/markdown report generation
└── cli.py         # Command-line interface
```

## Limitations

- Only analyzes Python files
- Ignores:
  - Comments (intentional)
  - Docstring changes
  - Whitespace-only changes
- Performance: O(n*m) for rename detection on large codebases

## Testing

Run the test suite:

```bash
pytest tests/
```

With coverage:

```bash
pytest --cov=codediff tests/
```

## Contributing

The tool is designed to be extended. Key extension points:

- **Custom normalizers**: Add more aggressive normalization
- **Alternative hash functions**: Use different hash algorithms
- **Output formats**: Add YAML, XML, or other formats
- **Language support**: Extend parser for other languages

## License

Apache License 2.0 (same as parent project)
