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

### Command-Line Options

```
positional arguments:
  source                Source directory or file to compare
  target                Target directory or file to compare

optional arguments:
  -o, --output OUTPUT   Output file path (default: stdout)
  -f, --format {json,markdown}
                        Output format (default: json)

Change Type Filters:
  --missing-only        Show only removed items (missing in target)
  --added-only          Show only added items (new in target)
  --modified-only       Show only modified items
  --renamed-only        Show only renamed items

Granular Filtering:
  --function FUNC       Filter to specific function(s) - can repeat
  --class CLASS         Filter to specific class(es) - can repeat
  --pattern PATTERN     Filter by name pattern (regex)

Detailed Output:
  --show-diff           Show detailed code differences
  --context-lines N     Lines of context for diffs (default: 3)

Feature Parity:
  --normalize-imports   Normalize imports for feature parity checking:
                        Replace mmcv→visdet.cv and mmengine→visdet.engine
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

### Advanced Filtering - Find Specific Changes

**Filter to specific function changes:**

```bash
# Show only changes to test_swin_transformer and related tests
codediff tests/ visdet/tests/ \
  --function test_swin_transformer \
  --modified-only \
  --show-diff

# Show multiple functions (can repeat --function)
codediff tests/ visdet/tests/ \
  --function test_forward \
  --function test_backward \
  --function test_loss
```

**Filter to specific class tests:**

```bash
# Show only changes to TestMaskRCNN class
codediff tests/test_models/ visdet/tests/test_models/ \
  --class TestMaskRCNN \
  --modified-only
```

**Filter by name pattern (regex):**

```bash
# Show only changes to test_* functions that contain "head"
codediff tests/test_models/ visdet/tests/test_models/ \
  --pattern "test_.*head.*" \
  --modified-only

# Show all test functions that start with test_swin
codediff tests/test_models/test_backbones/ \
         visdet/tests/test_models/test_backbones/ \
  --pattern "^test_swin" \
  --show-diff
```

**Detailed diff output:**

```bash
# Show code-level diffs for all modified tests
codediff tests/ visdet/tests/ \
  --modified-only \
  --show-diff \
  --context-lines 5 \
  --format markdown \
  --output detailed_changes.md

# Focus on specific test file with full diff
codediff tests/test_models/test_backbones/test_swin.py \
         visdet/tests/test_models/test_backbones/test_swin.py \
  --show-diff \
  --context-lines 3
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

## Feature Parity Checking

One of the most powerful use cases of `codediff` is validating **feature parity** between refactored codebases. When migrating code from one framework/library to another, import paths typically change even though the implementation remains the same. For example:

```python
# Original MMDetection
from mmcv.ops import nms
from mmengine.config import Config

# Refactored VisDet
from visdet.cv.ops import nms
from visdet.engine.config import Config
```

Without special handling, these would show as "modified" even though the logic is identical. The `--normalize-imports` flag solves this:

```bash
# Compare implementations while normalizing library imports
codediff archive/mmdet/models/ visdet/models/ \
  --normalize-imports \
  --modified-only \
  --show-diff \
  --format markdown
```

**What `--normalize-imports` does:**
- Replaces all `mmcv` imports with `visdet.cv`
- Replaces all `mmengine` imports with `visdet.engine`
- Allows you to see **only real semantic differences**, not import path changes
- Filters out false positives in your diff analysis

### Feature Parity Validation Workflow

1. **Find truly different implementations:**
```bash
codediff archive/mmdet/models/ visdet/models/ \
  --normalize-imports \
  --modified-only \
  --output real_changes.json
```

2. **Verify you have all required code (not just different imports):**
```bash
codediff archive/mmdet/models/ visdet/models/ \
  --normalize-imports \
  --missing-only \
  --format markdown \
  --output missing_code.md
```

3. **Check for renamed/moved functions:**
```bash
codediff archive/mmdet/models/ visdet/models/ \
  --normalize-imports \
  --renamed-only \
  --show-diff \
  --output renames.md
```

4. **Analyze specific module changes:**
```bash
codediff archive/mmdet/models/backbones/ visdet/models/backbones/ \
  --normalize-imports \
  --pattern "ResNet" \
  --modified-only \
  --show-diff \
  --context-lines 10
```

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
