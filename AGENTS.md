# Repository Guidelines

## Documentation Organization

This repository maintains a clean root directory with core documentation files:

### Root Level (Keep Clean)
- **README.md** - Main project documentation
- **AGENTS.md** - This file, repository-wide guidelines (do not move)
- Configuration files (pyproject.toml, etc.)

### Experimental & Planning Materials
**Policy**: All other capitalised markdown files (`*.md`) should be placed in the `scratch_pads/` directory.

This includes:
- Planning and refactoring documents
- Experimental documentation
- Reference materials
- Archived guides

Current files in `scratch_pads/`:
- DOCS_README.md
- README_zh-CN.md
- REFACTORING_PLAN.md
- SKYLOS.md

## Why This Structure?

- Keeps the root directory clean and focused
- Separates core documentation from experimental materials
- Makes it clear which documents are actively maintained
- Provides a designated space for work-in-progress materials

---

## Code Comparison Tool: Using Import Normalization

The repository includes **codediff**, a semantic code comparison tool that helps identify implementation differences while accounting for library migration. This is particularly useful when comparing code that uses different import paths but equivalent functionality.

### Installation

```bash
# Install codediff in development mode
uv pip install -e tools/codediff/
```

### Feature Parity Checking with Import Normalization

When comparing implementations across library migrations (e.g., MMDetection → VisDet), import paths change even though functionality remains identical:

```python
# Original MMDetection
from mmcv.ops import nms
from mmengine.config import Config

# Refactored VisDet
from visdet.cv.ops import nms
from visdet.engine.config import Config
```

The `--normalize-imports` flag handles this automatically, replacing:
- `mmcv` → `visdet.cv`
- `mmengine` → `visdet.engine`

This allows you to see **only real semantic differences**, not import path changes.

### Basic Import Normalization Usage

**Compare implementations while ignoring import path differences:**

```bash
# Find truly modified code (ignoring import changes)
codediff archive/mmdet/models/ visdet/models/ \
  --normalize-imports \
  --modified-only \
  --show-diff \
  --format markdown
```

### Feature Parity Validation Workflow

A complete feature parity check involves four steps:

**Step 1: Find implementations that are truly different**

```bash
codediff archive/mmdet/models/ visdet/models/ \
  --normalize-imports \
  --modified-only \
  --output real_changes.json
```

**Step 2: Verify all required code is present (not just different imports)**

```bash
codediff archive/mmdet/models/ visdet/models/ \
  --normalize-imports \
  --missing-only \
  --format markdown \
  --output missing_code.md
```

**Step 3: Check for renamed or moved functions**

```bash
codediff archive/mmdet/models/ visdet/models/ \
  --normalize-imports \
  --renamed-only \
  --show-diff \
  --output renames.md
```

**Step 4: Analyze specific module changes**

```bash
codediff archive/mmdet/models/backbones/ visdet/models/backbones/ \
  --normalize-imports \
  --pattern "ResNet" \
  --modified-only \
  --show-diff \
  --context-lines 10
```

### Advanced Filtering with Import Normalization

**Filter to specific functions while normalizing imports:**

```bash
# Show changes to specific function while ignoring import differences
codediff archive/mmdet/ visdet/ \
  --normalize-imports \
  --function forward_train \
  --modified-only \
  --show-diff
```

**Filter by pattern with detailed diff:**

```bash
# Show all detector-related changes with normalized imports
codediff archive/mmdet/models/detectors/ visdet/models/detectors/ \
  --normalize-imports \
  --pattern ".*Detector" \
  --modified-only \
  --show-diff \
  --context-lines 10 \
  --format markdown \
  --output detector_changes.md
```

**Find renamed items across codebase:**

```bash
# Detect functions that were moved/renamed with normalized imports
codediff archive/mmdet/ visdet/ \
  --normalize-imports \
  --renamed-only \
  --show-diff \
  --output renamed_items.json
```

### Real-World Example: Validating Backbone Implementations

Compare backbone implementations between archive and visdet:

```bash
# Full analysis with all features
codediff archive/mmdet/models/backbones/ visdet/models/backbones/ \
  --normalize-imports \
  --show-diff \
  --context-lines 5 \
  --format markdown \
  --output backbone_analysis.md
```

This command:
1. Normalizes all `mmcv` and `mmengine` imports
2. Shows detailed line-by-line diffs
3. Highlights renamed/moved code
4. Exports to human-readable markdown

### Command-Line Reference

**Import Normalization Option:**
```
--normalize-imports     Normalize imports for feature parity checking:
                        Replace mmcv→visdet.cv and mmengine→visdet.engine
```

**Change Type Filters:**
```
--missing-only          Show only removed items (missing in target)
--added-only            Show only added items (new in target)
--modified-only         Show only modified items
--renamed-only          Show only renamed items
```

**Granular Filtering:**
```
--function FUNC         Filter to specific function(s) - can repeat
--class CLASS           Filter to specific class(es) - can repeat
--pattern PATTERN       Filter by name pattern (regex)
```

**Output Options:**
```
-f, --format {json,markdown}  Output format (default: json)
-o, --output OUTPUT           Output file path (default: stdout)
--show-diff                   Show detailed code differences
--context-lines N             Lines of context for diffs (default: 3)
```

### Output Formats

**JSON (optimal for processing):**
```bash
codediff source/ target/ --normalize-imports --format json --output report.json
```

**Markdown (human-readable):**
```bash
codediff source/ target/ --normalize-imports --format markdown --output report.md
```

### Tips for Effective Use

1. **Always use `--normalize-imports` when comparing MMDetection and VisDet code** to avoid false positives from import path changes

2. **Combine filters for targeted analysis:**
   ```bash
   codediff archive/ visdet/ \
     --normalize-imports \
     --modified-only \
     --pattern "test_" \
     --show-diff
   ```

3. **Export results to files for documentation:**
   ```bash
   codediff archive/mmdet/ visdet/ \
     --normalize-imports \
     --format markdown \
     --output implementation_analysis.md
   ```

4. **Use for regression testing** - Verify that refactoring didn't inadvertently change logic

---

For project-specific guidelines, see `visdet/AGENTS.md` (model-focused documentation)
