# Feature Parity Validation Guide

## Overview

When refactoring code to use different libraries (e.g., from MMDetection's `mmcv`/`mmengine` to VisDet's `visdet.cv`/`visdet.engine`), the implementation logic stays the same but import paths change. This guide explains how to use `codediff` with the `--normalize-imports` flag to validate feature parity and identify **only real semantic differences**.

## The Problem

Without import normalization, comparing original and refactored code produces false positives:

```python
# Original MMDetection
from mmcv.ops import nms
from mmengine.config import Config

def train_detector():
    config = Config.fromfile('config.py')
    nms_results = nms(scores)
    # ... rest of implementation
```

```python
# Refactored VisDet
from visdet.cv.ops import nms
from visdet.engine.config import Config

def train_detector():
    config = Config.fromfile('config.py')
    nms_results = nms(scores)
    # ... rest of implementation (identical logic)
```

**Without normalization:** Both functions show as "modified" even though the logic is identical.

**With normalization:** They show as "unchanged" because the import path is normalized away.

## The Solution

Use `--normalize-imports` to automatically replace library paths:

- `mmcv` → `visdet.cv`
- `mmengine` → `visdet.engine`

This allows you to focus on **real semantic differences**, not import mechanics.

## Feature Parity Validation Workflow

### Step 1: Identify Truly Different Implementations

Find code that has actual semantic changes (not just import changes):

```bash
# Find all modified implementations
codediff archive/mmdet/models/ visdet/models/ \
  --normalize-imports \
  --modified-only \
  --format markdown \
  --output 1_real_changes.md

# Or just count them
codediff archive/mmdet/models/ visdet/models/ \
  --normalize-imports \
  --modified-only \
  --format json \
  --output 1_real_changes.json
```

This shows you which implementations have **actual code changes** beyond just import paths.

### Step 2: Verify Complete Code Migration

Check that all required code exists in the refactored version (no missing implementations):

```bash
# Find anything missing from the refactored version
codediff archive/mmdet/models/ visdet/models/ \
  --normalize-imports \
  --missing-only \
  --format markdown \
  --output 2_missing_code.md
```

Any items in this report are functions/classes that weren't ported yet.

### Step 3: Identify Renamed/Refactored Functions

Find code that exists but has been renamed or restructured:

```bash
# Show all renames detected via semantic hashing
codediff archive/mmdet/models/ visdet/models/ \
  --normalize-imports \
  --renamed-only \
  --show-diff \
  --format markdown \
  --output 3_renames.md
```

This shows functions that were moved to different files or renamed but have the same logic.

### Step 4: Deep Dive into Specific Modules

Analyze changes in specific parts of the codebase:

```bash
# Check all backbone implementations for changes
codediff archive/mmdet/models/backbones/ visdet/models/backbones/ \
  --normalize-imports \
  --modified-only \
  --show-diff \
  --context-lines 5 \
  --format markdown \
  --output 4_backbone_changes.md

# Check specific detector implementation
codediff archive/mmdet/models/detectors/faster_rcnn.py \
  visdet/models/detectors/faster_rcnn.py \
  --normalize-imports \
  --show-diff \
  --context-lines 10 \
  --format markdown
```

### Step 5: Pattern-Based Analysis

Find changes in specific function/class families:

```bash
# Check all loss functions for changes
codediff archive/mmdet/models/losses/ visdet/models/losses/ \
  --normalize-imports \
  --pattern ".*Loss.*" \
  --modified-only \
  --show-diff

# Check all NMS-related functions
codediff archive/mmdet/ visdet/ \
  --normalize-imports \
  --pattern ".*nms.*" \
  --show-diff

# Check all anchor-related code
codediff archive/mmdet/models/roi_heads/ visdet/models/roi_heads/ \
  --normalize-imports \
  --pattern "anchor" \
  --modified-only
```

## Example Validation Report

Here's what a feature parity report should look like:

```markdown
# Feature Parity Validation Report
Generated: $(date)

## Executive Summary
- **Total Files Analyzed:** 157
- **Missing Files:** 0 ✅
- **Unintended Changes:** 2 ⚠️
- **Expected Changes:** 3 (optimizations)
- **Renamed Functions:** 1

## Files Missing (Should be 0 for full parity)
None - All original implementations are present!

## Unexpected Modifications (Review these)
1. `models/detectors/faster_rcnn.py::post_process()`
   - Changed: Memory optimization (preallocation)
   - Status: Deliberate improvement ✅

2. `models/backbones/resnet.py::forward()`
   - Changed: Added Flash Attention support
   - Status: Feature addition ✅

## Intentional Changes (Expected)
1. `models/necks/fpn.py::_init_weights()`
   - Changed: Updated for visdet module structure

2. `models/heads/bbox_head.py::loss()`
   - Changed: Refactored for clarity

3. Various files: All imports normalized
   - mmcv → visdet.cv
   - mmengine → visdet.engine

## Renamed/Moved Code
1. `models/utils/misc.py::build_norm_layer()` → `models/utils/norm.py::build_norm_layer()`
   - Status: ✅ Code identical, moved to dedicated module

## Validation Status
✅ **FEATURE PARITY ACHIEVED**
- All required code present
- All changes are deliberate/documented
- Ready for production use
```

## Integration with CI/CD

Add feature parity checking to your CI pipeline:

```bash
#!/bin/bash
# ci_feature_parity_check.sh

set -e

echo "Checking feature parity with original MMDetection..."

# Check for missing code
MISSING=$(codediff archive/mmdet/models/ visdet/models/ \
  --normalize-imports \
  --missing-only \
  --format json)

if [ "$(echo $MISSING | jq '.stats.total_files')" -gt 0 ]; then
  echo "❌ FAILED: Missing implementations detected"
  echo "$MISSING" | jq '.'
  exit 1
fi

# Check for unexpected modifications
CHANGES=$(codediff archive/mmdet/models/ visdet/models/ \
  --normalize-imports \
  --modified-only \
  --format json)

CHANGE_COUNT=$(echo $CHANGES | jq '.stats.files.modified')
echo "✅ Found $CHANGE_COUNT intentional modifications (expected behavior)"

# Generate detailed report
codediff archive/mmdet/models/ visdet/models/ \
  --normalize-imports \
  --show-diff \
  --format markdown \
  --output /tmp/parity_report.md

echo "✅ Feature parity check passed"
echo "See /tmp/parity_report.md for detailed analysis"
```

## Common Issues & Solutions

### Issue: Too many "modified" files showing

**Solution:** You're comparing without `--normalize-imports`. Try:

```bash
codediff archive/mmdet/models/ visdet/models/ \
  --normalize-imports \
  --modified-only
```

### Issue: Import paths still showing as different

**Solution:** The tool normalizes source code. If imports still differ after normalization, they're using different libraries:

```python
# After normalization - these are DIFFERENT:
from visdet.cv.ops import nms
from custom_ops import nms  # Not normalized
```

This is a **real difference** and should be flagged.

### Issue: Functions showing as "added" when they should be "unchanged"

**Solution:** Check if the function was moved to a different file:

```bash
codediff archive/mmdet/models/utils.py \
  visdet/models/new_location.py \
  --normalize-imports \
  --renamed-only
```

The tool's `--renamed-only` filter uses semantic hashing to find moved/renamed code.

## Best Practices

1. **Always use `--normalize-imports` when comparing original vs refactored code:**
   ```bash
   codediff archive/mmdet/ visdet/ --normalize-imports
   ```

2. **Start with `--missing-only` to find gaps:**
   ```bash
   codediff archive/mmdet/ visdet/ \
     --normalize-imports \
     --missing-only \
     --format markdown
   ```

3. **Review modifications with diffs:**
   ```bash
   codediff archive/mmdet/ visdet/ \
     --normalize-imports \
     --modified-only \
     --show-diff \
     --context-lines 5
   ```

4. **Use pattern filtering for focused analysis:**
   ```bash
   codediff archive/mmdet/models/ visdet/models/ \
     --normalize-imports \
     --pattern "ResNet.*" \
     --show-diff
   ```

5. **Generate reports for documentation:**
   ```bash
   codediff archive/mmdet/ visdet/ \
     --normalize-imports \
     --format markdown \
     --output FEATURE_PARITY_REPORT.md
   ```

## Understanding Import Normalization

When you use `--normalize-imports`, the tool:

1. **Reads each Python file** as source code
2. **Normalizes imports** using regex substitution:
   - `from mmcv...` → `from visdet.cv...`
   - `import mmcv...` → `import visdet.cv...`
   - `mmcv.` → `visdet.cv.`
   - Same for `mmengine` → `visdet.engine`
3. **Generates semantic hashes** from the normalized code
4. **Compares hashes** to detect renames/moves
5. **Reports differences** based on normalized content

This is done **without modifying files** - it's purely for comparison purposes.

## Examples

### Example 1: Validate ResNet Implementation Parity

```bash
codediff archive/mmdet/models/backbones/resnet.py \
  visdet/models/backbones/resnet.py \
  --normalize-imports \
  --show-diff \
  --format markdown
```

Expected output: Either "unchanged" or list real improvements.

### Example 2: Find Missing Detectors

```bash
codediff archive/mmdet/models/detectors/ \
  visdet/models/detectors/ \
  --normalize-imports \
  --missing-only \
  --format json | jq '.stats'
```

Expected output: Should show 0 missing (if full parity achieved).

### Example 3: Check Data Pipeline Parity

```bash
codediff archive/mmdet/datasets/ \
  visdet/datasets/ \
  --normalize-imports \
  --pattern ".*Transform.*" \
  --modified-only \
  --show-diff
```

This shows which data transforms have been intentionally modified.

### Example 4: Validate Config System Parity

```bash
codediff archive/mmdet/config/ \
  visdet/config/ \
  --normalize-imports \
  --show-diff \
  --context-lines 3 \
  --format markdown \
  --output config_parity_report.md
```

## Troubleshooting

### Tool shows imports as modified even with `--normalize-imports`

Check if they're from non-MMDetection libraries:

```bash
# After normalization, check the output
codediff file1.py file2.py --normalize-imports --show-diff

# These would still show as different (correct behavior):
from pytorch import nn  # Not normalized
from tensorflow import keras  # Not normalized
```

### Too many false positives

Make sure you're using `--normalize-imports`:

```bash
# Wrong - will show many false positives
codediff archive/mmdet/ visdet/

# Correct
codediff archive/mmdet/ visdet/ --normalize-imports
```

### Need to check specific functions

Use `--function` with `--normalize-imports`:

```bash
codediff archive/mmdet/models/detectors/faster_rcnn.py \
  visdet/models/detectors/faster_rcnn.py \
  --normalize-imports \
  --function forward \
  --show-diff
```

## Conclusion

Feature parity validation with `--normalize-imports` ensures:

✅ **No missed implementations** (nothing accidentally left behind)
✅ **Only intentional changes tracked** (see actual improvements)
✅ **Clear refactoring documentation** (what changed and why)
✅ **Confidence in migration** (validated feature parity)

Use this guide to systematically validate that your refactored codebase maintains complete feature parity with the original!
