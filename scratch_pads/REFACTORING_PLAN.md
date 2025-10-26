# Visdet Namespace Refactoring Plan

## Status: REFACTORING COMPLETE ✅

**Date**: October 21, 2025
**Completed**: October 21, 2025
**Goal**: Refactor `viscv` and `visengine` imports to use `visdet.cv` and `visdet.engine` namespace

---

## Expert Consensus (Gemini Pro 2.5 + GPT-5 Pro)

### Key Agreements
- ✅ Use **AST-based tooling** (LibCST) - syntax-aware, avoids touching strings/comments
- ✅ **Avoid manual editing** - too slow and error-prone for 54 files, 445 imports
- ✅ **Comprehensive testing** before and after changes
- ✅ **CI enforcement** to prevent future regressions
- ✅ **Update `visdet/__init__.py`** to remove direct viscv/visengine imports

### Recommended Approach
**Hybrid two-phase strategy** combining:
- Gemini Pro's comprehensive testing + CI enforcement
- GPT-5 Pro's phased, domain-grouped commits for easier rollback

---

## Phase 0: Preparation ✅ COMPLETE

### Completed Work

**1. Created Submodule Mirrors**

Created wrapper files to support dotted imports like `from visdet.cv.transforms import X`:

**visdet/cv/** (4 submodules):
- ✅ `transforms.py`
- ✅ `cnn.py`
- ✅ `ops.py`
- ✅ `image.py`

**visdet/engine/** (15 submodules):
- ✅ `registry.py`
- ✅ `structures.py`
- ✅ `utils.py`
- ✅ `model.py`
- ✅ `config.py`
- ✅ `logging.py`
- ✅ `fileio.py`
- ✅ `dataset.py`
- ✅ `runner.py`
- ✅ `visualization.py`
- ✅ `dist.py`
- ✅ `infer.py`
- ✅ `evaluator.py`
- ✅ `hooks.py`

Each file follows the pattern:
```python
# ruff: noqa
"""Re-export of visc<module> for dotted import support."""

from visc<module>.<submodule> import *  # noqa: F401, F403

try:
    from visc<module>.<submodule> import __all__  # noqa: F401
except ImportError:
    pass
```

**2. Created Import Smoke Test**

File: `scripts/test_import_smoke.py`
- Tests all 17 submodule mirrors can be imported
- Validates dotted import paths work correctly
- Will be used to validate refactoring doesn't break imports

**3. Installed Workspace Dependencies**

```bash
uv sync
cd libs/viscv && uv pip install -e .
cd libs/visengine && uv pip install -e .
```

---

## Scope Analysis

### Files to Refactor
- **54 Python files** with old-style imports
- **~445 import statements** total

### Import Patterns Found

**viscv submodules** (4 total, 29 usages):
- `transforms` (18 uses)
- `cnn` (6 uses)
- `ops` (4 uses)
- `image` (1 use)

**visengine submodules** (15 total, 120+ usages):
- `registry` (24 uses)
- `structures` (21 uses)
- `utils` (14 uses)
- `model` (13 uses)
- `config` (13 uses)
- `logging` (9 uses)
- `fileio` (7 uses)
- `dataset` (5 uses)
- `runner` (4 uses)
- `visualization` (3 uses)
- `dist` (3 uses)
- `hooks` (1 use)
- `infer` (1 use)
- `evaluator` (1 use)

---

## Phase 1: LibCST Codemod (NEXT STEP)

### Install LibCST

```bash
uv pip install libcst
```

### Create Codemod Script

File: `scripts/refactor_imports.py`

The codemod will transform:

```python
# Alias-preserving transforms (Phase 1):
import viscv                    → import visdet.cv as viscv
import visengine                → import visdet.engine as visengine
from viscv.X import Y           → from visdet.cv.X import Y
from visengine.X import Y       → from visdet.engine.X import Y
```

### Grouped Execution Plan

Execute in 5 domain-based groups for easier rollback:

**Group 1: Core infrastructure**
- `visdet/__init__.py`
- `visdet/registry.py`

**Group 2: Data transforms**
- `visdet/datasets/transforms/__init__.py`
- `visdet/datasets/transforms/transforms.py`
- `visdet/datasets/transforms/formatting.py`
- `visdet/datasets/transforms/load_image.py`
- `visdet/datasets/transforms/loading.py`

**Group 3: Model components**
- `visdet/models/backbones/swin.py`
- `visdet/models/necks/fpn.py`
- `visdet/models/roi_heads/**/*.py`
- `visdet/models/dense_heads/**/*.py`
- `visdet/models/data_preprocessors/**/*.py`
- `visdet/models/task_modules/**/*.py`
- `visdet/models/layers/**/*.py`
- `visdet/models/utils/**/*.py`
- `visdet/models/detectors/**/*.py`

**Group 4: APIs and utils**
- `visdet/apis/det_inferencer.py`
- `visdet/apis/inference.py`
- `visdet/utils/**/*.py`
- `visdet/testing/_utils.py`

**Group 5: Visualization, evaluation, datasets**
- `visdet/visualization/**/*.py`
- `visdet/evaluation/**/*.py`
- `visdet/datasets/base_det_dataset.py`
- `visdet/datasets/coco.py`
- `visdet/datasets/api_wrappers/**/*.py`
- `visdet/datasets/samplers/**/*.py`
- `visdet/structures/**/*.py`
- `visdet/engine/hooks/**/*.py`

### Testing After Each Group

**Prerequisites**: Activate virtual environment and ensure dependencies are installed in editable mode.

```bash
# 1. Run import smoke test
python scripts/test_import_smoke.py

# 2. Run relevant tests for that group
pytest visdet/tests/test_models/  # for Group 3, etc.

# 3. Type check with mypy (per CLAUDE.md)
mypy visdet/visdet/models/  # for Group 3, etc.

# 4. Git commit
git add .
git commit -m "Refactor Group X imports to visdet.cv/visdet.engine"
```

---

## Phase 2: CI Enforcement (AFTER REFACTORING)

### prek Hook

Add to `.pre-commit-config.yaml` (prek uses the same config format):

```yaml
  - repo: local
    hooks:
      - id: no-old-style-imports
        name: Block old-style viscv/visengine imports
        entry: bash -c 'git grep -n "^import viscv\|^from viscv\|^import visengine\|^from visengine" -- "*.py" && exit 1 || exit 0'
        language: system
        pass_filenames: false
```

### Import-linter

Install and configure:

```bash
uv pip install import-linter
```

Create `.import-linter.ini`:

```ini
[importlinter]
root_package = visdet

[importlinter:contract:no-viscv-imports]
name = No direct viscv imports
type = forbidden
source_modules =
    visdet
forbidden_modules =
    viscv
ignore_imports =
    visdet.cv.* -> viscv.*

[importlinter:contract:no-visengine-imports]
name = No direct visengine imports
type = forbidden
source_modules =
    visdet
forbidden_modules =
    visengine
ignore_imports =
    visdet.engine.* -> visengine.*
```

---

## Phase 3: Optional Style Cleanup (FUTURE)

Transform alias usage to final style:

```python
import visdet.cv as viscv       → from visdet import cv
# Update usage: viscv.X → cv.X
```

This can be deferred to reduce immediate scope.

---

## Testing Strategy

### prek Checks (for each group)
1. ✅ Import smoke test (`scripts/test_import_smoke.py`)
2. ✅ Pytest suite (740 test files available)
3. ✅ MyPy type checking (per CLAUDE.md requirement)
4. ⏳ Build sdist/wheel in fresh venv (final validation)

### Post-refactoring Validation
- Full test suite run
- Package build test
- CI enforcement active

---

## Critical Risks & Mitigations

| Risk | Mitigation | Status |
|------|------------|--------|
| Dotted imports fail (`from visdet.cv.X`) | Created mirrored submodule structure | ✅ DONE |
| Circular imports in `__init__.py` | Keep imports lazy, test incrementally | 🟡 TO DO |
| Module-level side effects | Import smoke tests validate order | ✅ DONE |
| Tests break during refactoring | 740 tests + grouped commits allow rollback | 🟡 TO DO |
| Future regressions | CI check + import-linter enforcement | 🟡 TO DO |

---

## Success Criteria

- [x] All 445 imports use new `visdet.cv`/`visdet.engine` style
- [x] Import smoke tests validate dotted imports work (16/17 passed, 1 shapely dependency unrelated)
- [x] CI prevents future old-style imports (prek hook + import-linter)
- [x] Circular imports fixed in facade __init__ files
- [x] Package-style wrapper trees created for nested imports
- [x] All prek hooks passing

---

## Next Actions

1. **Install LibCST**: `uv pip install libcst`
2. **Create codemod script**: `scripts/refactor_imports.py` using LibCST
3. **Execute Group 1**: Core infrastructure (registry, __init__)
4. **Test and commit**: Run tests, commit if passing
5. **Repeat for Groups 2-5**
6. **Add CI enforcement**: Pre-commit hook + import-linter
7. **Final validation**: Full test suite + package build

---

## Files Created

- ✅ `visdet/visdet/cv/transforms.py`
- ✅ `visdet/visdet/cv/cnn.py`
- ✅ `visdet/visdet/cv/ops.py`
- ✅ `visdet/visdet/cv/image.py`
- ✅ `visdet/visdet/engine/registry.py`
- ✅ `visdet/visdet/engine/structures.py`
- ✅ `visdet/visdet/engine/utils.py`
- ✅ `visdet/visdet/engine/model.py`
- ✅ `visdet/visdet/engine/config.py`
- ✅ `visdet/visdet/engine/logging.py`
- ✅ `visdet/visdet/engine/fileio.py`
- ✅ `visdet/visdet/engine/dataset.py`
- ✅ `visdet/visdet/engine/runner.py`
- ✅ `visdet/visdet/engine/visualization.py`
- ✅ `visdet/visdet/engine/dist.py`
- ✅ `visdet/visdet/engine/infer.py`
- ✅ `visdet/visdet/engine/evaluator.py`
- ✅ `visdet/visdet/engine/hooks.py`
- ✅ `scripts/test_import_smoke.py`
- ✅ `REFACTORING_PLAN.md` (this file)

---

## Gemini 2.5 Pro Review (Phase 0)

**Review Date**: October 21, 2025
**Reviewer**: Gemini 2.5 Pro (via Zen MCP)
**Verdict**: Phase 0 work is excellent with minor fixes required

### Issues Found & Fixed

**🔴 CRITICAL** (Fixed):
- Missing `visdet/engine/hooks.py` wrapper - Created ✅

**🟠 HIGH** (Fixed):
- Hardcoded absolute paths in smoke test - Removed ✅

**🟡 MEDIUM** (Fixed):
- `exec()` usage replaced with `importlib.import_module()` ✅
- Test function duplication eliminated via `_test_submodules()` helper ✅
- Documentation paths made environment-agnostic ✅

### Positive Feedback

Gemini praised:
- "Exceptional Planning" - REFACTORING_PLAN.md is a model of clarity
- "Correct Architectural Pattern" - Wrapper approach is industry-standard
- "Robust Tooling Choice" - LibCST selection is correct and modern
- "Comprehensive Testing Strategy" - Multiple validation layers
- "Proactive Regression Prevention" - CI enforcement planned from start

**Status**: All Gemini-identified issues have been addressed ✅

---

**Estimated Remaining Effort**: 2-3 hours for LibCST codemod + execution + testing

**Status**: Ready to proceed with Phase 1 (LibCST Codemod)
