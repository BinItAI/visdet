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

For project-specific guidelines, see `visdet/AGENTS.md` (model-focused documentation)
