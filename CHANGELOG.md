# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Dynamic Annotation File Parameters** - SimpleRunner now accepts `train_ann_file` and `val_ann_file` parameters for specifying annotation files directly at runtime (#XX)
  - Enables seamless integration with ML pipelines where annotation files are generated on-the-fly
  - Supports MLflow experiment tracking artifact downloads
  - Enables cross-validation workflows with fold-specific annotation files
  - Supports A/B testing with programmatically generated data distributions
  - Works with CI/CD pipelines for integration testing with synthetic data
  - Both parameters validated during initialization for early error detection
  - Fully backward compatible - existing code without these parameters works unchanged
  - Comprehensive documentation with real-world examples in `docs/DYNAMIC_ANNOTATIONS.md`
  - Full test coverage with 13+ unit tests in `tests/test_simple_runner_annotation_files.py`
  - Integration examples demonstrating MLflow, cross-validation, and A/B testing workflows

### Changed

- SimpleRunner `__init__` now validates annotation files before building config
  - Provides clear error messages with both provided and resolved file paths
  - Catches configuration errors early before training starts

### Technical Details

- Added `_validate_annotation_files()` method to SimpleRunner for early validation
- Added logging for annotation file overrides in `_build_config()`
- Implementation uses priority hierarchy: explicit parameters > preset definitions > defaults
- DDP-compatible: annotation file parameters stored in `_init_args` for worker recreation
- No performance impact: validation is one-time during initialization

---

## Legend

- `Added` for new features
- `Changed` for changes in existing functionality
- `Deprecated` for soon-to-be removed features
- `Removed` for now removed features
- `Fixed` for any bug fixes
- `Security` in case of vulnerabilities
