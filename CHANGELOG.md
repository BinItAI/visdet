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

- **Automatic Class Detection and Configuration** - SimpleRunner now automatically detects and configures model classes from annotation files
  - Parses COCO annotation files at runtime to detect actual classes in data
  - Eliminates manual `num_classes` configuration
  - Uses UNION of classes when both train and val annotation files provided
  - Prevents mismatches between model architecture and training data
  - Supports both StandardRoIHead (single dict) and CascadeRoIHead (multi-stage list)
  - Clear priority hierarchy: annotation files > dataset metainfo > skip
  - HIGH severity warning for validation-only classes (model won't learn them)
  - MEDIUM severity warning for training-only classes (no validation metrics)
  - Error detection for category ID conflicts and non-contiguous IDs
  - Comprehensive logging for debugging class configuration issues
  - Zero performance impact - detection is one-time during initialization
  - Full test coverage with 8+ new test cases in `TestAnnotationClassDetection`

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
