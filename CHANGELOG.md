# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- SimpleRunner accepts `train_ann_file` and `val_ann_file` parameters to override annotation file paths
- Automatic class detection from annotation files - `num_classes` is configured automatically

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
