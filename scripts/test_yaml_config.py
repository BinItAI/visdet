#!/usr/bin/env python3
"""Test script for YAML configuration loading.

This script tests:
1. Loading YAML configs with _base_ inheritance
2. Resolving $ref references
3. Config attribute access
4. Backward compatibility with Config.fromfile()
"""

import sys
from pathlib import Path

# Add visdet to path
sys.path.insert(0, str(Path(__file__).parent.parent / "visdet"))

# Import directly from the config module to avoid importing the whole visdet package
from visdet.engine.config.config_wrapper import Config


def test_yaml_loading():
    """Test loading YAML config with _base_ and $ref."""
    print("=" * 80)
    print("Testing YAML Config Loading")
    print("=" * 80)

    # Load the experiment config
    config_path = Path(__file__).parent.parent / "configs/experiments/mask_rcnn_swin_tiny_coco.yaml"

    print(f"\nLoading config from: {config_path}")

    try:
        cfg = Config.fromfile(config_path, deprecation_warning=False)
        print("✓ Config loaded successfully!")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test _base_ inheritance
    print("\n" + "-" * 80)
    print("Testing _base_ Inheritance")
    print("-" * 80)

    # Check if dataset config was inherited
    if hasattr(cfg, "data_root"):
        print(f"✓ Dataset config inherited: data_root = {cfg.data_root}")
    else:
        print("✗ Dataset config not inherited")
        return False

    # Check if optimizer config was inherited
    if hasattr(cfg, "type") and cfg.type == "AdamW":
        print(f"✓ Optimizer config inherited: type = {cfg.type}")
    else:
        print("✗ Optimizer config not inherited")
        return False

    # Check if schedule config was inherited
    if hasattr(cfg, "max_epochs"):
        print(f"✓ Schedule config inherited: max_epochs = {cfg.max_epochs}")
    else:
        print("✗ Schedule config not inherited")
        return False

    # Test override mechanism
    print("\n" + "-" * 80)
    print("Testing Config Override")
    print("-" * 80)

    # Check that max_epochs was overridden from 12 to 24
    if cfg.max_epochs == 24:
        print(f"✓ Override successful: max_epochs = {cfg.max_epochs} (overrode from 12)")
    else:
        print(f"✗ Override failed: max_epochs = {cfg.max_epochs} (expected 24)")
        return False

    # Test $ref resolution
    print("\n" + "-" * 80)
    print("Testing $ref Resolution")
    print("-" * 80)

    # Check if backbone config was resolved
    if hasattr(cfg, "model") and hasattr(cfg.model, "backbone"):
        backbone = cfg.model.backbone
        if hasattr(backbone, "type") and backbone.type == "SwinTransformer":
            print(f"✓ Backbone $ref resolved: type = {backbone.type}")
            print(f"  - embed_dims = {backbone.embed_dims}")
            print(f"  - depths = {backbone.depths}")
        else:
            print("✗ Backbone $ref not resolved correctly")
            return False
    else:
        print("✗ Model backbone not found")
        return False

    # Check if neck config was resolved
    if hasattr(cfg.model, "neck"):
        neck = cfg.model.neck
        if hasattr(neck, "type") and neck.type == "FPN":
            print(f"✓ Neck $ref resolved: type = {neck.type}")
            print(f"  - out_channels = {neck.out_channels}")
        else:
            print("✗ Neck $ref not resolved correctly")
            return False
    else:
        print("✗ Model neck not found")
        return False

    # Test attribute access
    print("\n" + "-" * 80)
    print("Testing Attribute Access")
    print("-" * 80)

    # Test nested attribute access
    try:
        model_type = cfg.model.type
        backbone_type = cfg.model.backbone.type
        neck_out_channels = cfg.model.neck.out_channels
        print("✓ Attribute access works:")
        print(f"  - cfg.model.type = {model_type}")
        print(f"  - cfg.model.backbone.type = {backbone_type}")
        print(f"  - cfg.model.neck.out_channels = {neck_out_channels}")
    except AttributeError as e:
        print(f"✗ Attribute access failed: {e}")
        return False

    # Test dict-style access
    try:
        model_type = cfg["model"]["type"]
        backbone_type = cfg["model"]["backbone"]["type"]
        print("✓ Dict-style access works:")
        print(f"  - cfg['model']['type'] = {model_type}")
        print(f"  - cfg['model']['backbone']['type'] = {backbone_type}")
    except (KeyError, TypeError) as e:
        print(f"✗ Dict-style access failed: {e}")
        return False

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
    return True


def test_component_configs():
    """Test loading individual component configs."""
    print("\n" + "=" * 80)
    print("Testing Individual Component Configs")
    print("=" * 80)

    configs_dir = Path(__file__).parent.parent / "configs/components"

    # Test backbone config
    print("\n" + "-" * 80)
    print("Testing Backbone Config")
    print("-" * 80)

    backbone_path = configs_dir / "backbones/swin_tiny.yaml"
    try:
        cfg = Config.fromfile(backbone_path, deprecation_warning=False)
        print(f"✓ Loaded: {backbone_path.name}")
        print(f"  - type: {cfg.type}")
        print(f"  - embed_dims: {cfg.embed_dims}")
        print(f"  - depths: {cfg.depths}")
    except Exception as e:
        print(f"✗ Failed to load {backbone_path.name}: {e}")
        return False

    # Test neck config
    print("\n" + "-" * 80)
    print("Testing Neck Config")
    print("-" * 80)

    neck_path = configs_dir / "necks/fpn_256.yaml"
    try:
        cfg = Config.fromfile(neck_path, deprecation_warning=False)
        print(f"✓ Loaded: {neck_path.name}")
        print(f"  - type: {cfg.type}")
        print(f"  - out_channels: {cfg.out_channels}")
        print(f"  - num_outs: {cfg.num_outs}")
    except Exception as e:
        print(f"✗ Failed to load {neck_path.name}: {e}")
        return False

    # Test optimizer config
    print("\n" + "-" * 80)
    print("Testing Optimizer Config")
    print("-" * 80)

    optimizer_path = configs_dir / "optimizers/adamw_default.yaml"
    try:
        cfg = Config.fromfile(optimizer_path, deprecation_warning=False)
        print(f"✓ Loaded: {optimizer_path.name}")
        print(f"  - type: {cfg.type}")
        print(f"  - lr: {cfg.lr}")
        print(f"  - weight_decay: {cfg.weight_decay}")
    except Exception as e:
        print(f"✗ Failed to load {optimizer_path.name}: {e}")
        return False

    print("\n" + "=" * 80)
    print("All component config tests passed! ✓")
    print("=" * 80)
    return True


if __name__ == "__main__":
    print("\n")
    success = True

    # Test individual component configs first
    if not test_component_configs():
        success = False

    # Test experiment config with _base_ and $ref
    if not test_yaml_loading():
        success = False

    if success:
        print("\n🎉 All YAML configuration tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
