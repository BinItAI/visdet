#!/usr/bin/env python3
"""Simplified test for YAML configuration loading.

This test directly imports the YAML loader modules without importing
the full visdet package, avoiding import dependencies.
"""

import sys
from pathlib import Path

# Setup repo_root before imports
repo_root = Path(__file__).parent.parent

# Add parent directory to path for direct module imports
sys.path.insert(0, str(repo_root / "visdet/visdet"))

# Import YAML loader directly (avoid full visdet package import)
# ruff: noqa: E402
from engine.config.yaml_loader import load_yaml_config


def test_component_loading():
    """Test loading individual component configs."""
    print("=" * 80)
    print("Testing Individual Component YAML Loading")
    print("=" * 80)

    configs_dir = repo_root / "configs/components"

    # Test backbone config
    print("\n" + "-" * 80)
    print("Test 1: Load Backbone Config")
    print("-" * 80)

    backbone_path = configs_dir / "backbones/swin_tiny.yaml"
    print(f"Loading: {backbone_path}")

    try:
        cfg = load_yaml_config(backbone_path)
        print("✓ Loaded successfully!")
        print(f"  - type: {cfg.type}")
        print(f"  - embed_dims: {cfg.embed_dims}")
        print(f"  - depths: {cfg.depths}")
        print(f"  - num_heads: {cfg.num_heads}")
        assert cfg.type == "SwinTransformer"
        assert cfg.embed_dims == 96
        assert cfg.depths == [2, 2, 6, 2]
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test neck config
    print("\n" + "-" * 80)
    print("Test 2: Load Neck Config")
    print("-" * 80)

    neck_path = configs_dir / "necks/fpn_256.yaml"
    print(f"Loading: {neck_path}")

    try:
        cfg = load_yaml_config(neck_path)
        print("✓ Loaded successfully!")
        print(f"  - type: {cfg.type}")
        print(f"  - in_channels: {cfg.in_channels}")
        print(f"  - out_channels: {cfg.out_channels}")
        print(f"  - num_outs: {cfg.num_outs}")
        assert cfg.type == "FPN"
        assert cfg.out_channels == 256
        assert cfg.num_outs == 5
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test optimizer config
    print("\n" + "-" * 80)
    print("Test 3: Load Optimizer Config")
    print("-" * 80)

    optimizer_path = configs_dir / "optimizers/adamw_default.yaml"
    print(f"Loading: {optimizer_path}")

    try:
        cfg = load_yaml_config(optimizer_path)
        print("✓ Loaded successfully!")
        print(f"  - type: {cfg.type}")
        print(f"  - lr: {cfg.lr}")
        print(f"  - betas: {cfg.betas}")
        print(f"  - weight_decay: {cfg.weight_decay}")
        assert cfg.type == "AdamW"
        assert cfg.lr == 0.0001
        assert cfg.weight_decay == 0.05
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("✓ All component loading tests passed!")
    print("=" * 80)
    return True


def test_base_inheritance():
    """Test _base_ inheritance mechanism."""
    print("\n" + "=" * 80)
    print("Testing _base_ Inheritance")
    print("=" * 80)

    experiment_path = repo_root / "configs/experiments/mask_rcnn_swin_tiny_coco.yaml"
    print(f"\nLoading: {experiment_path}")

    try:
        cfg = load_yaml_config(experiment_path)
        print("✓ Loaded successfully!")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test that _base_ configs were merged
    print("\n" + "-" * 80)
    print("Checking _base_ Merged Configs")
    print("-" * 80)

    # Check dataset config (from coco_instance.yaml)
    if hasattr(cfg, "data_root") and cfg.data_root == "data/coco/":
        print("✓ Dataset config inherited correctly")
        print(f"  - data_root: {cfg.data_root}")
        print(f"  - ann_file: {cfg.ann_file}")
    else:
        print("✗ Dataset config not inherited")
        return False

    # Check optimizer config (from adamw_default.yaml)
    if hasattr(cfg, "type") and cfg.type == "AdamW":
        print("✓ Optimizer config inherited correctly")
        print(f"  - type: {cfg.type}")
        print(f"  - lr: {cfg.lr}")
    else:
        print("✗ Optimizer config not inherited")
        return False

    # Check schedule config (from 1x.yaml)
    if hasattr(cfg, "max_epochs"):
        print("✓ Schedule config inherited correctly")
        # Note: This should be 24 (overridden) not 12 (from base)
        print(f"  - max_epochs: {cfg.max_epochs}")
    else:
        print("✗ Schedule config not inherited")
        return False

    # Test override mechanism
    print("\n" + "-" * 80)
    print("Checking Config Overrides")
    print("-" * 80)

    if cfg.max_epochs == 24:
        print("✓ Override works: max_epochs overridden to 24 (from 12 in base)")
    else:
        print(f"✗ Override failed: max_epochs = {cfg.max_epochs} (expected 24)")
        return False

    if hasattr(cfg, "val_interval") and cfg.val_interval == 2:
        print("✓ Override works: val_interval overridden to 2 (from 1 in base)")
    else:
        print(f"✗ Override failed: val_interval = {getattr(cfg, 'val_interval', 'N/A')}")
        return False

    print("\n" + "=" * 80)
    print("✓ All _base_ inheritance tests passed!")
    print("=" * 80)
    return True


def test_ref_resolution():
    """Test $ref reference resolution."""
    print("\n" + "=" * 80)
    print("Testing $ref Resolution")
    print("=" * 80)

    experiment_path = repo_root / "configs/experiments/mask_rcnn_swin_tiny_coco.yaml"
    print(f"\nLoading: {experiment_path}")

    try:
        cfg = load_yaml_config(experiment_path)
        print("✓ Loaded successfully!")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return False

    # Check backbone $ref resolution
    print("\n" + "-" * 80)
    print("Checking Backbone $ref")
    print("-" * 80)

    if hasattr(cfg, "model") and hasattr(cfg.model, "backbone"):
        backbone = cfg.model.backbone
        if backbone.type == "SwinTransformer" and backbone.embed_dims == 96:
            print("✓ Backbone $ref resolved correctly")
            print(f"  - type: {backbone.type}")
            print(f"  - embed_dims: {backbone.embed_dims}")
            print(f"  - depths: {backbone.depths}")
        else:
            print("✗ Backbone $ref not resolved correctly")
            return False
    else:
        print("✗ Model backbone not found")
        return False

    # Check neck $ref resolution
    print("\n" + "-" * 80)
    print("Checking Neck $ref")
    print("-" * 80)

    if hasattr(cfg.model, "neck"):
        neck = cfg.model.neck
        if neck.type == "FPN" and neck.out_channels == 256:
            print("✓ Neck $ref resolved correctly")
            print(f"  - type: {neck.type}")
            print(f"  - out_channels: {neck.out_channels}")
            print(f"  - num_outs: {neck.num_outs}")
        else:
            print("✗ Neck $ref not resolved correctly")
            return False
    else:
        print("✗ Model neck not found")
        return False

    print("\n" + "=" * 80)
    print("✓ All $ref resolution tests passed!")
    print("=" * 80)
    return True


def test_attribute_access():
    """Test ConfigDict attribute access."""
    print("\n" + "=" * 80)
    print("Testing ConfigDict Attribute Access")
    print("=" * 80)

    experiment_path = repo_root / "configs/experiments/mask_rcnn_swin_tiny_coco.yaml"
    cfg = load_yaml_config(experiment_path)

    # Test attribute access
    print("\n" + "-" * 80)
    print("Test Attribute Access (dot notation)")
    print("-" * 80)

    try:
        assert cfg.model.type == "MaskRCNN"
        assert cfg.model.backbone.type == "SwinTransformer"
        assert cfg.model.neck.out_channels == 256
        assert cfg.max_epochs == 24
        print("✓ Attribute access works")
        print(f"  - cfg.model.type = {cfg.model.type}")
        print(f"  - cfg.model.backbone.type = {cfg.model.backbone.type}")
        print(f"  - cfg.max_epochs = {cfg.max_epochs}")
    except (AttributeError, AssertionError) as e:
        print(f"✗ Attribute access failed: {e}")
        return False

    # Test dict-style access
    print("\n" + "-" * 80)
    print("Test Dict-Style Access (bracket notation)")
    print("-" * 80)

    try:
        assert cfg["model"]["type"] == "MaskRCNN"
        assert cfg["model"]["backbone"]["type"] == "SwinTransformer"
        assert cfg["model"]["neck"]["out_channels"] == 256
        assert cfg["max_epochs"] == 24
        print("✓ Dict-style access works")
        print(f"  - cfg['model']['type'] = {cfg['model']['type']}")
        print(f"  - cfg['model']['backbone']['type'] = {cfg['model']['backbone']['type']}")
        print(f"  - cfg['max_epochs'] = {cfg['max_epochs']}")
    except (KeyError, AssertionError) as e:
        print(f"✗ Dict-style access failed: {e}")
        return False

    print("\n" + "=" * 80)
    print("✓ All attribute access tests passed!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    print("\n🔍 YAML Configuration System Test Suite\n")

    all_passed = True

    # Run all tests
    if not test_component_loading():
        all_passed = False

    if not test_base_inheritance():
        all_passed = False

    if not test_ref_resolution():
        all_passed = False

    if not test_attribute_access():
        all_passed = False

    # Final summary
    if all_passed:
        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe YAML configuration system is working correctly:")
        print("  ✓ Individual component configs load successfully")
        print("  ✓ _base_ inheritance merges parent configs")
        print("  ✓ Config overrides work as expected")
        print("  ✓ $ref resolution loads referenced files")
        print("  ✓ Attribute and dict-style access both work")
        print("\n")
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("❌ SOME TESTS FAILED")
        print("=" * 80)
        print("\nPlease review the output above for details.\n")
        sys.exit(1)
