#!/usr/bin/env python3
"""Simple test for preset registry (avoids full visdet import).

This script tests just the preset system without importing
the full visdet package.
"""

import sys
from pathlib import Path

# Add visdet to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "visdet/visdet"))


def test_preset_discovery():
    """Test preset discovery without full package import."""
    print("=" * 80)
    print("Testing Preset Discovery (Lightweight)")
    print("=" * 80)

    # Import only the preset registry, not the full package
    from presets.registry import (
        DATASET_PRESETS,
        MODEL_PRESETS,
        OPTIMIZER_PRESETS,
        SCHEDULER_PRESETS,
    )

    # Test model presets
    print("\n" + "-" * 80)
    print("Model Presets:")
    print("-" * 80)
    models = MODEL_PRESETS.list()
    for model in models:
        print(f"  - {model}")
    assert len(models) > 0, "No model presets found"
    print(f"✓ Found {len(models)} model presets")

    # Test dataset presets
    print("\n" + "-" * 80)
    print("Dataset Presets:")
    print("-" * 80)
    datasets = DATASET_PRESETS.list()
    for dataset in datasets:
        print(f"  - {dataset}")
    assert len(datasets) > 0, "No dataset presets found"
    print(f"✓ Found {len(datasets)} dataset presets")

    # Test optimizer presets
    print("\n" + "-" * 80)
    print("Optimizer Presets:")
    print("-" * 80)
    optimizers = OPTIMIZER_PRESETS.list()
    for optimizer in optimizers:
        print(f"  - {optimizer}")
    assert len(optimizers) > 0, "No optimizer presets found"
    print(f"✓ Found {len(optimizers)} optimizer presets")

    # Test scheduler presets
    print("\n" + "-" * 80)
    print("Scheduler Presets:")
    print("-" * 80)
    schedulers = SCHEDULER_PRESETS.list()
    for scheduler in schedulers:
        print(f"  - {scheduler}")
    print(f"✓ Found {len(schedulers)} scheduler presets")

    return True


def test_preset_loading():
    """Test that presets can be loaded."""
    print("\n" + "=" * 80)
    print("Testing Preset Loading")
    print("=" * 80)

    from presets.registry import DATASET_PRESETS, MODEL_PRESETS, OPTIMIZER_PRESETS

    # Test model loading
    print("\n" + "-" * 80)
    print("Loading: mask_rcnn_swin_s")
    print("-" * 80)
    try:
        model_cfg = MODEL_PRESETS.get("mask_rcnn_swin_s")
        assert isinstance(model_cfg, dict)
        assert "type" in model_cfg
        print(f"✓ Loaded model config (type={model_cfg['type']})")
        print(f"  - Keys: {', '.join(list(model_cfg.keys())[:10])}")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return False

    # Test dataset loading
    print("\n" + "-" * 80)
    print("Loading: coco_instance_segmentation")
    print("-" * 80)
    try:
        dataset_cfg = DATASET_PRESETS.get("coco_instance_segmentation")
        assert isinstance(dataset_cfg, dict)
        assert "type" in dataset_cfg
        print(f"✓ Loaded dataset config (type={dataset_cfg['type']})")
        print(f"  - Keys: {', '.join(list(dataset_cfg.keys())[:10])}")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return False

    # Test optimizer loading
    print("\n" + "-" * 80)
    print("Loading: adamw_default")
    print("-" * 80)
    try:
        optimizer_cfg = OPTIMIZER_PRESETS.get("adamw_default")
        assert isinstance(optimizer_cfg, dict)
        assert "type" in optimizer_cfg
        print(f"✓ Loaded optimizer config (type={optimizer_cfg['type']})")
        print(f"  - lr={optimizer_cfg.get('lr')}, weight_decay={optimizer_cfg.get('weight_decay')}")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return False

    return True


def test_preset_registration():
    """Test custom preset registration."""
    print("\n" + "=" * 80)
    print("Testing Custom Preset Registration")
    print("=" * 80)

    from presets.registry import MODEL_PRESETS

    # Register custom preset
    print("\n" + "-" * 80)
    print("Registering: my_custom_model")
    print("-" * 80)
    custom_cfg = {"type": "CustomModel", "param1": "value1", "param2": 42}
    MODEL_PRESETS.register("my_custom_model", custom_cfg)

    # Verify it appears in list
    models = MODEL_PRESETS.list()
    assert "my_custom_model" in models
    print("✓ Custom preset appears in list")

    # Load it back
    loaded_cfg = MODEL_PRESETS.get("my_custom_model")
    assert loaded_cfg == custom_cfg
    print("✓ Custom preset loaded correctly")

    return True


def test_lazy_loading():
    """Test that loading is actually lazy."""
    print("\n" + "=" * 80)
    print("Testing Lazy Loading")
    print("=" * 80)

    from presets.registry import PresetRegistry

    # Create a test registry
    preset_dir = repo_root / "configs/models"
    registry = PresetRegistry(preset_dir)

    print("\n" + "-" * 80)
    print("After initialization:")
    print("-" * 80)
    print(f"  Available names: {len(registry._available_names)}")
    print(f"  Cached configs: {len(registry._cache)}")
    assert len(registry._cache) == 0, "Cache should be empty initially"
    print("✓ Cache is empty after initialization (lazy loading confirmed)")

    print("\n" + "-" * 80)
    print("After loading one preset:")
    print("-" * 80)
    _ = registry.get("mask_rcnn_swin_s")
    print(f"  Available names: {len(registry._available_names)}")
    print(f"  Cached configs: {len(registry._cache)}")
    assert len(registry._cache) == 1, "Cache should have one entry"
    print("✓ Only loaded preset is cached")

    return True


if __name__ == "__main__":
    print("\n🔍 Preset System Test Suite (Lightweight)\n")

    all_passed = True

    try:
        if not test_preset_discovery():
            all_passed = False
        if not test_preset_loading():
            all_passed = False
        if not test_preset_registration():
            all_passed = False
        if not test_lazy_loading():
            all_passed = False
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False

    if all_passed:
        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe preset system is working correctly:")
        print("  ✓ Preset discovery works")
        print("  ✓ Preset loading works")
        print("  ✓ Custom registration works")
        print("  ✓ Lazy loading is confirmed")
        print("\n")
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("❌ SOME TESTS FAILED")
        print("=" * 80)
        sys.exit(1)
