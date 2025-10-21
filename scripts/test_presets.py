#!/usr/bin/env python3
"""Test script for preset registry and discovery.

This script tests:
1. Preset discovery (list functions)
2. Preset loading
3. Show preset functionality
4. Custom preset registration
"""

import sys
from pathlib import Path

# Add visdet to path
sys.path.insert(0, str(Path(__file__).parent.parent / "visdet/visdet"))


def test_discovery():
    """Test preset discovery functions."""
    print("=" * 80)
    print("Testing Preset Discovery")
    print("=" * 80)

    from visdet import SimpleRunner, presets

    # Test listing models
    print("\n" + "-" * 80)
    print("Available Model Presets:")
    print("-" * 80)
    models = SimpleRunner.list_models()
    for model in models:
        print(f"  - {model}")
    assert len(models) > 0, "No model presets found"
    print(f"✓ Found {len(models)} model presets")

    # Test listing datasets
    print("\n" + "-" * 80)
    print("Available Dataset Presets:")
    print("-" * 80)
    datasets = presets.list_datasets()
    for dataset in datasets:
        print(f"  - {dataset}")
    assert len(datasets) > 0, "No dataset presets found"
    print(f"✓ Found {len(datasets)} dataset presets")

    # Test listing optimizers
    print("\n" + "-" * 80)
    print("Available Optimizer Presets:")
    print("-" * 80)
    optimizers = presets.list_optimizers()
    for optimizer in optimizers:
        print(f"  - {optimizer}")
    assert len(optimizers) > 0, "No optimizer presets found"
    print(f"✓ Found {len(optimizers)} optimizer presets")

    # Test listing schedulers
    print("\n" + "-" * 80)
    print("Available Scheduler Presets:")
    print("-" * 80)
    schedulers = presets.list_schedulers()
    for scheduler in schedulers:
        print(f"  - {scheduler}")
    print(f"✓ Found {len(schedulers)} scheduler presets")

    return True


def test_loading():
    """Test preset loading."""
    print("\n" + "=" * 80)
    print("Testing Preset Loading")
    print("=" * 80)

    from visdet.presets import DATASET_PRESETS, MODEL_PRESETS, OPTIMIZER_PRESETS

    # Test model loading
    print("\n" + "-" * 80)
    print("Loading Model Preset: mask_rcnn_swin_s")
    print("-" * 80)
    model_cfg = MODEL_PRESETS.get("mask_rcnn_swin_s")
    assert isinstance(model_cfg, dict), "Model config should be a dict"
    assert "type" in model_cfg, "Model config should have 'type' key"
    assert model_cfg["type"] == "MaskRCNN", f"Expected MaskRCNN, got {model_cfg['type']}"
    print(f"✓ Loaded model config with type: {model_cfg['type']}")

    # Test dataset loading
    print("\n" + "-" * 80)
    print("Loading Dataset Preset: coco_instance_segmentation")
    print("-" * 80)
    dataset_cfg = DATASET_PRESETS.get("coco_instance_segmentation")
    assert isinstance(dataset_cfg, dict), "Dataset config should be a dict"
    assert "type" in dataset_cfg, "Dataset config should have 'type' key"
    print(f"✓ Loaded dataset config with type: {dataset_cfg['type']}")

    # Test optimizer loading
    print("\n" + "-" * 80)
    print("Loading Optimizer Preset: adamw_default")
    print("-" * 80)
    optimizer_cfg = OPTIMIZER_PRESETS.get("adamw_default")
    assert isinstance(optimizer_cfg, dict), "Optimizer config should be a dict"
    assert "type" in optimizer_cfg, "Optimizer config should have 'type' key"
    print(f"✓ Loaded optimizer config with type: {optimizer_cfg['type']}")

    return True


def test_show_preset():
    """Test show_preset functionality."""
    print("\n" + "=" * 80)
    print("Testing Show Preset")
    print("=" * 80)

    from visdet import SimpleRunner

    print("\n" + "-" * 80)
    print("Showing Model Preset: mask_rcnn_swin_s")
    print("-" * 80)
    SimpleRunner.show_preset("mask_rcnn_swin_s", category="model")

    return True


def test_registration():
    """Test custom preset registration."""
    print("\n" + "=" * 80)
    print("Testing Custom Preset Registration")
    print("=" * 80)

    from visdet import presets

    # Register a custom model
    print("\n" + "-" * 80)
    print("Registering Custom Model: my_custom_model")
    print("-" * 80)
    custom_config = {"type": "CustomModel", "param1": "value1"}
    presets.register_model("my_custom_model", custom_config)

    # Verify it was registered
    models = presets.list_models()
    assert "my_custom_model" in models, "Custom model should be in list"
    print("✓ Custom model registered successfully")

    # Load the custom model
    from visdet.presets import MODEL_PRESETS

    loaded_config = MODEL_PRESETS.get("my_custom_model")
    assert loaded_config == custom_config, "Loaded config should match registered config"
    print("✓ Custom model config loaded correctly")

    return True


def test_runner_initialization():
    """Test SimpleRunner initialization with presets."""
    print("\n" + "=" * 80)
    print("Testing SimpleRunner Initialization")
    print("=" * 80)

    from visdet import SimpleRunner

    # Test initialization with string presets
    print("\n" + "-" * 80)
    print("Initializing SimpleRunner with string presets")
    print("-" * 80)
    runner = SimpleRunner(
        model="mask_rcnn_swin_s",
        dataset="coco_instance_segmentation",
        optimizer="adamw_default",
    )
    assert runner.model_cfg["type"] == "MaskRCNN"
    assert runner.dataset_cfg["type"] == "CocoDataset"
    assert runner.optimizer_cfg["type"] == "AdamW"
    print("✓ SimpleRunner initialized successfully with string presets")

    # Test initialization with dict customization
    print("\n" + "-" * 80)
    print("Initializing SimpleRunner with customization via _base_")
    print("-" * 80)
    runner = SimpleRunner(
        model={"_base_": "mask_rcnn_swin_s", "test_override": "custom_value"},
        dataset="coco_instance_segmentation",
        optimizer="adamw_default",
    )
    assert runner.model_cfg["type"] == "MaskRCNN"
    assert "test_override" in runner.model_cfg
    assert runner.model_cfg["test_override"] == "custom_value"
    print("✓ SimpleRunner initialized with customization via _base_")

    return True


if __name__ == "__main__":
    print("\n🔍 Preset System Test Suite\n")

    all_passed = True

    # Run all tests
    try:
        if not test_discovery():
            all_passed = False
        if not test_loading():
            all_passed = False
        if not test_show_preset():
            all_passed = False
        if not test_registration():
            all_passed = False
        if not test_runner_initialization():
            all_passed = False
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False

    # Final summary
    if all_passed:
        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe preset system is working correctly:")
        print("  ✓ Preset discovery (list functions) works")
        print("  ✓ Preset loading works")
        print("  ✓ Show preset functionality works")
        print("  ✓ Custom preset registration works")
        print("  ✓ SimpleRunner initialization works")
        print("\n")
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("❌ SOME TESTS FAILED")
        print("=" * 80)
        print("\nPlease review the output above for details.\n")
        sys.exit(1)
