"""Test configuration and fixtures for Modal GPU testing.

This module provides pytest markers, fixtures, and utilities for GPU-accelerated testing.
"""

import os
from pathlib import Path

import pytest
import torch


# Test markers
def pytest_configure(config):
    """Register pytest markers."""
    config.addinivalue_line(
        "markers",
        "gpu: mark test as requiring GPU",
    )
    config.addinivalue_line(
        "markers",
        "multi_gpu: mark test as requiring multiple GPUs",
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test",
    )
    config.addinivalue_line(
        "markers",
        "benchmark: mark test as performance benchmark",
    )


@pytest.fixture(scope="session")
def gpu_available() -> bool:
    """Check if GPU is available."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def gpu_count() -> int:
    """Get number of available GPUs."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Get path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def cmr_test_data() -> dict:
    """Get CMR test data configuration."""
    return {
        "data_root": "/home/georgepearse/data/",
        "train_ann": "cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json",
        "val_ann": "cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json",
        "img_prefix": "images/",
        "num_classes": 69,
    }


@pytest.fixture
def cuda_device():
    """Get CUDA device for testing."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        yield device
        # Cleanup
        torch.cuda.empty_cache()
    else:
        pytest.skip("CUDA not available")


@pytest.fixture
def torch_device():
    """Get torch device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GPUTestConfig:
    """Configuration for GPU tests."""

    # Training parameters
    BATCH_SIZE = 2
    NUM_EPOCHS = 2
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0005

    # Inference parameters
    INFERENCE_BATCH_SIZE = 4
    NUM_INFERENCE_BATCHES = 10

    # Benchmark parameters
    BENCHMARK_NUM_RUNS = 5
    BENCHMARK_WARMUP_RUNS = 2

    # Distributed training
    DIST_BACKEND = "nccl"
    DIST_URL = "env://"

    # Timeouts
    TRAINING_TIMEOUT = 600  # 10 minutes
    INFERENCE_TIMEOUT = 300  # 5 minutes
    BENCHMARK_TIMEOUT = 600  # 10 minutes
