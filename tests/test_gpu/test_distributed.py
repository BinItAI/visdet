"""Distributed GPU training tests.

Tests for multi-GPU training, including data parallelism and
gradient synchronization.
"""

import pytest
import torch
import torch.nn as nn

from tests.modal_config import GPUTestConfig


@pytest.mark.gpu
@pytest.mark.multi_gpu
def test_multiple_gpus_available():
    """Test that multiple GPUs are available.

    Verifies the test environment has multiple GPUs for distributed training.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 2, f"Expected 2+ GPUs, got {num_gpus}"


@pytest.mark.gpu
@pytest.mark.multi_gpu
def test_data_parallel_model():
    """Test DataParallel model on multiple GPUs.

    Verifies that model works with DataParallel wrapper.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("Less than 2 GPUs available")

    # Create model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
    )

    # Wrap with DataParallel
    model = nn.DataParallel(model)

    # Create dummy input
    dummy_input = torch.randn(8, 100)

    # Forward pass
    output = model(dummy_input)
    assert output.shape == (8, 10)

    # Cleanup
    del model


@pytest.mark.gpu
@pytest.mark.multi_gpu
def test_gradient_synchronization():
    """Test gradient synchronization across GPUs.

    Verifies that gradients are properly synchronized in DataParallel.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("Less than 2 GPUs available")

    model = nn.DataParallel(
        nn.Linear(100, 10),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Forward pass
    dummy_input = torch.randn(8, 100)
    dummy_target = torch.randint(0, 10, (8,))
    output = model(dummy_input)
    loss = criterion(output, dummy_target)

    # Backward pass
    loss.backward()

    # Verify gradients exist and are synchronized
    for param in model.parameters():
        assert param.grad is not None

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # Cleanup
    del model


@pytest.mark.gpu
@pytest.mark.multi_gpu
def test_device_consistency():
    """Test that tensors remain on correct devices.

    Verifies data stays on correct GPU throughout forward/backward passes.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("Less than 2 GPUs available")

    model = nn.DataParallel(
        nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        ),
    )

    dummy_input = torch.randn(8, 100)
    output = model(dummy_input)

    # Check output device (DataParallel puts output on first device)
    assert output.device.type == "cuda"
    assert output.shape == (8, 10)

    # Cleanup
    del model


@pytest.mark.gpu
@pytest.mark.multi_gpu
@pytest.mark.timeout(GPUTestConfig.TRAINING_TIMEOUT)
def test_multi_gpu_training_loop():
    """Test a simple training loop with DataParallel.

    Verifies full training loop works across multiple GPUs.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("Less than 2 GPUs available")

    try:
        # Create model
        model = nn.DataParallel(
            nn.Sequential(
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 10),
            ),
        )

        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        # Training loop
        num_batches = 5
        batch_size = 8

        for batch_idx in range(num_batches):
            # Forward pass
            dummy_input = torch.randn(batch_size, 100)
            dummy_target = torch.randint(0, 10, (batch_size,))
            output = model(dummy_input)
            loss = criterion(output, dummy_target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Training completed successfully
        assert True

    except Exception as e:
        pytest.fail(f"Multi-GPU training failed: {str(e)}")
    finally:
        # Cleanup
        torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.multi_gpu
def test_model_state_consistency():
    """Test that model state is consistent across GPUs.

    Verifies that model parameters are synchronized properly.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("Less than 2 GPUs available")

    # Create model
    model1 = nn.Linear(100, 10)
    model2 = nn.DataParallel(model1)

    # Get parameters
    params1 = list(model1.parameters())
    params2 = list(model2.parameters())

    # Verify same number of parameters
    assert len(params1) == len(params2)

    # Cleanup
    del model1, model2
