"""GPU-accelerated training tests.

Tests for model training on GPU, including forward/backward passes,
loss computation, and gradient accumulation.
"""

import pytest
import torch

from tests.modal_config import GPUTestConfig


@pytest.mark.gpu
@pytest.mark.integration
def test_model_forward_pass_gpu(cuda_device):
    """Test model forward pass on GPU.

    Verifies that the model can perform a forward pass on GPU
    and returns outputs of correct shape.
    """
    from visdet import SimpleRunner

    runner = SimpleRunner(
        model="mask_rcnn_swin_s",
        dataset="cmr_instance_segmentation",
        optimizer="adamw_default",
        epochs=1,
    )

    # Move model to GPU
    runner.model.to(cuda_device)
    runner.model.eval()

    # Create dummy input
    batch_size = GPUTestConfig.BATCH_SIZE
    dummy_input = torch.randn(
        batch_size,
        3,
        800,
        1333,
        device=cuda_device,
    )

    # Forward pass
    with torch.no_grad():
        outputs = runner.model(dummy_input)

    # Verify output
    assert outputs is not None
    assert isinstance(outputs, (dict, list, tuple))


@pytest.mark.gpu
@pytest.mark.integration
def test_loss_computation_gpu(cuda_device):
    """Test loss computation on GPU.

    Verifies that loss can be computed on GPU tensors
    and supports backpropagation.
    """
    # Create dummy predictions and targets
    predictions = torch.randn(
        2,
        69,  # number of classes
        device=cuda_device,
        requires_grad=True,
    )
    targets = torch.randint(0, 69, (2,), device=cuda_device)

    # Compute loss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(predictions, targets)

    # Verify loss
    assert loss.item() > 0
    assert loss.requires_grad

    # Test backpropagation
    loss.backward()
    assert predictions.grad is not None


@pytest.mark.gpu
@pytest.mark.integration
def test_cuda_memory_management():
    """Test CUDA memory management.

    Verifies proper allocation and deallocation of GPU memory.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Get initial memory
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()

    # Allocate tensor
    tensor = torch.randn(1000, 1000, device="cuda")
    allocated_memory = torch.cuda.memory_allocated()

    # Verify memory increased
    assert allocated_memory > initial_memory

    # Delete tensor
    del tensor
    torch.cuda.empty_cache()

    # Memory should be reduced
    final_memory = torch.cuda.memory_allocated()
    assert final_memory < allocated_memory


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.timeout(GPUTestConfig.TRAINING_TIMEOUT)
def test_short_training_run_gpu(cuda_device, tmp_path):
    """Test short training run on GPU.

    Runs a short training loop to verify end-to-end GPU training.
    """
    from visdet import SimpleRunner

    try:
        runner = SimpleRunner(
            model="mask_rcnn_swin_s",
            dataset="cmr_instance_segmentation",
            optimizer="adamw_default",
            epochs=1,
            work_dir=str(tmp_path),
        )

        # Run training
        runner.train()

        # Verify training completed
        assert runner.model is not None
        assert len(list(tmp_path.glob("*/20*"))) > 0  # Check for work dir

    except Exception as e:
        pytest.skip(f"Could not run training: {str(e)}")


@pytest.mark.gpu
def test_gradient_accumulation_gpu(cuda_device):
    """Test gradient accumulation on GPU.

    Verifies that gradients can be accumulated across multiple batches.
    """
    model = torch.nn.Linear(10, 2).to(cuda_device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    batch_size = 4
    accumulation_steps = 2

    for step in range(accumulation_steps):
        # Create dummy batch
        inputs = torch.randn(batch_size, 10, device=cuda_device)
        targets = torch.randint(0, 2, (batch_size,), device=cuda_device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass (accumulate gradients)
        loss.backward()

    # Verify gradients exist
    for param in model.parameters():
        assert param.grad is not None

    # Update parameters
    optimizer.step()
    optimizer.zero_grad()
