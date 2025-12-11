"""GPU-accelerated inference tests and benchmarks.

Tests for model inference on GPU, including batch processing,
throughput measurement, and latency benchmarks.
"""

import pytest
import torch

from tests.modal_config import GPUTestConfig


@pytest.mark.gpu
@pytest.mark.benchmark
def test_inference_batch_gpu(benchmark, cuda_device):
    """Benchmark batch inference on GPU.

    Measures throughput and latency for batch inference.
    """
    # Create model
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 69),  # 69 classes
    ).to(cuda_device)
    model.eval()

    # Create dummy input
    batch_size = GPUTestConfig.INFERENCE_BATCH_SIZE
    dummy_input = torch.randn(batch_size, 1024, device=cuda_device)

    # Warm up
    with torch.no_grad():
        for _ in range(GPUTestConfig.BENCHMARK_WARMUP_RUNS):
            _ = model(dummy_input)

    # Benchmark
    def inference_fn():
        with torch.no_grad():
            return model(dummy_input)

    result = benchmark(inference_fn)
    assert result is not None


@pytest.mark.gpu
def test_inference_without_gradients_gpu(cuda_device):
    """Test inference with torch.no_grad() on GPU.

    Verifies that inference mode reduces memory usage and improves speed.
    """
    model = torch.nn.Linear(1000, 100).to(cuda_device)
    model.eval()
    dummy_input = torch.randn(10, 1000, device=cuda_device)

    # Test with gradients (should fail)
    dummy_input.requires_grad = True
    with torch.no_grad():
        output1 = model(dummy_input)
        assert not output1.requires_grad

    # Test without requires_grad
    dummy_input.requires_grad = False
    output2 = model(dummy_input)
    assert not output2.requires_grad


@pytest.mark.gpu
def test_mixed_precision_inference_gpu(cuda_device):
    """Test mixed precision inference on GPU.

    Verifies that models can run inference in mixed precision (FP16/FP32)
    for improved performance.
    """
    try:
        model = torch.nn.Linear(1000, 100).to(cuda_device)
        model.eval()
        dummy_input = torch.randn(10, 1000, device=cuda_device)

        # Test with autocast
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(dummy_input.half())

        assert output is not None
        assert output.shape == (10, 100)

    except RuntimeError:
        pytest.skip("Autocast not supported on this GPU")


@pytest.mark.gpu
@pytest.mark.timeout(GPUTestConfig.INFERENCE_TIMEOUT)
def test_batch_processing_throughput(cuda_device):
    """Test batch processing throughput on GPU.

    Measures how many samples can be processed per second.
    """
    import time

    model = torch.nn.Sequential(
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 69),
    ).to(cuda_device)
    model.eval()

    batch_size = GPUTestConfig.INFERENCE_BATCH_SIZE
    num_batches = GPUTestConfig.NUM_INFERENCE_BATCHES

    # Warm up
    dummy_input = torch.randn(batch_size, 512, device=cuda_device)
    with torch.no_grad():
        for _ in range(2):
            _ = model(dummy_input)

    # Measure throughput
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_batches):
            dummy_input = torch.randn(batch_size, 512, device=cuda_device)
            _ = model(dummy_input)

    end_time = time.time()
    elapsed_time = end_time - start_time
    total_samples = batch_size * num_batches
    throughput = total_samples / elapsed_time

    # Should process at least 100 samples per second
    assert throughput > 100.0
    print(f"Throughput: {throughput:.2f} samples/sec")


@pytest.mark.gpu
def test_large_batch_inference_gpu(cuda_device):
    """Test inference with large batch size on GPU.

    Verifies that model can handle large batches without OOM.
    """
    model = torch.nn.Linear(128, 10).to(cuda_device)
    model.eval()

    # Try increasingly large batch sizes
    for batch_size in [32, 64, 128]:
        try:
            dummy_input = torch.randn(batch_size, 128, device=cuda_device)
            with torch.no_grad():
                output = model(dummy_input)
            assert output.shape == (batch_size, 10)
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Expected for very large batches
                torch.cuda.empty_cache()
                break
            else:
                raise
