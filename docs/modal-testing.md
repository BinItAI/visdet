# Modal GPU Testing Guide

This document provides a comprehensive guide for using Modal to run GPU-accelerated tests in the visdet CI/CD pipeline.

## Overview

Modal is a serverless platform for running GPU-intensive workloads. We use it to run:

- **GPU unit tests**: Basic GPU functionality tests
- **Integration tests**: Full training runs on GPU
- **Performance benchmarks**: Inference throughput and latency measurements
- **Multi-GPU tests**: Distributed training with DataParallel

Tests are automatically triggered on pull requests labeled with `gpu-tests` or `modal-tests`, or can be run manually via GitHub Actions.

## Prerequisites

### For Local Development

1. **Install Modal CLI**:
   ```bash
   pip install modal>=0.63.0
   ```

2. **Set up Modal credentials**:
   ```bash
   modal token new
   ```
   This creates `~/.modal/token_file` with your credentials.

3. **Verify installation**:
   ```bash
   modal version
   ```

### For CI/CD (GitHub Actions)

1. **Get Modal API token**:
   - Log in to your Modal account dashboard
   - Navigate to Account Settings > API Tokens section
   - Create a new API token
   - Copy the `token_id` and `token_secret`

2. **Add GitHub secrets**:
   ```bash
   # In repository Settings > Secrets and variables > Actions
   MODAL_TOKEN_ID=<your_token_id>
   MODAL_TOKEN_SECRET=<your_token_secret>
   ```

## Running Tests Locally

### Run All GPU Tests

```bash
modal run tests.modal_runner --test-type all
```

### Run Specific Test Suite

```bash
# GPU unit tests only
modal run tests.modal_runner --test-type gpu-unit

# Integration tests (short training runs)
modal run tests.modal_runner --test-type integration

# Performance benchmarks
modal run tests.modal_runner --test-type benchmarks

# Multi-GPU tests
modal run tests.modal_runner --test-type multi-gpu
```

### Run Individual Test Files

```bash
# Using pytest directly on Modal
modal run tests.modal_runner
```

## Running Tests via GitHub Actions

### Automatic Trigger (PR Labels)

Tests automatically run when a PR is labeled with:
- `gpu-tests`: Run all GPU test suites
- `modal-tests`: Alias for `gpu-tests`

### Manual Trigger

1. Go to **Actions** > **Modal GPU Tests**
2. Click **Run workflow**
3. Select branch and run

### Workflow Configuration

The workflow (`.github/workflows/modal-gpu-tests.yml`) runs:

- **For Python 3.10, 3.11, 3.12**:
  - GPU unit tests (T4, 30 min timeout)
  - Integration tests (T4, 40 min timeout)
  - Performance benchmarks (T4, 20 min timeout)

- **For Python 3.12 only** (cost optimization):
  - Multi-GPU tests with 2x T4 (20 min timeout)

## Test Structure

### Test Files

```
tests/
├── test_gpu/
│   ├── __init__.py
│   ├── test_training_gpu.py         # Training & backprop tests
│   ├── test_inference_gpu.py        # Inference & benchmark tests
│   └── test_distributed.py          # Multi-GPU tests
├── modal_config.py                  # Pytest fixtures & configuration
└── modal_runner.py                  # Modal serverless functions
```

### Test Categories

#### Training Tests (`test_training_gpu.py`)

- `test_model_forward_pass_gpu`: Forward pass on GPU
- `test_loss_computation_gpu`: Loss computation and backpropagation
- `test_cuda_memory_management`: Memory allocation/deallocation
- `test_short_training_run_gpu`: Full 1-epoch training run
- `test_gradient_accumulation_gpu`: Multi-batch gradient accumulation

**Markers**: `@pytest.mark.gpu`, `@pytest.mark.integration`

#### Inference Tests (`test_inference_gpu.py`)

- `test_inference_batch_gpu`: Benchmark batch inference
- `test_inference_without_gradients_gpu`: `torch.no_grad()` optimization
- `test_mixed_precision_inference_gpu`: FP16 autocast inference
- `test_batch_processing_throughput`: Samples/sec measurement
- `test_large_batch_inference_gpu`: OOM handling

**Markers**: `@pytest.mark.gpu`, `@pytest.mark.benchmark`

#### Distributed Tests (`test_distributed.py`)

- `test_multiple_gpus_available`: GPU count assertion
- `test_data_parallel_model`: DataParallel wrapper
- `test_gradient_synchronization`: Cross-GPU gradient sync
- `test_device_consistency`: Tensor device placement
- `test_multi_gpu_training_loop`: 5-batch distributed training
- `test_model_state_consistency`: Parameter synchronization

**Markers**: `@pytest.mark.gpu`, `@pytest.mark.multi_gpu`

## Configuration

### GPU Resources

Tests use T4 GPUs by default. Configuration in `modal_runner.py`:

```python
@stub.function(
    gpu="T4",  # NVIDIA T4, ~$0.35/hr
    timeout=600,
)
```

Available GPU options:
- `T4`: $0.35/hr (budget, suitable for most tests)
- `A10G`: $1.05/hr (faster, for heavy compute)
- `H100`: $9.00/hr (enterprise only)

### Test Configuration

Modify `tests/modal_config.py` to adjust:

```python
class GPUTestConfig:
    BATCH_SIZE = 2              # Training batch size
    NUM_EPOCHS = 2              # Training epochs
    LEARNING_RATE = 0.001       # LR for training

    INFERENCE_BATCH_SIZE = 4    # Inference batch size
    NUM_INFERENCE_BATCHES = 10  # Batches for throughput test

    TRAINING_TIMEOUT = 600      # 10 minutes
    INFERENCE_TIMEOUT = 300     # 5 minutes
```

## Cost Estimation

### Per-Test Costs (T4 @ $0.35/hr)

| Test Suite | Duration | Cost |
|-----------|----------|------|
| GPU Unit Tests | ~10 min | ~$0.06 |
| Integration Tests | ~20 min | ~$0.12 |
| Benchmarks | ~5 min | ~$0.03 |
| Multi-GPU Tests (2x T4) | ~10 min | ~$0.12 |
| **Full Suite** | ~45 min | **~$0.33** |

### Optimization Strategies

1. **Use label-based triggers**: Tests only run on demand
2. **Python 3.12 only for multi-GPU**: Saves 2-3 T4 hours per PR
3. **Batch multiple tests**: Amortizes setup costs
4. **Use T4 instead of A10G**: $0.35/hr vs $1.05/hr

## Troubleshooting

### Common Issues

#### Modal CLI Not Found

```bash
# Reinstall Modal
pip install --upgrade modal

# Verify
modal version
```

#### "Invalid token" Error

```bash
# Regenerate token
modal token new

# Verify authentication
modal profile list
```

#### "CUDA not available" in Modal

Modal runs on `ubuntu-gpu-nvidia` image by default. If CUDA isn't available:

```python
@stub.function(
    image=modal.Image.ubuntu().run_commands(
        "apt-get update",
        "apt-get install -y nvidia-driver-535",
        "pip install torch torchvision",
    )
)
```

#### Test Timeout (> 600 seconds)

Increase timeout in `modal_runner.py`:

```python
@stub.function(gpu="T4", timeout=1200)  # 20 minutes
def run_gpu_unit_tests():
    ...
```

#### Out of Memory (OOM) on T4

T4 has 16GB VRAM. If OOM:

1. Reduce `BATCH_SIZE` in `modal_config.py`
2. Use A10G GPU instead (`gpu="A10G"`)
3. Enable gradient checkpointing in model

```python
# In model config
model.gradient_checkpointing = True
```

#### Artifacts Not Persisting

Modal volumes require explicit mounting:

```python
@stub.function(volumes={"/root/visdet": volume})
def run_tests():
    # Artifacts saved to /root/visdet/artifacts
    ...
```

### Debug Tips

1. **Enable verbose logging**:
   ```bash
   modal run --debug tests.modal_runner --test-type gpu-unit
   ```

2. **Check Modal dashboard**:
   - Log in to your Modal account and navigate to the Apps section
   - View live logs of running functions
   - Monitor resource usage and costs

3. **Run locally with pytest**:
   ```bash
   pytest tests/test_gpu/ -v -k "gpu"
   ```

4. **Capture full output**:
   ```bash
   modal run tests.modal_runner --test-type all 2>&1 | tee results.log
   ```

## Integration with CI/CD

### GitHub Actions Workflow

The workflow is triggered by:

1. **PR label**: Add `gpu-tests` or `modal-tests` label
2. **Manual dispatch**: Actions > Modal GPU Tests > Run workflow
3. **Schedule** (optional): Nightly full suite runs

Workflow steps:

```yaml
- Install dependencies
- Set up Modal credentials
- Run GPU tests (3 Python versions)
- Run multi-GPU tests (Python 3.12)
- Upload artifacts and results
- Comment on PR with summary
```

### PR Comments

After tests complete, the workflow posts a summary comment with:

- Test results (passed/failed)
- Performance metrics
- Artifact links
- Execution time and cost

## Best Practices

1. **Label PRs appropriately**:
   - Use `gpu-tests` only when GPU changes are made
   - Avoid running on every PR to save costs

2. **Monitor Modal dashboard**:
   - Check active functions
   - Review cost trends
   - Optimize slow tests

3. **Keep tests isolated**:
   - Each test should be independent
   - Clean up resources in fixtures
   - Use `pytest.skip()` for unavailable resources

4. **Document test dependencies**:
   - Specify required GPU count
   - Note timeout requirements
   - List data dependencies

## Advanced Usage

### Custom GPU Tiers

To use different GPUs for different tests:

```python
@stub.function(gpu="A10G")  # Faster GPU for intensive tests
def run_performance_benchmarks():
    ...

@stub.function(gpu="T4")    # Budget GPU for unit tests
def run_gpu_unit_tests():
    ...
```

### Parallel Test Execution

Modal supports parallel runs via `@stub.map()`:

```python
@stub.function()
def process_test_file(test_file: str):
    return subprocess.run(["pytest", test_file])

@stub.function()
def run_all_tests_parallel():
    test_files = ["test_training.py", "test_inference.py"]
    return process_test_file.map(test_files)
```

### Volume Persistence

Store large artifacts or datasets:

```python
volume = modal.Volume.persisted("test-artifacts")

@stub.function(volumes={"/mnt/data": volume})
def run_tests():
    # Artifacts persisted across runs
    os.makedirs("/mnt/data/results", exist_ok=True)
```

## Contributing

When adding new GPU tests:

1. **Place in appropriate file**:
   - `test_training_gpu.py`: Training/backprop tests
   - `test_inference_gpu.py`: Inference/benchmark tests
   - `test_distributed.py`: Multi-GPU tests

2. **Use fixtures from `modal_config.py`**:
   ```python
   def test_example(cuda_device, benchmark):
       model = MyModel().to(cuda_device)
       result = benchmark(lambda: model(dummy_input))
   ```

3. **Mark appropriately**:
   ```python
   @pytest.mark.gpu
   @pytest.mark.integration
   @pytest.mark.timeout(300)
   def test_my_feature():
       ...
   ```

4. **Commit and push** (pre-commit hooks will validate):
   ```bash
   git add tests/test_gpu/
   git commit -m "Add GPU test for feature X"
   git push
   ```

## Further Resources

- [Modal Documentation](https://modal.com/docs)
- [PyTest Documentation](https://docs.pytest.org/)
- [PyTorch GPU Documentation](https://pytorch.org/docs/stable/cuda.html)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
