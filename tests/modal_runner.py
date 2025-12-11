"""Modal GPU test runner for visdet.

This module defines Modal functions for running GPU-accelerated tests,
integration tests, performance benchmarks, and distributed training tests.

Modal (modal.com) provides serverless GPU compute for CI/CD workflows.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import modal

# Define Modal image with GPU support
gpu_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch",
    "torchvision",
    "pytest",
    "pytest-benchmark",
    "pytest-timeout",
    "pytest-cov",
    "pytest-xdist",
    "pydantic>=2.0.0",
    "pycocotools",
    "opencv-python",
    "rich",
    "tqdm",
)

app = modal.App("visdet-gpu-tests", image=gpu_image)

# Shared volume for test artifacts
test_artifacts = modal.Volume.from_name("test-artifacts", create_if_missing=True)


@app.function(
    gpu="t4",
    timeout=1800,  # 30 minutes
    volumes={"/artifacts": test_artifacts},
    retries=1,
)
def run_gpu_unit_tests() -> Dict[str, any]:
    """Run GPU-accelerated unit tests with PyTest.

    Returns:
        Dictionary with test results and metrics.
    """
    import os

    os.chdir("/root")
    result = {
        "test_type": "gpu_unit_tests",
        "start_time": time.time(),
        "artifacts_path": "/artifacts",
    }

    try:
        # Run GPU unit tests with markers
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_gpu/",
            "-v",
            "-k",
            "gpu",
            "--tb=short",
            "--timeout=300",
            "-x",  # Stop on first failure
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        result["returncode"] = proc.returncode
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        result["passed"] = proc.returncode == 0

        # Save test output
        output_file = Path("/artifacts/gpu_unit_tests_output.txt")
        output_file.write_text(proc.stdout + "\n" + proc.stderr)

    except Exception as e:
        result["error"] = str(e)
        result["passed"] = False

    result["end_time"] = time.time()
    return result


@app.function(
    gpu="t4",
    timeout=2400,  # 40 minutes for training
    volumes={"/artifacts": test_artifacts},
    retries=1,
)
def run_integration_tests(epochs: int = 2) -> Dict[str, any]:
    """Run integration tests with short training runs.

    Args:
        epochs: Number of epochs to train for.

    Returns:
        Dictionary with integration test results.
    """
    import os

    os.chdir("/root")
    result = {
        "test_type": "integration_tests",
        "epochs": epochs,
        "start_time": time.time(),
        "artifacts_path": "/artifacts",
    }

    try:
        # Run integration tests
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_gpu/test_training_gpu.py",
            "-v",
            "--tb=short",
            "--timeout=600",
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        result["returncode"] = proc.returncode
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        result["passed"] = proc.returncode == 0

        # Save test output
        output_file = Path("/artifacts/integration_tests_output.txt")
        output_file.write_text(proc.stdout + "\n" + proc.stderr)

    except Exception as e:
        result["error"] = str(e)
        result["passed"] = False

    result["end_time"] = time.time()
    return result


@app.function(
    gpu="t4",
    timeout=1800,  # 30 minutes
    volumes={"/artifacts": test_artifacts},
    retries=1,
)
def run_performance_benchmarks() -> Dict[str, any]:
    """Run performance benchmarks for inference and training.

    Returns:
        Dictionary with benchmark results including throughput and latency.
    """
    import os

    os.chdir("/root")
    result = {
        "test_type": "performance_benchmarks",
        "start_time": time.time(),
        "artifacts_path": "/artifacts",
        "benchmarks": {},
    }

    try:
        # Run benchmark tests
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_gpu/test_inference_gpu.py",
            "-v",
            "--benchmark-only",
            "--benchmark-json=/artifacts/benchmark_results.json",
            "--timeout=300",
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        result["returncode"] = proc.returncode
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        result["passed"] = proc.returncode == 0

        # Load benchmark results
        benchmark_file = Path("/artifacts/benchmark_results.json")
        if benchmark_file.exists():
            try:
                result["benchmarks"] = json.loads(benchmark_file.read_text())
            except json.JSONDecodeError:
                result["benchmarks"] = {}

        # Save test output
        output_file = Path("/artifacts/performance_benchmarks_output.txt")
        output_file.write_text(proc.stdout + "\n" + proc.stderr)

    except Exception as e:
        result["error"] = str(e)
        result["passed"] = False

    result["end_time"] = time.time()
    return result


@app.function(
    gpu="t4",
    count=2,  # Request 2 GPUs
    timeout=2400,  # 40 minutes
    volumes={"/artifacts": test_artifacts},
    retries=1,
)
def run_multi_gpu_tests() -> Dict[str, any]:
    """Run distributed training tests on multiple GPUs.

    Returns:
        Dictionary with multi-GPU test results.
    """
    import os

    os.chdir("/root")
    result = {
        "test_type": "multi_gpu_tests",
        "gpu_count": 2,
        "start_time": time.time(),
        "artifacts_path": "/artifacts",
    }

    try:
        # Run multi-GPU tests with DDP (Distributed Data Parallel)
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_gpu/test_distributed.py",
            "-v",
            "--tb=short",
            "--timeout=600",
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        result["returncode"] = proc.returncode
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        result["passed"] = proc.returncode == 0

        # Save test output
        output_file = Path("/artifacts/multi_gpu_tests_output.txt")
        output_file.write_text(proc.stdout + "\n" + proc.stderr)

    except Exception as e:
        result["error"] = str(e)
        result["passed"] = False

    result["end_time"] = time.time()
    return result


def run_all_tests(include_multi_gpu: bool = True) -> Dict[str, List[Dict]]:
    """Run all GPU tests and return results.

    Args:
        include_multi_gpu: Whether to run multi-GPU tests.

    Returns:
        Dictionary mapping test types to their results.
    """
    results = {}

    print("Starting GPU unit tests...")
    results["gpu_unit_tests"] = run_gpu_unit_tests.remote()

    print("Starting integration tests...")
    results["integration_tests"] = run_integration_tests.remote(epochs=2)

    print("Starting performance benchmarks...")
    results["performance_benchmarks"] = run_performance_benchmarks.remote()

    if include_multi_gpu:
        print("Starting multi-GPU tests...")
        results["multi_gpu_tests"] = run_multi_gpu_tests.remote()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Modal GPU tests for visdet",
    )
    parser.add_argument(
        "--test-type",
        choices=[
            "gpu-unit",
            "integration",
            "benchmarks",
            "multi-gpu",
            "all",
        ],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument(
        "--skip-multi-gpu",
        action="store_true",
        help="Skip multi-GPU tests",
    )

    args = parser.parse_args()

    if args.test_type == "gpu-unit":
        result = run_gpu_unit_tests.remote()
    elif args.test_type == "integration":
        result = run_integration_tests.remote()
    elif args.test_type == "benchmarks":
        result = run_performance_benchmarks.remote()
    elif args.test_type == "multi-gpu":
        result = run_multi_gpu_tests.remote()
    else:  # all
        result = run_all_tests(include_multi_gpu=not args.skip_multi_gpu)

    print(json.dumps(result, indent=2, default=str))
