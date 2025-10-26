#!/usr/bin/env python3
"""
PyTorch GPU Verification Script

This script verifies that PyTorch can successfully access and use the GPU after
a NVIDIA driver update. It performs comprehensive checks including:
- CUDA availability
- GPU detection and properties
- Basic CUDA operations
- PyTorch version compatibility

Usage:
    python verify-pytorch-gpu.py

Returns:
    0 - All checks passed
    1 - One or more checks failed
"""

import sys
import warnings
from typing import Optional


# Color codes for terminal output
class Colors:
    GREEN = "\033[0;32m"
    RED = "\033[0;31m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    NC = "\033[0m"  # No Color


def print_success(message: str) -> None:
    """Print a success message with green checkmark."""
    print(f"{Colors.GREEN}✓{Colors.NC} {message}")


def print_error(message: str) -> None:
    """Print an error message with red X."""
    print(f"{Colors.RED}✗{Colors.NC} {message}")


def print_info(message: str) -> None:
    """Print an informational message with blue marker."""
    print(f"{Colors.BLUE}ℹ{Colors.NC} {message}")


def print_warning(message: str) -> None:
    """Print a warning message with yellow triangle."""
    print(f"{Colors.YELLOW}⚠{Colors.NC} {message}")


def check_torch_import() -> Optional[object]:
    """Check if PyTorch can be imported."""
    print("\n=== Checking PyTorch Installation ===")
    try:
        import torch

        print_success(f"PyTorch imported successfully (version: {torch.__version__})")
        return torch
    except ImportError as e:
        print_error(f"Failed to import PyTorch: {e}")
        return None


def check_cuda_availability(torch_module) -> bool:
    """Check if CUDA is available."""
    print("\n=== Checking CUDA Availability ===")
    try:
        is_available = torch_module.cuda.is_available()
        if is_available:
            print_success("CUDA is available")
            return True
        else:
            print_error("CUDA is not available")
            return False
    except Exception as e:
        print_error(f"Error checking CUDA availability: {e}")
        return False


def check_gpu_count(torch_module) -> int:
    """Check the number of available GPUs."""
    print("\n=== Checking GPU Count ===")
    try:
        count = torch_module.cuda.device_count()
        if count > 0:
            print_success(f"Detected {count} GPU(s)")
            return count
        else:
            print_error("No GPUs detected")
            return 0
    except Exception as e:
        print_error(f"Error checking GPU count: {e}")
        return 0


def check_gpu_properties(torch_module) -> bool:
    """Check properties of each available GPU."""
    print("\n=== Checking GPU Properties ===")
    try:
        count = torch_module.cuda.device_count()
        if count == 0:
            print_error("No GPUs detected")
            return False

        all_ok = True
        for i in range(count):
            try:
                name = torch_module.cuda.get_device_name(i)
                capability = torch_module.cuda.get_device_capability(i)
                total_memory = torch_module.cuda.get_device_properties(i).total_memory / (1024**3)
                print_success(
                    f"GPU {i}: {name} (Compute Capability: {capability[0]}.{capability[1]}, "
                    f"Memory: {total_memory:.1f}GB)"
                )
            except Exception as e:
                print_error(f"Error getting properties for GPU {i}: {e}")
                all_ok = False

        return all_ok
    except Exception as e:
        print_error(f"Error checking GPU properties: {e}")
        return False


def check_cuda_version(torch_module) -> bool:
    """Check CUDA version compatibility."""
    print("\n=== Checking CUDA Version ===")
    try:
        cuda_version = torch_module.version.cuda
        if cuda_version:
            print_success(f"CUDA Version: {cuda_version}")
            # Check if it's 12.8 or compatible
            major, minor = map(int, cuda_version.split(".")[:2])
            if major == 12 and minor >= 4:
                print_success(f"CUDA {major}.{minor} is compatible with PyTorch 2.8")
                return True
            elif major >= 13:
                print_success(f"CUDA {major}.{minor} is compatible with PyTorch 2.8")
                return True
            else:
                print_warning(f"CUDA {major}.{minor} may have limited support. Recommended: 12.4+")
                return True  # Don't fail on this, just warn
        else:
            print_warning("CUDA version not available")
            return False
    except Exception as e:
        print_error(f"Error checking CUDA version: {e}")
        return False


def check_cudnn_version(torch_module) -> bool:
    """Check cuDNN version."""
    print("\n=== Checking cuDNN Version ===")
    try:
        cudnn_version = torch_module.backends.cudnn.version()
        if cudnn_version:
            print_success(f"cuDNN Version: {cudnn_version}")
            return True
        else:
            print_warning("cuDNN version not available")
            return False
    except Exception as e:
        print_warning(f"cuDNN not available: {e}")
        return False


def test_cuda_tensor(torch_module) -> bool:
    """Test creating and using CUDA tensors."""
    print("\n=== Testing CUDA Tensor Operations ===")
    try:
        # Create a simple tensor on CPU and move to GPU
        cpu_tensor = torch_module.randn(100, 100)
        gpu_tensor = cpu_tensor.to("cuda")
        print_success("Successfully created tensor on GPU")

        # Perform a simple operation
        result = torch_module.matmul(gpu_tensor, gpu_tensor)
        print_success("Successfully performed matrix multiplication on GPU")

        # Move back to CPU
        result.cpu()
        print_success("Successfully transferred tensor back to CPU")

        return True
    except Exception as e:
        print_error(f"Error testing CUDA operations: {e}")
        return False


def test_pytorch_version(torch_module) -> bool:
    """Check PyTorch version compatibility with CUDA."""
    print("\n=== Checking PyTorch Version ===")
    try:
        version = torch_module.__version__
        print_success(f"PyTorch Version: {version}")

        # Check if version is 2.8 or compatible
        major, minor = map(int, version.split(".")[:2])
        if major >= 2 and minor >= 8:
            print_success("PyTorch version is 2.8 or newer (compatible)")
            return True
        else:
            print_warning(f"PyTorch {major}.{minor} may have limited features. Recommended: 2.8+")
            return True  # Don't fail, just warn
    except Exception as e:
        print_error(f"Error checking PyTorch version: {e}")
        return False


def main() -> int:
    """Main verification function."""
    print("\n" + "=" * 50)
    print("PyTorch GPU Verification")
    print("=" * 50)

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Check PyTorch import
    torch = check_torch_import()
    if torch is None:
        print_error("\nPyTorch is not installed. Cannot continue verification.")
        return 1

    # Check PyTorch version
    pytorch_ok = test_pytorch_version(torch)

    # Check CUDA availability
    cuda_ok = check_cuda_availability(torch)
    if not cuda_ok:
        print_error("\nCUDA is not available. GPU support is not functional.")
        return 1

    # Check GPU count
    gpu_count = check_gpu_count(torch)
    if gpu_count == 0:
        print_error("\nNo GPUs detected. Hardware may not be properly configured.")
        return 1

    # Check GPU properties
    properties_ok = check_gpu_properties(torch)

    # Check CUDA version
    cuda_version_ok = check_cuda_version(torch)

    # Check cuDNN version
    cudnn_ok = check_cudnn_version(torch)

    # Test CUDA operations
    cuda_ops_ok = test_cuda_tensor(torch)

    # Print summary
    print("\n" + "=" * 50)
    print("Verification Summary")
    print("=" * 50)

    checks = [
        ("PyTorch Installation", True),
        ("PyTorch Version", pytorch_ok),
        ("CUDA Availability", cuda_ok),
        ("GPU Detection", gpu_count > 0),
        ("GPU Properties", properties_ok),
        ("CUDA Version", cuda_version_ok),
        ("cuDNN Version", cudnn_ok),
        ("CUDA Operations", cuda_ops_ok),
    ]

    all_ok = True
    for check_name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        color = Colors.GREEN if result else Colors.RED
        print(f"{color}{status}{Colors.NC} {check_name}")
        if not result:
            all_ok = False

    print("=" * 50)

    if all_ok:
        print_success("\nAll checks passed! GPU is ready for PyTorch training.")
        return 0
    else:
        print_error("\nSome checks failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
