#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""
Minimal setup.py for custom build logic.
All metadata and dependencies are now in pyproject.toml.
This file only handles:
1. MIM extension setup (symlink/copy configs and tools)
2. CUDA extension building (currently no extensions defined)
"""

import os
import os.path as osp
import platform
import shutil
import sys
import warnings

from setuptools import setup

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch not available. CUDA extensions will not be built.")


def make_cuda_ext(name, module, sources, sources_cuda=[]):
    """
    Create CUDA extension for compilation.

    Args:
        name: Extension name
        module: Module path
        sources: C++ source files
        sources_cuda: CUDA source files (optional)
    """
    if not TORCH_AVAILABLE:
        return None

    define_macros = []
    extra_compile_args = {"cxx": []}

    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
        define_macros += [("WITH_CUDA", None)]
        extension = CUDAExtension
        extra_compile_args["nvcc"] = [
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        sources += sources_cuda
    else:
        print(f"Compiling {name} without CUDA")
        extension = CppExtension

    return extension(
        name=f"{module}.{name}",
        sources=[os.path.join(*module.split("."), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


def add_mim_extension():
    """
    Add extra files that are required to support MIM into the package.

    These files will be added by creating a symlink to the originals if the
    package is installed in `editable` mode (e.g. pip install -e .), or by
    copying from the originals otherwise.

    TODO: Long-term, migrate this to package_data in pyproject.toml
    """
    # parse installment mode
    if "develop" in sys.argv:
        # installed by `pip install -e .`
        if platform.system() == "Windows":
            # set `copy` mode here since symlink fails on Windows.
            mode = "copy"
        else:
            mode = "symlink"
    elif "sdist" in sys.argv or "bdist_wheel" in sys.argv:
        # installed by `pip install .`
        # or create source distribution by `python setup.py sdist`
        mode = "copy"
    else:
        return

    filenames = ["tools", "configs", "demo", "model-index.yml"]
    repo_path = osp.dirname(__file__)
    mim_path = osp.join(repo_path, "mmdet", ".mim")
    os.makedirs(mim_path, exist_ok=True)

    for filename in filenames:
        if osp.exists(filename):
            src_path = osp.join(repo_path, filename)
            tar_path = osp.join(mim_path, filename)

            if osp.isfile(tar_path) or osp.islink(tar_path):
                os.remove(tar_path)
            elif osp.isdir(tar_path):
                shutil.rmtree(tar_path)

            if mode == "symlink":
                src_relpath = osp.relpath(src_path, osp.dirname(tar_path))
                os.symlink(src_relpath, tar_path)
            elif mode == "copy":
                if osp.isfile(src_path):
                    shutil.copyfile(src_path, tar_path)
                elif osp.isdir(src_path):
                    shutil.copytree(src_path, tar_path)
                else:
                    warnings.warn(f"Cannot copy file {src_path}.")
            else:
                raise ValueError(f"Invalid mode {mode}")


if __name__ == "__main__":
    # Add MIM extension files
    add_mim_extension()

    # Define CUDA extensions (currently none)
    # To add CUDA ops, uncomment and modify:
    # ext_modules = [
    #     make_cuda_ext(
    #         name="ops",
    #         module="mmdet._ext",
    #         sources=["src/ops.cpp"],
    #         sources_cuda=["src/ops_cuda.cu"],
    #     ),
    # ]
    ext_modules = []

    # Filter out None values (from missing torch)
    ext_modules = [ext for ext in ext_modules if ext is not None]

    # Build configuration
    cmdclass = {}
    if TORCH_AVAILABLE and ext_modules:
        cmdclass["build_ext"] = BuildExtension

    # Setup - all metadata comes from pyproject.toml
    setup(
        ext_modules=ext_modules,
        cmdclass=cmdclass,
    )
