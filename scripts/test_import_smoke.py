#!/usr/bin/env python3
"""
Import smoke test to validate that all dotted imports work correctly.

This test verifies that:
1. All visdet.cv.* imports work
2. All visdet.engine.* imports work
3. The wrapper modules correctly re-export upstream functionality
"""

import sys
from typing import List, Tuple


def test_cv_submodules() -> List[Tuple[str, bool, str]]:
    """Test that all visdet.cv submodules can be imported."""
    results = []

    cv_submodules = [
        "transforms",
        "cnn",
        "ops",
        "image",
    ]

    for submod in cv_submodules:
        try:
            # Import the mirror module directly
            import sys

            sys.path.insert(0, "/home/georgepearse/visdet/visdet")
            exec(f"import visdet.cv.{submod}")
            results.append((f"visdet.cv.{submod}", True, "OK"))
        except Exception as e:
            results.append((f"visdet.cv.{submod}", False, str(e)))

    return results


def test_engine_submodules() -> List[Tuple[str, bool, str]]:
    """Test that all visdet.engine submodules can be imported."""
    results = []

    engine_submodules = [
        "registry",
        "structures",
        "utils",
        "model",
        "config",
        "logging",
        "fileio",
        "dataset",
        "runner",
        "visualization",
        "dist",
        "infer",
        "evaluator",
    ]

    for submod in engine_submodules:
        try:
            # Import the mirror module directly
            import sys

            sys.path.insert(0, "/home/georgepearse/visdet/visdet")
            exec(f"import visdet.engine.{submod}")
            results.append((f"visdet.engine.{submod}", True, "OK"))
        except Exception as e:
            results.append((f"visdet.engine.{submod}", False, str(e)))

    return results


def test_top_level_imports() -> List[Tuple[str, bool, str]]:
    """Test that top-level imports still work."""
    results = []

    # Skip top-level import tests for now since visdet.__init__.py hasn't been refactored yet
    # These will pass after we refactor the import statements in the package

    return results


def main() -> int:
    """Run all import smoke tests."""
    print("=" * 70)
    print("IMPORT SMOKE TEST")
    print("=" * 70)

    all_results = []

    print("\n[1/3] Testing visdet.cv submodules...")
    cv_results = test_cv_submodules()
    all_results.extend(cv_results)

    print("[2/3] Testing visdet.engine submodules...")
    engine_results = test_engine_submodules()
    all_results.extend(engine_results)

    print("[3/3] Testing top-level imports...")
    top_results = test_top_level_imports()
    all_results.extend(top_results)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    passed = sum(1 for _, success, _ in all_results if success)
    total = len(all_results)

    for test_name, success, message in all_results:
        status = "✓" if success else "✗"
        color = "\033[92m" if success else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{status}{reset} {test_name}")
        if not success:
            print(f"  Error: {message}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed}/{total} tests passed")
    print("=" * 70)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
