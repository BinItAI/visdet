"""YAML configuration loader with _base_ inheritance and $ref resolution.

This module provides a custom YAML loader that supports:
1. _base_ inheritance: Merge configurations from parent YAML files
2. $ref resolution: Reference other YAML files for component composition

Execution order:
1. Load YAML file
2. Recursively merge all _base_ files (depth-first)
3. Recursively resolve all $ref references
4. Return final merged configuration dict
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


class ConfigDict(dict):
    """Dictionary subclass that allows attribute-style access.

    This matches the behavior of MMEngine's ConfigDict for compatibility.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")


class YAMLConfigLoader:
    """Custom YAML loader with _base_ and $ref support."""

    REF_KEY = "$ref"
    BASE_KEY = "_base_"

    def __init__(self) -> None:
        """Initialize the YAML config loader."""
        self._loading_stack: List[Path] = []

    def load(self, filepath: Union[str, Path]) -> ConfigDict:
        """Load a YAML config file with _base_ and $ref support.

        Args:
            filepath: Path to the YAML config file

        Returns:
            ConfigDict with all inheritance and references resolved

        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If circular dependencies are detected
        """
        filepath = Path(filepath).resolve()

        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        # Check for circular dependencies
        if filepath in self._loading_stack:
            cycle = " -> ".join(str(p) for p in self._loading_stack + [filepath])
            raise ValueError(f"Circular dependency detected: {cycle}")

        self._loading_stack.append(filepath)

        try:
            # Load the YAML file
            with open(filepath, "r") as f:
                data = yaml.safe_load(f) or {}

            # Step 1: Process _base_ inheritance (depth-first merge)
            if self.BASE_KEY in data:
                data = self._merge_base_configs(data, filepath)

            # Step 2: Recursively resolve $ref references
            data = self._resolve_references(data, filepath)

            # Convert to ConfigDict for attribute access
            data = self._dict_to_config(data)

            return data
        finally:
            self._loading_stack.pop()

    def _merge_base_configs(self, data: Dict[str, Any], current_file: Path) -> Dict[str, Any]:
        """Merge configurations from _base_ files.

        Args:
            data: Current config dict containing _base_ key
            current_file: Path to the current config file (for resolving relative paths)

        Returns:
            Merged configuration dict
        """
        base_files = data.pop(self.BASE_KEY)

        # Normalize to list
        if isinstance(base_files, str):
            base_files = [base_files]
        elif not isinstance(base_files, list):
            raise ValueError(f"_base_ must be a string or list, got {type(base_files)}")

        # Load and merge all base configs
        merged = {}
        for base_file in base_files:
            base_path = self._resolve_path(base_file, current_file)
            base_config = self.load(base_path)
            merged = self._deep_merge(merged, base_config)

        # Merge current config on top (overrides base)
        merged = self._deep_merge(merged, data)

        return merged

    def _resolve_references(self, data: Any, current_file: Path) -> Any:
        """Recursively resolve $ref references in the config.

        Args:
            data: Config data (dict, list, or primitive)
            current_file: Path to the current config file (for resolving relative paths)

        Returns:
            Data with all $ref references resolved
        """
        if isinstance(data, dict):
            # Check if this dict is a $ref
            if self.REF_KEY in data:
                if len(data) > 1:
                    raise ValueError(f"$ref must be the only key in a dict, got: {list(data.keys())}")
                ref_path = data[self.REF_KEY]
                resolved_path = self._resolve_path(ref_path, current_file)
                # Load the referenced file (this will recursively handle its _base_ and $ref)
                return self.load(resolved_path)
            else:
                # Recursively process all values
                return {k: self._resolve_references(v, current_file) for k, v in data.items()}
        elif isinstance(data, list):
            # Recursively process list items
            return [self._resolve_references(item, current_file) for item in data]
        else:
            # Primitive value, return as-is
            return data

    def _resolve_path(self, ref_path: str, current_file: Path) -> Path:
        """Resolve a file path relative to the current config file.

        Args:
            ref_path: Path to resolve (can be relative or absolute)
            current_file: Path to the current config file

        Returns:
            Absolute resolved path
        """
        ref_path = Path(ref_path)

        if ref_path.is_absolute():
            return ref_path
        else:
            # Resolve relative to the directory containing current_file
            return (current_file.parent / ref_path).resolve()

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence.

        Args:
            base: Base dictionary
            override: Dictionary with values that override base

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                result[key] = self._deep_merge(result[key], value)
            else:
                # Override takes precedence
                result[key] = value

        return result

    def _dict_to_config(self, data: Any) -> Any:
        """Recursively convert dicts to ConfigDict for attribute access.

        Args:
            data: Data to convert (dict, list, or primitive)

        Returns:
            Data with all dicts converted to ConfigDict
        """
        if isinstance(data, dict):
            return ConfigDict({k: self._dict_to_config(v) for k, v in data.items()})
        elif isinstance(data, list):
            return [self._dict_to_config(item) for item in data]
        else:
            return data


# Global loader instance
_loader = YAMLConfigLoader()


def load_yaml_config(filepath: Union[str, Path]) -> ConfigDict:
    """Load a YAML config file with _base_ and $ref support.

    This is the primary function users should call to load YAML configs.

    Args:
        filepath: Path to the YAML config file

    Returns:
        ConfigDict with all inheritance and references resolved

    Example:
        >>> cfg = load_yaml_config('configs/experiments/mask_rcnn.yaml')
        >>> print(cfg.model.type)
        'MaskRCNN'
        >>> print(cfg.train_cfg.max_epochs)
        12
    """
    return _loader.load(filepath)
