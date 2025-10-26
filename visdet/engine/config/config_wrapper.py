"""Enhanced Config class with YAML support.

This module provides an enhanced Config class that extends visengine.Config
with support for YAML configuration files.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

from visdet.engine.config import Config as BaseConfig

from .schema_generator import validate_config_with_schema
from .yaml_loader import load_yaml_config


class Config(BaseConfig):
    """Enhanced Config class with YAML support and Pydantic validation.

    This class extends visengine.Config to support:
    1. Loading YAML files with _base_ inheritance and $ref resolution
    2. Optional Pydantic validation for type safety
    3. Backward compatibility with Python .py config files

    Example:
        >>> # Load YAML config
        >>> cfg = Config.fromfile('configs/experiments/mask_rcnn.yaml')
        >>>
        >>> # Load Python config (legacy)
        >>> cfg = Config.fromfile('configs/_base_/models/mask_rcnn_r50_fpn.py')
        >>>
        >>> # Load with validation
        >>> cfg = Config.fromfile('configs/experiments/mask_rcnn.yaml', validate=True)
    """

    @staticmethod
    def fromfile(
        filename: Union[str, Path],
        validate: bool = False,
        deprecation_warning: bool = True,
    ) -> "Config":
        """Load a config file (YAML or Python).

        Args:
            filename: Path to config file (.yaml or .py)
            validate: Whether to validate with Pydantic schemas
            deprecation_warning: Whether to show deprecation warning for .py configs

        Returns:
            Config instance

        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If file extension is not supported
            ValidationError: If validation fails (when validate=True)
        """
        filename = Path(filename)

        if not filename.exists():
            raise FileNotFoundError(f"Config file not found: {filename}")

        # Handle YAML files
        if filename.suffix in (".yaml", ".yml"):
            return Config._load_yaml_config(filename, validate=validate)

        # Handle Python files (legacy)
        elif filename.suffix == ".py":
            if deprecation_warning:
                warnings.warn(
                    f"Loading Python config files is deprecated and will be removed in a future version. "
                    f"Please migrate to YAML format. Use 'tools/convert_config.py' for automatic conversion. "
                    f"File: {filename}",
                    DeprecationWarning,
                    stacklevel=2,
                )
            # Use the parent class's fromfile method for .py files
            return super(Config, Config).fromfile(str(filename))

        else:
            raise ValueError(f"Unsupported config file extension: {filename.suffix}. Supported: .yaml, .yml, .py")

    @staticmethod
    def _load_yaml_config(filename: Path, validate: bool = False) -> "Config":
        """Load a YAML config file.

        Args:
            filename: Path to YAML config file
            validate: Whether to validate with Pydantic schemas

        Returns:
            Config instance
        """
        # Load YAML with _base_ and $ref resolution
        data = load_yaml_config(filename)

        # Optional Pydantic validation
        if validate:
            data = Config._validate_config(data)

        # Convert to Config object
        # Note: BaseConfig expects a dict, so we convert ConfigDict to dict
        return Config(dict(data))

    @staticmethod
    def _validate_config(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate config data with Pydantic schemas.

        Args:
            data: Configuration dictionary

        Returns:
            Validated configuration dictionary

        Note:
            This is a placeholder implementation. Full validation requires
            schemas for all component types. For now, we just validate
            components that have registered schemas.
        """
        # Validate model config if present
        if "model" in data and isinstance(data["model"], dict):
            model_type = data["model"].get("type")
            if model_type:
                try:
                    validated = validate_config_with_schema(data["model"], type_name=model_type)
                    data["model"] = validated.model_dump()
                except (ValueError, NotImplementedError):
                    # Schema not found or validation not implemented for this type
                    pass

        # Validate dataset config if present
        if "data" in data:
            # Similar validation for datasets
            pass

        # Additional validation can be added here for other components

        return data


def load_config(filename: Union[str, Path], validate: bool = False, deprecation_warning: bool = True) -> Config:
    """Convenience function to load a config file.

    Args:
        filename: Path to config file (.yaml or .py)
        validate: Whether to validate with Pydantic schemas
        deprecation_warning: Whether to show deprecation warning for .py configs

    Returns:
        Config instance

    Example:
        >>> from visdet.engine.config import load_config
        >>> cfg = load_config('configs/experiments/mask_rcnn.yaml')
    """
    return Config.fromfile(filename, validate=validate, deprecation_warning=deprecation_warning)
