"""Configuration system for visdet.

This module provides YAML-based configuration with:
- _base_ inheritance for composing experiment configs
- $ref resolution for component references
- Pydantic validation for type safety
"""

from .config_wrapper import Config, load_config
from .schema_generator import (
    generate_schema_from_class,
    generate_schemas_for_registry,
    get_schema_registry,
    schema_for,
    validate_config_with_schema,
)
from .yaml_loader import ConfigDict, YAMLConfigLoader, load_yaml_config

__all__ = [
    "Config",
    "load_config",
    "ConfigDict",
    "YAMLConfigLoader",
    "load_yaml_config",
    "schema_for",
    "generate_schema_from_class",
    "generate_schemas_for_registry",
    "validate_config_with_schema",
    "get_schema_registry",
]
