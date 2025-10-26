"""Pydantic schema auto-generation from component __init__ signatures.

This module provides tools to automatically generate Pydantic models from
Python classes, with support for manual overrides via decorators.
"""

import inspect
from typing import Any, Callable, Dict, Optional, Type, Union, get_args, get_origin

from pydantic import BaseModel, Field, create_model


class SchemaRegistry:
    """Registry for Pydantic schemas mapped to component classes."""

    def __init__(self) -> None:
        """Initialize the schema registry."""
        self._schemas: Dict[Type, Type[BaseModel]] = {}
        self._manual_overrides: Dict[Type, Type[BaseModel]] = {}

    def register_schema(self, component_cls: Type, schema_cls: Type[BaseModel], is_manual: bool = False) -> None:
        """Register a Pydantic schema for a component class.

        Args:
            component_cls: The component class to register a schema for
            schema_cls: The Pydantic model schema
            is_manual: Whether this is a manually created schema (overrides auto-generated)
        """
        if is_manual:
            self._manual_overrides[component_cls] = schema_cls
        else:
            self._schemas[component_cls] = schema_cls

    def get_schema(self, component_cls: Type) -> Optional[Type[BaseModel]]:
        """Get the Pydantic schema for a component class.

        Args:
            component_cls: The component class to get the schema for

        Returns:
            The Pydantic model schema, or None if not registered
        """
        # Manual overrides take precedence
        if component_cls in self._manual_overrides:
            return self._manual_overrides[component_cls]
        return self._schemas.get(component_cls)

    def has_schema(self, component_cls: Type) -> bool:
        """Check if a schema exists for a component class.

        Args:
            component_cls: The component class to check

        Returns:
            True if a schema is registered
        """
        return component_cls in self._manual_overrides or component_cls in self._schemas


# Global schema registry
_schema_registry = SchemaRegistry()


def schema_for(component_cls: Type) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
    """Decorator to register a manual Pydantic schema for a component class.

    This decorator allows manual override of auto-generated schemas for cases
    where custom validation rules are needed.

    Args:
        component_cls: The component class this schema is for

    Returns:
        Decorator function

    Example:
        >>> from visdet.models.backbones.swin import SwinTransformer
        >>> @schema_for(SwinTransformer)
        ... class SwinTransformerConfig(BaseModel):
        ...     type: Literal['SwinTransformer'] = 'SwinTransformer'
        ...     embed_dims: int = Field(gt=0, description="Embedding dimensions")
        ...     depths: List[int] = Field(min_items=1)
    """

    def decorator(schema_cls: Type[BaseModel]) -> Type[BaseModel]:
        _schema_registry.register_schema(component_cls, schema_cls, is_manual=True)
        return schema_cls

    return decorator


def generate_schema_from_class(
    component_cls: Type,
    include_type_field: bool = True,
    type_literal: Optional[str] = None,
) -> Type[BaseModel]:
    """Auto-generate a Pydantic model from a class's __init__ signature.

    Args:
        component_cls: The class to generate a schema for
        include_type_field: Whether to add a 'type' field for registry lookup
        type_literal: Value for the type field (defaults to class name)

    Returns:
        Pydantic model class

    Example:
        >>> from visdet.models.backbones.swin import SwinTransformer
        >>> schema = generate_schema_from_class(SwinTransformer)
        >>> config = schema(embed_dims=96, depths=[2,2,6,2], num_heads=[3,6,12,24])
    """
    # Check if manual override exists
    existing_schema = _schema_registry.get_schema(component_cls)
    if existing_schema:
        return existing_schema

    # Inspect __init__ signature
    try:
        sig = inspect.signature(component_cls.__init__)
    except (ValueError, TypeError):
        # Fallback for classes without inspectable __init__
        return create_model(
            f"{component_cls.__name__}Config",
            __base__=BaseModel,
        )

    # Extract fields from signature
    fields: Dict[str, Any] = {}

    for param_name, param in sig.parameters.items():
        # Skip 'self' and special parameters
        if param_name in ("self", "args", "kwargs"):
            continue

        # Extract type annotation
        if param.annotation != inspect.Parameter.empty:
            param_type = param.annotation
        else:
            param_type = Any

        # Extract default value
        if param.default != inspect.Parameter.empty:
            # Has default value
            fields[param_name] = (param_type, param.default)
        else:
            # Required field (no default)
            fields[param_name] = (param_type, ...)

    # Add 'type' field for registry lookup
    if include_type_field:
        from typing import Literal

        type_value = type_literal or component_cls.__name__
        fields["type"] = (Literal[type_value], type_value)  # type: ignore

    # Create Pydantic model
    schema_cls = create_model(
        f"{component_cls.__name__}Config",
        __base__=BaseModel,
        **fields,  # type: ignore
    )

    # Register the auto-generated schema
    _schema_registry.register_schema(component_cls, schema_cls, is_manual=False)

    return schema_cls


def generate_schemas_for_registry(registry: Any, type_key: str = "type") -> Dict[str, Type[BaseModel]]:
    """Auto-generate Pydantic schemas for all components in a registry.

    Args:
        registry: MMEngine registry object (e.g., MODELS, DATASETS)
        type_key: Name of the type field for registry lookup

    Returns:
        Dict mapping component names to Pydantic schemas

    Example:
        >>> from visdet.registry import MODELS
        >>> schemas = generate_schemas_for_registry(MODELS)
        >>> swin_schema = schemas['SwinTransformer']
    """
    schemas: Dict[str, Type[BaseModel]] = {}

    # Get all registered components
    # Note: This assumes the registry has a _module_dict attribute
    # which is standard for MMEngine registries
    if hasattr(registry, "_module_dict"):
        for name, component_cls in registry._module_dict.items():
            schema = generate_schema_from_class(component_cls, include_type_field=True, type_literal=name)
            schemas[name] = schema

    return schemas


def validate_config_with_schema(
    config: Dict[str, Any], component_cls: Optional[Type] = None, type_name: Optional[str] = None
) -> BaseModel:
    """Validate a configuration dict using a Pydantic schema.

    Args:
        config: Configuration dictionary to validate
        component_cls: The component class (if known)
        type_name: The 'type' value from config (for registry lookup)

    Returns:
        Validated Pydantic model instance

    Raises:
        ValueError: If no schema can be found
        ValidationError: If config doesn't match schema

    Example:
        >>> config = {'type': 'SwinTransformer', 'embed_dims': 96, 'depths': [2,2,6,2]}
        >>> validated = validate_config_with_schema(config, type_name='SwinTransformer')
    """
    # Try to get schema by component class
    if component_cls:
        schema = _schema_registry.get_schema(component_cls)
        if schema:
            return schema(**config)

    # Try to get schema by type name
    if type_name:
        # This would require a reverse lookup in registries
        # For now, raise an error - this can be implemented later
        raise NotImplementedError(
            "Schema lookup by type name not yet implemented. Please provide component_cls instead."
        )

    raise ValueError(
        "Cannot validate config: no component_cls or type_name provided, or no schema registered for the component"
    )


# Convenience function for getting the global registry
def get_schema_registry() -> SchemaRegistry:
    """Get the global schema registry.

    Returns:
        Global SchemaRegistry instance
    """
    return _schema_registry
