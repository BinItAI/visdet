"""Base Pydantic schema classes for visdet configuration.

This module provides the foundation for type-safe configuration management.
All component configs inherit from these base classes.

Key Benefits:
1. Validation/Type Safety - Catch config errors before runtime, clear error messages
2. Reduce Repetition - Factory functions, inheritance, composition
3. IDE Experience - Autocomplete, jump-to-definition, inline docs
4. Performance - Pydantic v2 uses Rust core, fast validation
"""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class VisdetBaseConfig(BaseModel):
    """Base class for all visdet configuration schemas.

    Provides common configuration and serialization behavior.

    Attributes:
        model_config: Pydantic configuration that:
            - extra='forbid': Catches typos by rejecting unknown fields
            - validate_default=True: Validates default values
            - use_enum_values=True: Serializes enums to their values

    Example:
        >>> class MyConfig(VisdetBaseConfig):
        ...     learning_rate: float = Field(gt=0, description="Learning rate")
        >>> cfg = MyConfig(learning_rate=0.001)
        >>> cfg.to_dict()
        {'learning_rate': 0.001}
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        use_enum_values=True,
        populate_by_name=True,
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for MMEngine compatibility.

        Returns:
            Dictionary representation of the config.
        """
        return self.model_dump(exclude_none=True)

    def to_yaml(self, path: str) -> None:
        """Export config to YAML file.

        Args:
            path: Output file path.
        """
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "VisdetBaseConfig":
        """Load config from YAML file.

        Args:
            path: Path to YAML file.

        Returns:
            Validated config instance.
        """
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


class ComponentConfig(VisdetBaseConfig):
    """Base class for registry-buildable component configurations.

    Components are classes registered in the visdet registry system.
    The 'type' field specifies which class to instantiate.

    Attributes:
        type: Registry key for the component class.

    Example:
        >>> class SwinConfig(ComponentConfig):
        ...     type: Literal['SwinTransformer'] = 'SwinTransformer'
        ...     embed_dims: int = 96
    """

    type: str = Field(..., description="Registry key for component class")


class BackboneConfig(ComponentConfig):
    """Base class for backbone network configurations.

    Backbones are feature extraction networks (e.g., ResNet, Swin Transformer).
    """

    pass


class NeckConfig(ComponentConfig):
    """Base class for neck configurations.

    Necks connect backbones to heads (e.g., FPN).
    """

    pass


class HeadConfig(ComponentConfig):
    """Base class for detection head configurations.

    Heads perform the final predictions (e.g., RPNHead, RoIHead).
    """

    pass


class LossConfig(ComponentConfig):
    """Base class for loss function configurations."""

    pass


class DatasetConfig(ComponentConfig):
    """Base class for dataset configurations."""

    pass


class TransformConfig(ComponentConfig):
    """Base class for data transform configurations."""

    pass


# Type aliases for optional configs (common pattern in MMEngine)
OptionalConfig = Optional[dict[str, Any]]
"""Type alias for optional component config dicts."""

ConfigList = list[dict[str, Any]]
"""Type alias for lists of component configs (e.g., data pipelines)."""
