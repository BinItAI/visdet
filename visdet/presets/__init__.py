"""Preset management and discovery.

This module provides functions for:
- Listing available presets (models, datasets, optimizers, etc.)
- Viewing preset configurations
- Registering custom presets

Example:
    >>> from visdet import presets
    >>> presets.list_models()
    ['mask_rcnn_swin_s', 'faster_rcnn_r50', ...]
    >>> presets.show_preset('mask_rcnn_swin_s', category='model')
    >>> presets.register_model('my_model', {...})
"""

import yaml

from .registry import (
    DATASET_PRESETS,
    MODEL_PRESETS,
    OPTIMIZER_PRESETS,
    PIPELINE_PRESETS,
    SCHEDULER_PRESETS,
)


def list_models() -> list:
    """List all available model presets.

    Returns:
        Sorted list of model preset names
    """
    return MODEL_PRESETS.list()


def list_datasets() -> list:
    """List all available dataset presets.

    Returns:
        Sorted list of dataset preset names
    """
    return DATASET_PRESETS.list()


def list_optimizers() -> list:
    """List all available optimizer presets.

    Returns:
        Sorted list of optimizer preset names
    """
    return OPTIMIZER_PRESETS.list()


def list_schedulers() -> list:
    """List all available scheduler presets.

    Returns:
        Sorted list of scheduler preset names
    """
    return SCHEDULER_PRESETS.list()


def list_pipelines() -> list:
    """List all available pipeline presets.

    Returns:
        Sorted list of pipeline preset names
    """
    return PIPELINE_PRESETS.list()


def show_preset(name: str, category: str = "model") -> None:
    """Display preset configuration in YAML format.

    Args:
        name: Preset name
        category: Preset category ('model', 'dataset', 'optimizer', 'scheduler', 'pipeline')

    Raises:
        ValueError: If category is unknown or preset not found

    Example:
        >>> show_preset('mask_rcnn_swin_s', category='model')
        type: MaskRCNN
        backbone:
          type: SwinTransformer
          ...
    """
    registry_map = {
        "model": MODEL_PRESETS,
        "dataset": DATASET_PRESETS,
        "optimizer": OPTIMIZER_PRESETS,
        "scheduler": SCHEDULER_PRESETS,
        "pipeline": PIPELINE_PRESETS,
    }

    if category not in registry_map:
        raise ValueError(f"Unknown category: {category}. Valid categories: {', '.join(registry_map.keys())}")

    registry = registry_map[category]
    config = registry.get(name)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))


def register_model(name: str, config: dict) -> None:
    """Register a custom model preset.

    Args:
        name: Preset name
        config: Model configuration dictionary

    Example:
        >>> register_model('my_model', {
        ...     'type': 'MaskRCNN',
        ...     'backbone': {'type': 'ResNet', ...},
        ...     ...
        ... })
    """
    MODEL_PRESETS.register(name, config)


def register_dataset(name: str, config: dict) -> None:
    """Register a custom dataset preset.

    Args:
        name: Preset name
        config: Dataset configuration dictionary
    """
    DATASET_PRESETS.register(name, config)


def register_optimizer(name: str, config: dict) -> None:
    """Register a custom optimizer preset.

    Args:
        name: Preset name
        config: Optimizer configuration dictionary
    """
    OPTIMIZER_PRESETS.register(name, config)


def register_scheduler(name: str, config: dict) -> None:
    """Register a custom scheduler preset.

    Args:
        name: Preset name
        config: Scheduler configuration dictionary
    """
    SCHEDULER_PRESETS.register(name, config)


def register_pipeline(name: str, config: dict) -> None:
    """Register a custom pipeline preset.

    Args:
        name: Preset name
        config: Pipeline configuration dictionary
    """
    PIPELINE_PRESETS.register(name, config)


__all__ = [
    # Listing functions
    "list_models",
    "list_datasets",
    "list_optimizers",
    "list_schedulers",
    "list_pipelines",
    # Display function
    "show_preset",
    # Registration functions
    "register_model",
    "register_dataset",
    "register_optimizer",
    "register_scheduler",
    "register_pipeline",
    # Registry objects
    "MODEL_PRESETS",
    "DATASET_PRESETS",
    "OPTIMIZER_PRESETS",
    "SCHEDULER_PRESETS",
    "PIPELINE_PRESETS",
]
