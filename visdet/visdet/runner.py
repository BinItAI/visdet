"""Simple runner API for training with string-based presets.

This module provides a user-friendly API for training models:

Example:
    >>> from visdet import SimpleRunner
    >>> runner = SimpleRunner(
    ...     model='mask_rcnn_swin_s',
    ...     dataset='coco_instance_segmentation',
    ...     optimizer='adamw_8bit',
    ...     scheduler='1cycle'
    ... )
    >>> runner.train()
"""

from pathlib import Path
from typing import Optional, Union

from visdet.presets import (
    DATASET_PRESETS,
    MODEL_PRESETS,
    OPTIMIZER_PRESETS,
    PIPELINE_PRESETS,
    SCHEDULER_PRESETS,
)


class SimpleRunner:
    """Simple, string-based runner API for training.

    This class provides a user-friendly interface for training models using
    preset configurations. Presets can be specified by name (string) or by
    passing a full configuration dictionary.

    Attributes:
        model_cfg: Resolved model configuration
        dataset_cfg: Resolved dataset configuration
        optimizer_cfg: Resolved optimizer configuration
        scheduler_cfg: Resolved scheduler configuration (optional)
        pipeline_cfg: Resolved pipeline configuration (optional)
        work_dir: Working directory for outputs
        cfg: Full merged configuration

    Example:
        >>> # Simple usage with presets
        >>> runner = SimpleRunner(
        ...     model='mask_rcnn_swin_s',
        ...     dataset='coco_instance_segmentation',
        ...     optimizer='adamw_8bit'
        ... )
        >>> runner.train()
        >>>
        >>> # Customization via dict with _base_
        >>> runner = SimpleRunner(
        ...     model={'_base_': 'mask_rcnn_swin_s', 'backbone': {'embed_dims': 128}},
        ...     dataset='coco_instance_segmentation'
        ... )
        >>>
        >>> # List available presets
        >>> SimpleRunner.list_models()
        ['mask_rcnn_swin_s', ...]
    """

    def __init__(
        self,
        model: Union[str, dict],
        dataset: Union[str, dict],
        optimizer: Union[str, dict] = "adamw_default",
        scheduler: Optional[Union[str, dict]] = None,
        pipeline: Optional[Union[str, dict]] = None,
        work_dir: str = "./work_dirs",
        epochs: int = 12,
        val_interval: int = 1,
        **kwargs,
    ):
        """Initialize the SimpleRunner.

        Args:
            model: Model preset name or config dict
            dataset: Dataset preset name or config dict
            optimizer: Optimizer preset name or config dict (default: 'adamw_default')
            scheduler: Scheduler preset name or config dict (optional)
            pipeline: Pipeline preset name or config dict (optional)
            work_dir: Working directory for logs and checkpoints
            epochs: Number of training epochs
            val_interval: Validation interval in epochs
            **kwargs: Additional config overrides
        """
        # Resolve all presets to configs
        self.model_cfg = self._resolve_preset(model, MODEL_PRESETS, "model")
        self.dataset_cfg = self._resolve_preset(dataset, DATASET_PRESETS, "dataset")
        self.optimizer_cfg = self._resolve_preset(optimizer, OPTIMIZER_PRESETS, "optimizer")

        self.scheduler_cfg = None
        if scheduler:
            self.scheduler_cfg = self._resolve_preset(scheduler, SCHEDULER_PRESETS, "scheduler")

        self.pipeline_cfg = None
        if pipeline:
            self.pipeline_cfg = self._resolve_preset(pipeline, PIPELINE_PRESETS, "pipeline")

        self.work_dir = work_dir
        self.epochs = epochs
        self.val_interval = val_interval
        self.kwargs = kwargs

        # Build the full config
        self._build_config()

    def _resolve_preset(self, value: Union[str, dict], registry, name: str) -> dict:
        """Resolve a preset string or dict to a config dict.

        Args:
            value: Preset name (string) or config dict
            registry: PresetRegistry to look up string names
            name: Parameter name (for error messages)

        Returns:
            Resolved configuration dictionary

        Raises:
            TypeError: If value is neither string nor dict
            ValueError: If preset name not found
        """
        if isinstance(value, str):
            # Look up preset by name
            return registry.get(value)
        elif isinstance(value, dict):
            # Support _base_ for customization
            if "_base_" in value:
                base_name = value["_base_"]
                base_config = registry.get(base_name)
                # Deep merge overrides
                return self._deep_merge(base_config, value)
            return value
        else:
            raise TypeError(f"{name} must be str or dict, got {type(value).__name__}")

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Override dictionary (takes precedence)

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key == "_base_":
                continue  # Skip the _base_ key itself

            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                result[key] = self._deep_merge(result[key], value)
            else:
                # Override takes precedence
                result[key] = value

        return result

    def _build_config(self) -> None:
        """Build full configuration from resolved presets."""
        from visdet.engine import Config

        # Merge all configs into one
        config_dict = {
            "model": self.model_cfg,
            "data": self.dataset_cfg,
            "optimizer": self.optimizer_cfg,
            "work_dir": self.work_dir,
            "total_epochs": self.epochs,
            "val_interval": self.val_interval,
            **self.kwargs,
        }

        if self.scheduler_cfg:
            config_dict["lr_schedule"] = self.scheduler_cfg

        if self.pipeline_cfg:
            # TODO: Merge pipeline into dataset transforms
            # This would require understanding the pipeline structure
            pass

        self.cfg = Config(config_dict)

    def train(self) -> None:
        """Start training.

        This method builds the model, dataset, and runner, then starts training.
        """
        from torch.utils.data import DataLoader

        from visdet.engine import Runner
        from visdet.registry import DATASETS, MODELS

        # Build model
        print("Building model...")
        model = MODELS.build(self.cfg.model)

        # Build dataset
        print("Building dataset...")
        train_dataset = DATASETS.build(self.cfg.data)

        # Build dataloader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.cfg.data.get("batch_size", 2),
            shuffle=True,
            num_workers=self.cfg.data.get("num_workers", 2),
            persistent_workers=self.cfg.data.get("persistent_workers", True),
        )

        # Create MMEngine Runner
        print("Creating runner...")
        runner = Runner(
            model=model,
            work_dir=self.cfg.work_dir,
            train_dataloader=train_dataloader,
            optim_wrapper=dict(optimizer=self.cfg.optimizer),
            train_cfg=dict(
                by_epoch=True,
                max_epochs=self.cfg.total_epochs,
                val_interval=self.cfg.val_interval,
            ),
        )

        # Start training
        print(f"Starting training for {self.cfg.total_epochs} epochs...")
        runner.train()

    # Discoverability class methods
    @classmethod
    def list_models(cls) -> list:
        """List all available model presets.

        Returns:
            Sorted list of model preset names
        """
        return MODEL_PRESETS.list()

    @classmethod
    def list_datasets(cls) -> list:
        """List all available dataset presets.

        Returns:
            Sorted list of dataset preset names
        """
        return DATASET_PRESETS.list()

    @classmethod
    def list_optimizers(cls) -> list:
        """List all available optimizer presets.

        Returns:
            Sorted list of optimizer preset names
        """
        return OPTIMIZER_PRESETS.list()

    @classmethod
    def list_schedulers(cls) -> list:
        """List all available scheduler presets.

        Returns:
            Sorted list of scheduler preset names
        """
        return SCHEDULER_PRESETS.list()

    @classmethod
    def list_pipelines(cls) -> list:
        """List all available pipeline presets.

        Returns:
            Sorted list of pipeline preset names
        """
        return PIPELINE_PRESETS.list()

    @classmethod
    def show_preset(cls, name: str, category: str = "model") -> None:
        """Display preset configuration.

        Args:
            name: Preset name
            category: Preset category ('model', 'dataset', 'optimizer', 'scheduler', 'pipeline')
        """
        from visdet import presets

        presets.show_preset(name, category)


# Convenience alias for shorter import
Runner = SimpleRunner

__all__ = ["SimpleRunner", "Runner"]
