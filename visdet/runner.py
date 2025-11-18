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

import copy
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
    preset configurations. It assembles a complete MMEngine-compatible config
    and uses the framework's runner to execute the training loop.

    Attributes:
        cfg: Full, merged MMEngine configuration object.

    Example:
        >>> # Simple usage with presets
        >>> runner = SimpleRunner(
        ...     model='mask_rcnn_swin_s',
        ...     dataset='cmr_instance_segmentation',
        ...     epochs=12
        ... )
        >>> runner.train()
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
        work_dir: str = "./work_dirs",
        epochs: int = 12,
        val_interval: int = 1,
        batch_size: int = 2,
        num_workers: int = 2,
        **kwargs,
    ):
        """Initialize the SimpleRunner.

        Args:
            model: Model preset name or config dict.
            dataset: Dataset preset name or config dict.
            optimizer: Optimizer preset name or config dict.
            scheduler: Scheduler preset name or config dict (optional).
            work_dir: Working directory for logs and checkpoints.
            epochs: Number of training epochs.
            val_interval: Validation interval in epochs.
            batch_size: Batch size for the dataloader.
            num_workers: Number of worker processes for the dataloader.
            **kwargs: Additional config overrides to be set at the top level.
        """
        # Resolve all presets to configs
        self.model_cfg = self._resolve_preset(model, MODEL_PRESETS, "model")
        self.dataset_cfg = self._resolve_preset(dataset, DATASET_PRESETS, "dataset")
        self.optimizer_cfg = self._resolve_preset(optimizer, OPTIMIZER_PRESETS, "optimizer")

        self.scheduler_cfg = None
        if scheduler:
            self.scheduler_cfg = self._resolve_preset(scheduler, SCHEDULER_PRESETS, "scheduler")

        self.work_dir = work_dir
        self.epochs = epochs
        self.val_interval = val_interval
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

        # Build the full config
        self._build_config()

    def _resolve_preset(self, value: Union[str, dict], registry, name: str) -> dict:
        """Resolve a preset string or dict to a config dict."""
        if isinstance(value, str):
            config = registry.get(value)
            if config is None:
                raise ValueError(f"Preset '{value}' not found in {name} presets.")
            return copy.deepcopy(config)
        elif isinstance(value, dict):
            if "_base_" in value:
                base_name = value["_base_"]
                base_config = registry.get(base_name)
                if base_config is None:
                    raise ValueError(f"Base preset '{base_name}' not found in {name} presets.")
                return self._deep_merge(copy.deepcopy(base_config), value)
            return copy.deepcopy(value)
        else:
            raise TypeError(f"{name} must be str or dict, got {type(value).__name__}")

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries."""
        result = base
        for key, value in override.items():
            if key == "_base_":
                continue
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _build_config(self) -> None:
        """Build a full MMEngine-compatible configuration from resolved presets."""
        from visdet.engine.config import Config

        # Automatically sync num_classes from dataset to model
        self._sync_num_classes()

        # --- Train Dataloader ---
        train_dataloader = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "persistent_workers": True,
            "sampler": {"type": "DefaultSampler", "shuffle": True},
            "dataset": self.dataset_cfg,
        }

        # --- Validation Dataloader & Evaluator (Conditional) ---
        val_dataloader = None
        val_evaluator = None
        val_dataset_cfg = copy.deepcopy(self.dataset_cfg)

        # MMDetection convention: look for a validation-specific annotation file
        # To enable validation, add `val_ann_file` to your dataset preset YAML.
        if "val_ann_file" in val_dataset_cfg:
            val_dataset_cfg["ann_file"] = val_dataset_cfg.pop("val_ann_file")

            # Use a different pipeline for validation if provided
            if "val_pipeline" in val_dataset_cfg:
                val_dataset_cfg["pipeline"] = val_dataset_cfg.pop("val_pipeline")
            else:  # Or remove augmentations from the training pipeline
                val_dataset_cfg["pipeline"] = [
                    p for p in val_dataset_cfg.get("pipeline", []) if p.get("type") not in ["RandomFlip"]
                ]

            val_dataloader = {
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "persistent_workers": True,
                "sampler": {"type": "DefaultSampler", "shuffle": False},
                "dataset": val_dataset_cfg,
            }

            # Assume CocoMetric for CocoDataset
            if val_dataset_cfg.get("type") == "CocoDataset":
                val_evaluator = {
                    "type": "CocoMetric",
                    "ann_file": str(Path(val_dataset_cfg["data_root"]) / val_dataset_cfg["ann_file"]),
                    "metric": ["bbox", "segm"],
                }

        # --- Assemble Final Config ---
        config_dict = {
            "default_scope": "visdet",
            "model": self.model_cfg,
            "work_dir": self.work_dir,
            "train_dataloader": train_dataloader,
            "optim_wrapper": {"optimizer": self.optimizer_cfg},
            "train_cfg": {
                "type": "EpochBasedTrainLoop",
                "max_epochs": self.epochs,
                "val_interval": self.val_interval,
            },
            "val_cfg": {"type": "ValLoop"} if val_dataloader else None,
            "val_dataloader": val_dataloader,
            "val_evaluator": val_evaluator,
            # Default hooks are essential for logging, checkpoints, etc.
            "default_hooks": {
                "timer": {"type": "IterTimerHook"},
                "logger": {"type": "LoggerHook", "interval": 50},
                "param_scheduler": {"type": "ParamSchedulerHook"},
                "checkpoint": {"type": "CheckpointHook", "interval": 1},
                "sampler_seed": {"type": "DistSamplerSeedHook"},
                "visualization": {"type": "DetVisualizationHook"},
            },
            **self.kwargs,
        }

        if self.scheduler_cfg:
            config_dict["param_scheduler"] = self.scheduler_cfg

        self.cfg = Config(config_dict)

    def _sync_num_classes(self) -> None:
        """Automatically sync num_classes from dataset metainfo to model config."""
        metainfo = self.dataset_cfg.get("metainfo", {})
        classes = metainfo.get("classes")

        if not classes:
            return

        num_classes = len(classes)
        print(f"Auto-detected {num_classes} classes from dataset metainfo.")

        # Safely update roi_head.bbox_head and roi_head.mask_head if they exist.
        if "roi_head" in self.model_cfg:
            roi_head = self.model_cfg["roi_head"]

            if "bbox_head" not in roi_head:
                roi_head["bbox_head"] = {}
            roi_head["bbox_head"]["num_classes"] = num_classes

            if "mask_head" not in roi_head:
                roi_head["mask_head"] = {}
            roi_head["mask_head"]["num_classes"] = num_classes

            print(f"Automatically set model num_classes to {num_classes}")

    def train(self) -> None:
        """Start training using the assembled configuration.

        This method delegates all building and training logic to the MMEngine
        Runner, which correctly instantiates components from the config.
        """
        # MMEngineRunner is imported here to avoid potential circular dependencies
        # and to ensure registries are populated first.
        from visdet.engine.registry import DefaultScope
        from visdet.engine.runner import Runner as MMEngineRunner

        # Ensure the 'visdet' scope is active for component registration.
        DefaultScope.get_instance("visdet", scope_name="visdet")

        print("Building runner from config...")
        runner = MMEngineRunner.from_cfg(self.cfg)

        print(f"Starting training for {self.epochs} epochs...")
        runner.train()

    # Discoverability class methods
    @classmethod
    def list_models(cls) -> list:
        """List all available model presets."""
        return MODEL_PRESETS.list()

    @classmethod
    def list_datasets(cls) -> list:
        """List all available dataset presets."""
        return DATASET_PRESETS.list()

    @classmethod
    def list_optimizers(cls) -> list:
        """List all available optimizer presets."""
        return OPTIMIZER_PRESETS.list()

    @classmethod
    def list_schedulers(cls) -> list:
        """List all available scheduler presets."""
        return SCHEDULER_PRESETS.list()

    @classmethod
    def list_pipelines(cls) -> list:
        """List all available pipeline presets."""
        return PIPELINE_PRESETS.list()

    @classmethod
    def show_preset(cls, name: str, category: str = "model") -> None:
        """Display preset configuration."""
        from visdet import presets

        presets.show_preset(name, category)


# Convenience alias for shorter import
Runner = SimpleRunner

__all__ = ["SimpleRunner", "Runner"]
