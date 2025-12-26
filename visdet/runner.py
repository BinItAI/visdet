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
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Optional, Union

from visdet.presets import (
    DATASET_PRESETS,
    MODEL_PRESETS,
    OPTIMIZER_PRESETS,
    PIPELINE_PRESETS,
    SCHEDULER_PRESETS,
)

logger = logging.getLogger(__name__)


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
        train_ann_file: Optional[str] = None,
        val_ann_file: Optional[str] = None,
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
            train_ann_file: Path to training annotation file (COCO format).
                Overrides 'ann_file' in dataset config.
            val_ann_file: Path to validation annotation file (COCO format).
                Overrides 'val_ann_file' in dataset config.
            **kwargs: Additional config overrides to be set at the top level.

        Example:
            >>> runner = SimpleRunner(
            ...     model='mask_rcnn_swin_s',
            ...     dataset='coco_instance_segmentation',
            ...     train_ann_file='/data/train.json',
            ...     val_ann_file='/data/val.json',
            ...     epochs=12
            ... )
            >>> runner.train()
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

        # Store annotation file parameters for dynamic specification
        self.train_ann_file = train_ann_file
        self.val_ann_file = val_ann_file

        # Validate annotation files exist before building config
        self._validate_annotation_files()

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

    def _validate_annotation_files(self) -> None:
        """Validate that annotation files exist if provided.

        Checks whether train_ann_file and val_ann_file (if provided) exist
        before training starts. This catches configuration errors early.

        Raises:
            FileNotFoundError: If annotation file doesn't exist.
        """
        if self.train_ann_file is not None:
            train_path = Path(self.train_ann_file)
            if not train_path.is_absolute():
                # Make relative to data_root if defined in dataset config
                data_root = self.dataset_cfg.get("data_root", "")
                if data_root:
                    train_path = Path(data_root) / train_path
            if not train_path.exists():
                raise FileNotFoundError(
                    f"Training annotation file not found: {train_path}\n"
                    f"Provided path: {self.train_ann_file}\n"
                    f"Ensure the file exists and the path is correct before training."
                )
            logger.info(f"Validated training annotation file: {train_path}")

        if self.val_ann_file is not None:
            val_path = Path(self.val_ann_file)
            if not val_path.is_absolute():
                # Make relative to data_root if defined in dataset config
                data_root = self.dataset_cfg.get("data_root", "")
                if data_root:
                    val_path = Path(data_root) / val_path
            if not val_path.exists():
                raise FileNotFoundError(
                    f"Validation annotation file not found: {val_path}\n"
                    f"Provided path: {self.val_ann_file}\n"
                    f"Ensure the file exists and the path is correct before training."
                )
            logger.info(f"Validated validation annotation file: {val_path}")

    def _extract_classes_from_annotation(self, ann_file: str) -> tuple[list[int], dict[int, str]]:
        """Extract category IDs and names from COCO annotation file.

        Parses the COCO format annotation file and extracts the categories field.
        This enables automatic detection of classes in dynamically provided
        annotation files, which may differ from the dataset preset's class definitions.

        Args:
            ann_file: Path to COCO annotation file (JSON format).

        Returns:
            Tuple of:
            - category_ids: List of category IDs sorted numerically
            - category_dict: Dict mapping category ID to name

        Raises:
            FileNotFoundError: If annotation file doesn't exist.
            json.JSONDecodeError: If annotation file is not valid JSON.
            KeyError: If annotation file doesn't have 'categories' field.
        """
        ann_path = Path(ann_file)
        if not ann_path.is_absolute():
            # Make relative to data_root if defined
            data_root = self.dataset_cfg.get("data_root", "")
            if data_root:
                ann_path = Path(data_root) / ann_path

        if not ann_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")

        with open(ann_path, "r") as f:
            annotation_data = json.load(f)

        if "categories" not in annotation_data:
            raise KeyError(f"Annotation file {ann_path} missing 'categories' field")

        categories = annotation_data["categories"]
        category_dict: dict[int, str] = {}

        for cat in categories:
            cat_id = cat["id"]
            cat_name = cat["name"]
            category_dict[cat_id] = cat_name

        # Return category IDs sorted numerically
        category_ids = sorted(category_dict.keys())
        return category_ids, category_dict

    def _merge_and_validate_classes(
        self,
        train_ids: list[int],
        train_dict: dict[int, str],
        val_ids: list[int] | None = None,
        val_dict: dict[int, str] | None = None,
    ) -> tuple[int, list[str]]:
        """Merge and validate classes from training and validation annotation files.

        Uses UNION approach: the model must be able to predict any class that appears
        in either the training or validation set. This ensures the model can handle
        all classes it may encounter during evaluation, even if some classes only
        appear in validation.

        Validates:
        - Category ID conflicts (same ID with different names across train/val)
        - Non-contiguous category IDs (e.g., [1, 2, 5] instead of [1, 2, 3])
        - Warns when validation-only classes exist (HIGH severity - model won't learn them)
        - Warns when training-only classes exist (MEDIUM severity - no validation metrics)

        Args:
            train_ids: Sorted list of category IDs from training annotation file.
            train_dict: Dict mapping category ID to name from training file.
            val_ids: Sorted list of category IDs from validation annotation file (optional).
            val_dict: Dict mapping category ID to name from validation file (optional).

        Returns:
            Tuple of:
            - num_classes: Number of unique classes (max ID + 1 for contiguous IDs)
            - class_names: List of class names indexed by category ID (None for missing IDs)

        Raises:
            ValueError: If category IDs are not contiguous or if there are ID conflicts.
        """
        # Start with training classes
        all_ids = set(train_ids)
        all_dict = copy.deepcopy(train_dict)

        # Merge with validation classes if provided
        val_only_ids = set()
        train_only_ids = set()

        if val_ids is not None and val_dict is not None:
            val_set = set(val_ids)
            train_set = set(train_ids)

            # Find classes only in validation
            val_only_ids = val_set - train_set

            # Find classes only in training
            train_only_ids = train_set - val_set

            # Check for ID conflicts (same ID with different names)
            for cat_id in val_set & train_set:
                if all_dict[cat_id] != val_dict[cat_id]:
                    raise ValueError(
                        f"Category ID {cat_id} has conflicting names: "
                        f"'{all_dict[cat_id]}' (train) vs '{val_dict[cat_id]}' (val). "
                        f"Category IDs must have consistent names across datasets."
                    )

            # Merge validation classes
            all_ids.update(val_ids)
            for cat_id in val_ids:
                if cat_id not in all_dict:
                    all_dict[cat_id] = val_dict[cat_id]

        # Warn about class mismatches
        if val_only_ids:
            val_only_names = [val_dict[cid] for cid in sorted(val_only_ids)]
            warnings.warn(
                f"HIGH: Validation has {len(val_only_ids)} classes not in training: "
                f"{val_only_names}. The model will not learn to predict these classes "
                f"during training. Consider adding training samples for these classes.",
                UserWarning,
                stacklevel=2,
            )

        if train_only_ids:
            train_only_names = [train_dict[cid] for cid in sorted(train_only_ids)]
            warnings.warn(
                f"MEDIUM: Training has {len(train_only_ids)} classes not in validation: "
                f"{train_only_names}. No validation metrics will be computed for these classes.",
                UserWarning,
                stacklevel=2,
            )

        # Validate contiguous category IDs
        sorted_ids = sorted(all_ids)
        if sorted_ids and sorted_ids != list(range(min(sorted_ids), max(sorted_ids) + 1)):
            raise ValueError(
                f"Category IDs are not contiguous: {sorted_ids}. "
                f"Category IDs must be consecutive integers starting from the minimum ID. "
                f"For example, [1, 2, 3, 4] is valid, but [1, 2, 4, 5] is not."
            )

        # Build class names list with proper indexing
        # For non-contiguous IDs starting at 0, use all IDs as-is
        # For IDs starting at 1, create list where index 0 is unused
        num_classes = len(all_ids)
        class_names: list[str | None] = [None] * len(all_ids)

        if sorted_ids:
            min_id = min(sorted_ids)
            for cat_id in sorted_ids:
                # If IDs start at 0, use cat_id as index directly
                # If IDs start at 1, shift index by -1
                index = cat_id if min_id == 0 else cat_id - min_id
                class_names[index] = all_dict[cat_id]

        return num_classes, class_names

    def _build_config(self) -> None:
        """Build a full MMEngine-compatible configuration from resolved presets."""
        from visdet.engine import Config

        # Automatically sync num_classes from dataset to model
        self._sync_num_classes()

        # Override train annotation file if provided dynamically
        if self.train_ann_file is not None:
            logger.info(f"Overriding training annotation file with: {self.train_ann_file}")
            self.dataset_cfg["ann_file"] = self.train_ann_file

        # Rename train_pipeline to pipeline for training dataset
        if "train_pipeline" in self.dataset_cfg:
            self.dataset_cfg["pipeline"] = self.dataset_cfg.pop("train_pipeline")

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

        # Priority hierarchy for validation annotation file:
        # 1. Explicit val_ann_file parameter (highest priority)
        # 2. Dataset preset's val_ann_file field (medium priority)
        # 3. No validation (lowest priority)
        has_validation = False

        if self.val_ann_file is not None:
            # User provided explicit validation annotation file
            logger.info(f"Overriding validation annotation file with: {self.val_ann_file}")
            val_dataset_cfg["ann_file"] = self.val_ann_file
            has_validation = True
        elif "val_ann_file" in val_dataset_cfg:
            # MMDetection convention: look for a validation-specific annotation file
            # To enable validation, add `val_ann_file` to your dataset preset YAML.
            logger.info(f"Using validation annotation file from preset: {val_dataset_cfg['val_ann_file']}")
            val_dataset_cfg["ann_file"] = val_dataset_cfg.pop("val_ann_file")
            has_validation = True

        if has_validation:
            # Use validation data prefix if provided, otherwise keep training prefix
            if "val_data_prefix" in val_dataset_cfg:
                val_dataset_cfg["data_prefix"] = val_dataset_cfg.pop("val_data_prefix")

            # Use test_pipeline for validation if provided, otherwise fall back to pipeline
            if "test_pipeline" in val_dataset_cfg:
                val_dataset_cfg["pipeline"] = val_dataset_cfg.pop("test_pipeline")
            elif "train_pipeline" in val_dataset_cfg:
                # Remove train_pipeline and use existing pipeline without augmentations
                val_dataset_cfg.pop("train_pipeline")
                val_dataset_cfg["pipeline"] = [
                    p for p in val_dataset_cfg.get("pipeline", []) if p.get("type") not in ["RandomFlip"]
                ]

            # Remove validation-specific keys that shouldn't be passed to dataset
            for key in ["val_ann_file", "val_data_prefix"]:
                val_dataset_cfg.pop(key, None)

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

        # Clean up non-dataset keys from configs
        # These are convenience keys in the preset that shouldn't be passed to the dataset
        keys_to_remove = [
            "val_ann_file",
            "val_data_prefix",
            "train_pipeline",
            "test_pipeline",
            "batch_size",
            "num_workers",
            "persistent_workers",
        ]
        for key in keys_to_remove:
            self.dataset_cfg.pop(key, None)
            val_dataset_cfg.pop(key, None) if val_dataset_cfg else None

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
        """Automatically sync num_classes from annotation files or dataset metainfo.

        Priority hierarchy:
        1. Annotation files (train_ann_file, val_ann_file) - parsed at runtime
        2. Dataset metainfo (from preset) - fallback if no annotation files provided
        3. Skip if no classes found - preserve existing model config

        When both train and val annotation files are provided, uses UNION of classes
        to ensure the model can predict any class that appears in either split.

        Supports both StandardRoIHead (single bbox_head dict) and CascadeRoIHead
        (multiple bbox_head stages as list).
        """
        num_classes = None
        source = None

        # Priority 1: Extract classes from annotation files if provided
        if self.train_ann_file is not None:
            train_ids, train_dict = self._extract_classes_from_annotation(self.train_ann_file)
            source = "training annotation file"

            # If validation annotation file is also provided, merge classes
            if self.val_ann_file is not None:
                val_ids, val_dict = self._extract_classes_from_annotation(self.val_ann_file)
                num_classes, class_names = self._merge_and_validate_classes(train_ids, train_dict, val_ids, val_dict)
                source = "training + validation annotation files (UNION)"
            else:
                # Only training file provided
                num_classes = len(train_ids)
                class_names = [train_dict[cid] for cid in sorted(train_ids)]

            logger.info(f"Auto-detected {num_classes} classes from {source}: {class_names}")

        # Priority 2: Fallback to dataset metainfo if no annotation files provided
        if num_classes is None:
            metainfo = self.dataset_cfg.get("metainfo", {})
            classes = metainfo.get("classes")

            if not classes:
                return

            num_classes = len(classes)
            source = "dataset metainfo (preset)"
            logger.info(f"Auto-detected {num_classes} classes from {source}")

        # Apply num_classes to model config
        # Two-stage detectors
        if "roi_head" in self.model_cfg:
            roi_head = self.model_cfg["roi_head"]

            # Handle StandardRoIHead: bbox_head is a dict
            if "bbox_head" in roi_head and isinstance(roi_head["bbox_head"], dict):
                roi_head["bbox_head"]["num_classes"] = num_classes
                if "mask_head" in roi_head and isinstance(roi_head["mask_head"], dict):
                    roi_head["mask_head"]["num_classes"] = num_classes
                logger.info(f"Automatically set model num_classes to {num_classes} (from {source})")

            # Handle CascadeRoIHead: bbox_head is a list of dicts (one per refinement stage)
            elif "bbox_head" in roi_head and isinstance(roi_head["bbox_head"], list):
                for bbox_head in roi_head["bbox_head"]:
                    bbox_head["num_classes"] = num_classes
                logger.info(
                    f"Automatically set CascadeRoIHead num_classes to {num_classes} "
                    f"across {len(roi_head['bbox_head'])} stages (from {source})"
                )

        # Single-stage detectors
        elif "bbox_head" in self.model_cfg:
            bbox_head = self.model_cfg["bbox_head"]
            if isinstance(bbox_head, dict) and "num_classes" in bbox_head:
                bbox_head["num_classes"] = num_classes
                logger.info(f"Automatically set model num_classes to {num_classes} (from {source})")
            elif isinstance(bbox_head, list):
                updated = False
                for head in bbox_head:
                    if isinstance(head, dict) and "num_classes" in head:
                        head["num_classes"] = num_classes
                        updated = True
                if updated:
                    logger.info(f"Automatically set model num_classes to {num_classes} (from {source})")

    def train(self) -> None:
        """Start training using the assembled configuration.

        This method delegates all building and training logic to the MMEngine
        Runner, which correctly instantiates components from the config.
        """
        # MMEngineRunner is imported here to avoid potential circular dependencies
        # and to ensure registries are populated first.
        from visdet.engine import DefaultScope
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
