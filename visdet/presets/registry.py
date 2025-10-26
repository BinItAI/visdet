"""Preset registry system with lazy loading.

This module provides a registry for configuration presets that:
- Eagerly discovers preset names at import time (fast)
- Lazily loads actual configs on first access (performance)
- Supports custom preset registration
"""

from pathlib import Path
from typing import Dict, List, Optional, Union


class PresetRegistry:
    """Lazy-loading registry for configuration presets.

    The registry scans preset directories at initialization to discover
    available preset names, but only loads the actual YAML configs when
    they are first accessed. This provides fast imports with instant
    discoverability.

    Example:
        >>> registry = PresetRegistry(Path('configs/presets/models'))
        >>> registry.list()  # Fast - just returns names
        ['mask_rcnn_swin_s', 'faster_rcnn_r50']
        >>> config = registry.get('mask_rcnn_swin_s')  # Loads YAML on first access
    """

    def __init__(self, preset_dir: Union[str, Path]):
        """Initialize the preset registry.

        Args:
            preset_dir: Directory containing preset YAML files
        """
        self._preset_dir = Path(preset_dir)
        self._cache: Dict[str, dict] = {}
        self._available_names: Dict[str, Optional[Path]] = {}
        self._discover_presets()

    def _discover_presets(self) -> None:
        """Scan preset directory and populate available names.

        This is a fast I/O operation that only reads filenames, not file contents.
        """
        if not self._preset_dir.exists():
            return

        for yaml_file in self._preset_dir.glob("*.yaml"):
            preset_name = yaml_file.stem
            self._available_names[preset_name] = yaml_file

    def get(self, name: str) -> dict:
        """Get preset config by name (lazy load).

        Args:
            name: Preset name (without .yaml extension)

        Returns:
            Configuration dictionary

        Raises:
            ValueError: If preset name is not found
        """
        # Return from cache if already loaded
        if name in self._cache:
            return self._cache[name].copy()  # Return copy to prevent mutations

        # Check if preset exists
        if name not in self._available_names:
            available = ", ".join(self.list())
            raise ValueError(f"Preset '{name}' not found. Available presets: {available or '(none)'}")

        yaml_path = self._available_names[name]

        # If None, it's a programmatically registered preset (already in cache)
        if yaml_path is None:
            return self._cache[name].copy()

        # Load YAML using Config.fromfile (supports _base_ and config resolution)
        from visdet.engine.config import Config

        config = Config.fromfile(str(yaml_path))

        # Convert Config/ConfigDict to plain dict and cache
        self._cache[name] = dict(config)
        return self._cache[name].copy()

    def list(self) -> List[str]:
        """List all available preset names.

        Returns:
            Sorted list of preset names
        """
        return sorted(self._available_names.keys())

    def register(self, name: str, config: dict) -> None:
        """Register a custom preset programmatically.

        Args:
            name: Preset name
            config: Configuration dictionary

        Example:
            >>> registry.register('my_model', {'type': 'MaskRCNN', ...})
            >>> registry.get('my_model')  # Returns the registered config
        """
        self._cache[name] = config
        self._available_names[name] = None  # Mark as programmatic (no file)

    def has(self, name: str) -> bool:
        """Check if a preset exists.

        Args:
            name: Preset name

        Returns:
            True if preset exists
        """
        return name in self._available_names

    def __contains__(self, name: str) -> bool:
        """Check if preset exists (supports 'in' operator).

        Args:
            name: Preset name

        Returns:
            True if preset exists
        """
        return self.has(name)

    def __repr__(self) -> str:
        """String representation."""
        count = len(self._available_names)
        cached = len(self._cache)
        return f"PresetRegistry(dir={self._preset_dir}, presets={count}, cached={cached})"


# Global preset registries
# These are initialized at import time with eager name discovery


def _get_preset_root() -> Path:
    """Get the root configs/presets directory."""
    # visdet/visdet/presets/registry.py → visdet/ (repo root)
    return Path(__file__).parent.parent.parent.parent / "configs" / "presets"


_PRESET_ROOT = _get_preset_root()

# Initialize global registries for each preset category
MODEL_PRESETS = PresetRegistry(_PRESET_ROOT / "models")
DATASET_PRESETS = PresetRegistry(_PRESET_ROOT / "datasets")
OPTIMIZER_PRESETS = PresetRegistry(_PRESET_ROOT / "optimizers")
SCHEDULER_PRESETS = PresetRegistry(_PRESET_ROOT / "schedulers")
PIPELINE_PRESETS = PresetRegistry(_PRESET_ROOT / "pipelines")

__all__ = [
    "PresetRegistry",
    "MODEL_PRESETS",
    "DATASET_PRESETS",
    "OPTIMIZER_PRESETS",
    "SCHEDULER_PRESETS",
    "PIPELINE_PRESETS",
]
