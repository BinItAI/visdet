# ruff: noqa
"""
Visualization module.

This module provides access to visualization functionality for visdet.
"""

from typing import Any, Dict, Optional, Union
import numpy as np


class Visualizer:
    """Stub visualizer class for visdet.

    This is a minimal implementation for type checking.
    In a full implementation, this would provide comprehensive visualization features.
    """

    _instance: Optional["Visualizer"] = None

    def __init__(self, name: str = "visdet") -> None:
        """Initialize visualizer.

        Args:
            name: Visualizer name
        """
        self.name = name
        self._image = None
        self._vis_backends = {}

    @classmethod
    def get_current_instance(cls) -> Optional["Visualizer"]:
        """Get current visualizer instance.

        Returns:
            Current visualizer instance or None
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_image(self, image: np.ndarray) -> None:
        """Set image for visualization.

        Args:
            image: Image array
        """
        self._image = image

    def get_image(self) -> np.ndarray:
        """Get current image.

        Returns:
            Current image array
        """
        if self._image is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        return self._image

    def add_datasample(
        self,
        name: str,
        image: np.ndarray,
        data_sample: Any = None,
        show: bool = False,
        wait_time: float = 0,
        pred_score_thr: float = 0.3,
        step: int = 0,
        out_file: Optional[str] = None,
    ) -> None:
        """Add a data sample visualization.

        Args:
            name: Name of the visualization
            image: Image array
            data_sample: Data sample to visualize
            show: Whether to show the visualization
            wait_time: Time to wait (for show mode)
            pred_score_thr: Prediction score threshold
            step: Step number
            out_file: Output file path
        """
        self.set_image(image)

    def draw_bboxes(
        self,
        bboxes: Union[np.ndarray, list],
        edge_colors: Union[tuple, str] = "green",
        face_colors: Union[tuple, str] = "green",
        alpha: float = 0.8,
        **kwargs,
    ) -> None:
        """Draw bounding boxes on the image.

        Args:
            bboxes: Bounding boxes array
            edge_colors: Edge colors
            face_colors: Face colors
            alpha: Alpha value for transparency
        """
        pass

    def draw_texts(
        self,
        texts: Union[str, list],
        positions: Union[np.ndarray, list],
        colors: Union[tuple, str] = "white",
        font_sizes: Union[int, list] = 13,
        font_families: str = "sans-serif",
        bboxes: Optional[list] = None,
        **kwargs,
    ) -> None:
        """Draw text on the image.

        Args:
            texts: Text(s) to draw
            positions: Position(s) for text
            colors: Text color(s)
            font_sizes: Font size(s)
            font_families: Font family
            bboxes: Optional bounding box configs
        """
        pass

    def show(
        self,
        drawn_img: np.ndarray,
        win_name: str = "image",
        wait_time: float = 0,
    ) -> None:
        """Show the image.

        Args:
            drawn_img: Image to show
            win_name: Window name
            wait_time: Time to wait
        """
        pass


__all__ = ["Visualizer"]
