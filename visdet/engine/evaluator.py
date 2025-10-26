# ruff: noqa
"""
Evaluator module.

This module provides access to evaluator functionality for visdet.
"""

from typing import Any, Dict, List, Optional


class BaseMetric:
    """Base metric class for visdet.

    This is a minimal implementation for type checking.
    In a full implementation, this would provide comprehensive metrics.
    """

    metric_name = "metric"

    def __init__(self, collect_device: str = "cpu", prefix: Optional[str] = None) -> None:
        """Initialize base metric.

        Args:
            collect_device: Device to collect results on
            prefix: Prefix for metric names
        """
        self.collect_device = collect_device
        self.prefix = prefix

    def process(self, data_batch: Dict, data_samples: List) -> None:
        """Process a batch of data samples.

        Args:
            data_batch: Input batch data
            data_samples: Data samples with predictions
        """
        pass

    def compute_metrics(self, results: List) -> Dict:
        """Compute metrics from processed results.

        Args:
            results: List of processed results

        Returns:
            Dictionary of computed metrics
        """
        return {}


__all__ = ["BaseMetric"]
