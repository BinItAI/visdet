# ruff: noqa
"""
Dataset module.

This module provides access to dataset functionality for visdet.
"""

from typing import Any, Dict, List, Optional
import torch


def pseudo_collate(data_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate batch data items into a single batch.

    This function takes a list of data items and collates them into a single
    batch by stacking tensors and grouping other data.

    Args:
        data_batch: List of data items, each with "inputs" and "data_samples" keys

    Returns:
        Collated batch dictionary with "inputs" as stacked tensor and
        "data_samples" as list of samples
    """
    inputs = []
    data_samples = []

    for data_item in data_batch:
        inputs.append(data_item.get("inputs"))
        data_samples.append(data_item.get("data_samples"))

    # Stack inputs into a batch tensor
    if inputs and inputs[0] is not None:
        inputs = torch.stack(inputs, dim=0)
    else:
        inputs = None

    return {"inputs": inputs, "data_samples": data_samples}


class BaseDataset:
    """Base dataset class for visdet.

    This is a minimal implementation for type checking.
    In a full implementation, this would provide comprehensive dataset features.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize dataset.

        Args:
            **kwargs: Dataset configuration arguments
        """
        pass

    def __len__(self) -> int:
        """Get dataset length."""
        return 0

    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Sample dictionary
        """
        return {}

    def get_cat_ids(self, idx: int) -> List:
        """Get category IDs for a sample.

        Args:
            idx: Sample index

        Returns:
            List of category IDs
        """
        return []


__all__ = ["BaseDataset", "pseudo_collate"]
