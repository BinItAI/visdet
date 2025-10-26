# Copyright (c) OpenMMLab. All rights reserved.
"""Legacy dataset utility functions for compatibility."""

import warnings


def get_loading_pipeline(pipeline):
    """Extract loading pipeline from a pipeline config.

    Args:
        pipeline (list[dict]): Pipeline configuration.

    Returns:
        list[dict]: Loading pipeline (LoadImageFromFile, LoadAnnotations, etc.)
    """
    loading_pipeline = []
    for transform in pipeline:
        if transform["type"].startswith("Load"):
            loading_pipeline.append(transform)
        else:
            break
    return loading_pipeline


def replace_ImageToTensor(pipelines):
    """Replace ImageToTensor with DefaultFormatBundle.

    Args:
        pipelines (list[dict]): Pipeline configuration.

    Returns:
        list[dict]: Pipeline with ImageToTensor replaced.
    """
    warnings.warn("ImageToTensor is deprecated, use DefaultFormatBundle instead", UserWarning)

    def _replace(pipeline):
        """Recursively replace ImageToTensor in pipeline."""
        if isinstance(pipeline, list):
            new_pipeline = []
            for item in pipeline:
                if isinstance(item, dict):
                    if item.get("type") == "ImageToTensor":
                        new_pipeline.append({"type": "DefaultFormatBundle"})
                    elif "transforms" in item:
                        # Recursively handle nested transforms
                        item = item.copy()
                        item["transforms"] = _replace(item["transforms"])
                        new_pipeline.append(item)
                    else:
                        new_pipeline.append(item)
                else:
                    new_pipeline.append(item)
            return new_pipeline
        return pipeline

    return _replace(pipelines)
