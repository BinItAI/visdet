# Copyright (c) OpenMMLab. All rights reserved.
from visdet.engine.registry import TRANSFORMS
from visdet.engine.registry import build_from_cfg as engine_build_from_cfg


def _map_legacy_transform_args(cfg, transform_type):
    """Map legacy transform argument names to new names for backward compatibility.

    Args:
        cfg (dict): Transform config dict.
        transform_type (str): Name of the transform class.
    """
    # Mapping of old parameter names to new ones
    legacy_mappings = {
        "RandomFlip": {"flip_ratio": "prob"},
        # "Resize": Removed - now supports full img_scale API like original
        "Pad": {"pad_val": "pad_val"},  # Keep as-is but might need updates
    }

    if transform_type in legacy_mappings:
        mappings = legacy_mappings[transform_type]
        for old_name, new_name in mappings.items():
            if old_name in cfg and new_name not in cfg:
                cfg[new_name] = cfg.pop(old_name)


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Configuration dict. It should at least contain the key "type".
        registry (Registry): The registry to find the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        obj: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, but got {type(cfg)}")

    if "type" not in cfg:
        raise KeyError('cfg must contain the key "type"')

    cfg = cfg.copy()

    # Merge default arguments
    if default_args is not None:
        for k, v in default_args.items():
            cfg.setdefault(k, v)

    obj_type = cfg.get("type")

    # Map legacy args before building
    if isinstance(obj_type, str):
        _map_legacy_transform_args(cfg, obj_type)

    return engine_build_from_cfg(cfg, registry, default_args)


def build_transforms(cfg):
    """Build a transform or a sequence of transforms.

    Args:
        cfg (dict, list[dict]): Transform config or list of configs.

    Returns:
        transform: The transform or a composed transform.
    """
    if isinstance(cfg, list):
        transforms = []
        for transform_cfg in cfg:
            transform = build_from_cfg(transform_cfg, TRANSFORMS)
            transforms.append(transform)

        # Import Compose here to avoid circular imports
        from visdet.cv.transforms.wrappers import Compose

        return Compose(transforms)
    else:
        return build_from_cfg(cfg, TRANSFORMS)
