# Copyright (c) OpenMMLab. All rights reserved.
# Create a simple registry for transforms
class Registry:
    """A simple registry to register transforms."""

    def __init__(self, name):
        self._name = name
        self._module_dict = {}

    def register_module(self, name=None, module=None, force=False):
        """Register a module.

        Args:
            name (str | None): The module name to be registered. If not
                specified, the class name will be used.
            module (type): Module class to be registered.
            force (bool): Whether to override an existing class with the same
                name. Default: False.
        """
        if module is not None:
            self._register_module(module_class=module, module_name=name, force=force)
            return module

        # use as a decorator
        def _register(cls):
            self._register_module(module_class=cls, module_name=name, force=force)
            return cls

        return _register

    def _register_module(self, module_class, module_name=None, force=False):
        if module_name is None:
            module_name = module_class.__name__
        if not force and module_name in self._module_dict:
            raise KeyError(f"{module_name} is already registered in {self._name}")
        self._module_dict[module_name] = module_class

    def get(self, key):
        """Get the registered module."""
        return self._module_dict.get(key, None)

    def build(self, cfg):
        """Build a module from config dict."""
        if isinstance(cfg, dict):
            cfg = cfg.copy()
            if "type" not in cfg:
                raise KeyError('cfg must contain the key "type"')
            module_type = cfg.pop("type")
            if module_type not in self._module_dict:
                raise KeyError(f"{module_type} is not in the {self._name} registry")
            module_cls = self._module_dict[module_type]
            return module_cls(**cfg)
        else:
            raise TypeError("cfg must be a dict")


TRANSFORMS = Registry("transforms")


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

    obj_type = cfg.pop("type")

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type} is not in the {registry._name} registry")
    else:
        obj_cls = obj_type

    return obj_cls(**cfg)


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
        from viscv.transforms.compose import Compose

        return Compose(transforms)
    else:
        return build_from_cfg(cfg, TRANSFORMS)
