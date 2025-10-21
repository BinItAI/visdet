# ruff: noqa
# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import logging
from typing import Any, Union

from visengine.config import Config, ConfigDict
from visengine.utils import ManagerMixin, digit_version

from .registry import Registry

from typing import TYPE_CHECKING, Any, Union

import torch.nn as nn

# Neither of these imports are actually neeed, they're just for type checking
# from mmengine.optim.scheduler import _ParamScheduler
# from mmengine.runner import Runner

import torch


def build_from_cfg(
    cfg: dict | ConfigDict | Config,
    registry: Registry,
    default_args: dict | ConfigDict | Config | None = None,
) -> Any:
    """Build a module from config dict when it is a class configuration, or
    call a function from config dict when it is a function configuration.

    If the global variable default scope (:obj:`DefaultScope`) exists,
    :meth:`build` will firstly get the responding registry and then call
    its own :meth:`build`.

    At least one of the ``cfg`` and ``default_args`` contains the key "type",
    which should be either str or class. If they all contain it, the key
    in ``cfg`` will be used because ``cfg`` has a high priority than
    ``default_args`` that means if a key exists in both of them, the value of
    the key will be ``cfg[key]``. They will be merged first and the key "type"
    will be popped up and the remaining keys will be used as initialization
    arguments.

    Examples:
        >>> from mmengine import Registry, build_from_cfg
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module(force=True)
        >>> class ResNet:
        >>>     def __init__(self, depth, stages=4):
        >>>         self.depth = depth
        >>>         self.stages = stages
        >>> cfg = dict(type='ResNet', depth=50)
        >>> model = build_from_cfg(cfg, MODELS)
        >>> # Returns an instantiated object
        >>> @MODELS.register_module(force=True)
        >>> def resnet50():
        >>>     pass
        >>> resnet = build_from_cfg(dict(type='resnet50'), MODELS)
        >>> # Return a result of the calling function

    Args:
        cfg (dict or ConfigDict or Config): Config dict. It should at least
            contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict or ConfigDict or Config, optional): Default
            initialization arguments. Defaults to None.

    Returns:
        object: The constructed object.
    """
    # Avoid circular import
    from ..logging import print_log

    if not isinstance(cfg, dict | ConfigDict | Config):
        raise TypeError(f"cfg should be a dict, ConfigDict or Config, but got {type(cfg)}")

    if "type" not in cfg:
        if default_args is None or "type" not in default_args:
            raise KeyError(f'`cfg` or `default_args` must contain the key "type", but got {cfg}\n{default_args}')

    if not isinstance(registry, Registry):
        raise TypeError(f"registry must be a mmengine.Registry object, but got {type(registry)}")

    if not (isinstance(default_args, dict | ConfigDict | Config) or default_args is None):
        raise TypeError(f"default_args should be a dict, ConfigDict, Config or None, but got {type(default_args)}")

    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    # Instance should be built under target scope, if `_scope_` is defined
    # in cfg, current default scope should switch to specified scope
    # temporarily.
    scope = args.pop("_scope_", None)
    with registry.switch_scope_and_registry(scope) as registry:
        obj_type = args.pop("type")
        if isinstance(obj_type, str):
            obj_cls = registry.get(obj_type)
            if obj_cls is None:
                raise KeyError(
                    f"{obj_type} is not in the {registry.scope}::{registry.name} registry. "
                    f"Please check whether the value of `{obj_type}` is "
                    "correct or it was registered as expected. More details "
                    "can be found at "
                    "https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#import-the-custom-module"
                )
        # this will include classes, functions, partial functions and more
        elif callable(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")

        # If `obj_cls` inherits from `ManagerMixin`, it should be
        # instantiated by `ManagerMixin.get_instance` to ensure that it
        # can be accessed globally.
        if inspect.isclass(obj_cls) and issubclass(obj_cls, ManagerMixin):  # type: ignore
            obj = obj_cls.get_instance(**args)  # type: ignore
        else:
            obj = obj_cls(**args)  # type: ignore

        if inspect.isclass(obj_cls) or inspect.isfunction(obj_cls) or inspect.ismethod(obj_cls):
            print_log(
                f"An `{obj_cls.__name__}` instance is built from "  # type: ignore
                "registry, and its implementation can be found in "
                f"{obj_cls.__module__}",  # type: ignore
                logger="current",
                level=logging.DEBUG,
            )
        else:
            print_log(
                f"An instance is built from registry, and its constructor is {obj_cls}",
                logger="current",
                level=logging.DEBUG,
            )
        return obj


def build_runner_from_cfg(cfg: dict | ConfigDict | Config, registry: Registry):  # -> Runner:
    """Build a Runner object.

    Examples:
        >>> from visengine.registry import Registry, build_runner_from_cfg
        >>> RUNNERS = Registry('runners', build_func=build_runner_from_cfg)
        >>> @RUNNERS.register_module(force=True)
        >>> class CustomRunner(Runner):
        >>>     def setup_env(env_cfg):
        >>>         pass
        >>> cfg = dict(runner_type='CustomRunner', ...)
        >>> custom_runner = RUNNERS.build(cfg)

    Args:
        cfg (dict or ConfigDict or Config): Config dict. If "runner_type" key
            exists, it will be used to build a custom runner. Otherwise, it
            will be used to build a default runner.
        registry (:obj:`Registry`): The registry to search the type from.

    Returns:
        object: The constructed runner object.
    """

    assert isinstance(cfg, dict | ConfigDict | Config), (
        f"cfg should be a dict, ConfigDict or Config, but got {type(cfg)}"
    )
    assert isinstance(registry, Registry), (
        "registry should be a mmengine.Registry object",
        f"but got {type(registry)}",
    )

    args = cfg.copy()
    # Runner should be built under target scope, if `_scope_` is defined
    # in cfg, current default scope should switch to specified scope
    # temporarily.
    scope = args.pop("_scope_", None)
    with registry.switch_scope_and_registry(scope) as registry:
        obj_type = args.get("runner_type", "Runner")
        if isinstance(obj_type, str):
            runner_cls = registry.get(obj_type)
            if runner_cls is None:
                raise KeyError(
                    f"{obj_type} is not in the {registry.name} registry. "
                    f"Please check whether the value of `{obj_type}` is "
                    "correct or it was registered as expected. More details "
                    "can be found at https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#import-the-custom-module"
                )
        elif inspect.isclass(obj_type):
            runner_cls = obj_type
        else:
            raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")

        runner = runner_cls.from_cfg(args)  # type: ignore
        print_log(
            f"An `{runner_cls.__name__}` instance is built from "  # type: ignore
            "registry, its implementation can be found in"
            f"{runner_cls.__module__}",  # type: ignore
            logger="current",
            level=logging.DEBUG,
        )
        return runner


def build_model_from_cfg(
    cfg: dict | ConfigDict | Config,
    registry: Registry,
    default_args: Union[dict, "ConfigDict", "Config"] | None = None,
) -> "nn.Module":
    """Build a PyTorch model from config dict(s). Different from
    ``build_from_cfg``, if cfg is a list, a ``nn.Sequential`` will be built.

    Args:
        cfg (dict, list[dict]): The config of modules, which is either a config
            dict or a list of config dicts. If cfg is a list, the built
            modules will be wrapped with ``nn.Sequential``.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn.Module.
    """
    from ..model import Sequential

    if isinstance(cfg, list):
        modules = [build_from_cfg(_cfg, registry, default_args) for _cfg in cfg]
        return Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_optimizer_from_cfg(
    cfg: dict | ConfigDict | Config,
    registry: Registry,
    default_args: dict | ConfigDict | Config | None = None,
) -> Any:
    if "type" in cfg and "Adafactor" == cfg["type"] and digit_version(torch.__version__) >= digit_version("2.5.0"):
        print_log("the torch version of Adafactor is registered as TorchAdafactor")
    return build_from_cfg(cfg, registry, default_args)


def build_scheduler_from_cfg(
    cfg: dict | ConfigDict | Config,
    registry: Registry,
    default_args: dict | ConfigDict | Config | None = None,
):  # -> "_ParamScheduler":
    """Builds a ``ParamScheduler`` instance from config.

    ``ParamScheduler`` supports building instance by its constructor or
    method ``build_iter_from_epoch``. Therefore, its registry needs a build
    function to handle both cases.

    Args:
        cfg (dict or ConfigDict or Config): Config dictionary. If it contains
            the key ``convert_to_iter_based``, instance will be built by method
            ``convert_to_iter_based``, otherwise instance will be built by its
            constructor.
        registry (:obj:`Registry`): The ``PARAM_SCHEDULERS`` registry.
        default_args (dict or ConfigDict or Config, optional): Default
            initialization arguments. It must contain key ``optimizer``. If
            ``convert_to_iter_based`` is defined in ``cfg``, it must
            additionally contain key ``epoch_length``. Defaults to None.

    Returns:
        object: The constructed ``ParamScheduler``.
    """
    assert isinstance(cfg, dict | ConfigDict | Config), (
        f"cfg should be a dict, ConfigDict or Config, but got {type(cfg)}"
    )
    assert isinstance(registry, Registry), (
        "registry should be a mmengine.Registry object",
        f"but got {type(registry)}",
    )

    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    scope = args.pop("_scope_", None)
    with registry.switch_scope_and_registry(scope) as registry:
        convert_to_iter = args.pop("convert_to_iter_based", False)
        if convert_to_iter:
            scheduler_type = args.pop("type")
            assert "epoch_length" in args and args.get("by_epoch", True), (
                "Only epoch-based parameter scheduler can be converted to iter-based, and `epoch_length` should be set"
            )
            if isinstance(scheduler_type, str):
                scheduler_cls = registry.get(scheduler_type)
                if scheduler_cls is None:
                    raise KeyError(
                        f"{scheduler_type} is not in the {registry.name} "
                        "registry. Please check whether the value of "
                        f"`{scheduler_type}` is correct or it was registered "
                        "as expected. More details can be found at https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#import-the-custom-module"
                    )
            elif inspect.isclass(scheduler_type):
                scheduler_cls = scheduler_type
            else:
                raise TypeError(f"type must be a str or valid type, but got {type(scheduler_type)}")
            return scheduler_cls.build_iter_from_epoch(**args)  # type: ignore
        else:
            args.pop("epoch_length", None)
            return build_from_cfg(args, registry)
