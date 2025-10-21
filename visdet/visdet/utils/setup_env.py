# Copyright (c) OpenMMLab. All rights reserved.
"""Environment setup utilities for visdet."""

import datetime
import warnings

from visengine.registry import DefaultScope  # type: ignore


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in visdet into the registries.

    Args:
        init_default_scope (bool): Whether initialize the visdet default scope.
            When `init_default_scope=True`, the global default scope will be
            set to `visdet`, and all registries will build modules from visdet's
            registry node. To understand more about the registry, please refer
            to https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """
    # Import all modules to register them
    # These imports are already done in visdet/__init__.py but we ensure they're loaded
    import visdet.datasets  # type: ignore
    import visdet.engine  # type: ignore
    import visdet.evaluation  # type: ignore
    import visdet.models  # type: ignore
    import visdet.visualization  # type: ignore # noqa: F401

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None or not DefaultScope.check_instance_created("visdet")
        if never_created:
            DefaultScope.get_instance("visdet", scope_name="visdet")
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != "visdet":
            warnings.warn(
                "The current default scope "
                f'"{current_scope.scope_name}" is not "visdet", '
                "`register_all_modules` will force the current"
                'default scope to be "visdet". If this is not '
                "expected, please set `init_default_scope=False`.",
                stacklevel=2,
            )
            # avoid name conflict
            new_instance_name = f"visdet-{datetime.datetime.now()}"
            DefaultScope.get_instance(new_instance_name, scope_name="visdet")
