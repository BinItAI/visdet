import copy
import inspect

import torch
import torch.nn as nn

from visengine.config import Config, ConfigDict
from visengine.registry import OPTIM_WRAPPER_CONSTRUCTORS, OPTIMIZERS

from .optimizer_wrapper import OptimWrapper


def register_torch_optimizers() -> list[str]:
    """Register optimizers in ``torch.optim`` to the ``OPTIMIZERS`` registry.

    Returns:
        List[str]: A list of registered optimizers' name.
    """
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith("__"):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim, torch.optim.Optimizer):
            if module_name == "Adafactor":
                OPTIMIZERS.register_module(name="TorchAdafactor", module=_optim)
            else:
                OPTIMIZERS.register_module(module=_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers


TORCH_OPTIMIZERS = register_torch_optimizers()


def register_torch_npu_optimizers() -> list[str]:
    """Register optimizers in ``torch npu`` to the ``OPTIMIZERS`` registry.

    Returns:
        List[str]: A list of registered optimizers' name.
    """
    return []


NPU_OPTIMIZERS = []


def register_dadaptation_optimizers() -> list[str]:
    """Register optimizers in ``dadaptation`` to the ``OPTIMIZERS`` registry.

    Returns:
        List[str]: A list of registered optimizers' name.
    """
    dadaptation_optimizers = []
    try:
        import dadaptation
    except ImportError:
        pass
    else:
        for module_name in ["DAdaptAdaGrad", "DAdaptAdam", "DAdaptSGD"]:
            _optim = getattr(dadaptation, module_name)
            if inspect.isclass(_optim) and issubclass(_optim, torch.optim.Optimizer):
                OPTIMIZERS.register_module(module=_optim)
                dadaptation_optimizers.append(module_name)
    return dadaptation_optimizers


DADAPTATION_OPTIMIZERS = register_dadaptation_optimizers()


def register_lion_optimizers() -> list[str]:
    """Register Lion optimizer to the ``OPTIMIZERS`` registry.

    Returns:
        List[str]: A list of registered optimizers' name.
    """
    optimizers = []
    try:
        from lion_pytorch import Lion
    except ImportError:
        pass
    else:
        OPTIMIZERS.register_module(module=Lion)
        optimizers.append("Lion")
    return optimizers


LION_OPTIMIZERS = register_lion_optimizers()


def register_sophia_optimizers() -> list[str]:
    """Register Sophia optimizer to the ``OPTIMIZERS`` registry.

    Returns:
        List[str]: A list of registered optimizers' name.
    """
    optimizers = []
    try:
        import Sophia
    except ImportError:
        pass
    else:
        for module_name in dir(Sophia):
            _optim = getattr(Sophia, module_name)
            if inspect.isclass(_optim) and issubclass(_optim, torch.optim.Optimizer):
                OPTIMIZERS.register_module(module=_optim)
                optimizers.append(module_name)
    return optimizers


SOPHIA_OPTIMIZERS = register_sophia_optimizers()


def register_bitsandbytes_optimizers() -> list[str]:
    """Register optimizers in ``bitsandbytes`` to the ``OPTIMIZERS`` registry.

    In the `bitsandbytes` library, optimizers that have the same name as the
    default optimizers in PyTorch are prefixed with ``bnb_``. For example,
    ``bnb_Adagrad``.

    Returns:
        List[str]: A list of registered optimizers' name.
    """
    dadaptation_optimizers = []
    import bitsandbytes as bnb

    optim_classes = inspect.getmembers(
        bnb.optim,
        lambda _optim: (inspect.isclass(_optim) and issubclass(_optim, torch.optim.Optimizer)),
    )
    for name, optim_cls in optim_classes:
        if name in OPTIMIZERS:
            name = f"bnb_{name}"
        OPTIMIZERS.register_module(module=optim_cls, name=name)
        dadaptation_optimizers.append(name)
    return dadaptation_optimizers


BITSANDBYTES_OPTIMIZERS = register_bitsandbytes_optimizers()


def register_transformers_optimizers():
    transformer_optimizers = []
    try:
        from transformers import Adafactor
    except ImportError:
        pass
    else:
        OPTIMIZERS.register_module(name="Adafactor", module=Adafactor)
        transformer_optimizers.append("Adafactor")
    return transformer_optimizers


TRANSFORMERS_OPTIMIZERS = register_transformers_optimizers()


def build_optim_wrapper(model: nn.Module, cfg: dict | Config | ConfigDict) -> OptimWrapper:
    """Build function of OptimWrapper.

    If ``constructor`` is set in the ``cfg``, this method will build an
    optimizer wrapper constructor, and use optimizer wrapper constructor to
    build the optimizer wrapper. If ``constructor`` is not set, the
    ``DefaultOptimWrapperConstructor`` will be used by default.

    Args:
        model (nn.Module): Model to be optimized.
        cfg (dict): Config of optimizer wrapper, optimizer constructor and
            optimizer.

    Returns:
        OptimWrapper: The built optimizer wrapper.
    """
    optim_wrapper_cfg = copy.deepcopy(cfg)
    constructor_type = optim_wrapper_cfg.pop("constructor", "DefaultOptimWrapperConstructor")
    paramwise_cfg = optim_wrapper_cfg.pop("paramwise_cfg", None)

    # NPU is no longer supported

    optim_wrapper_constructor = OPTIM_WRAPPER_CONSTRUCTORS.build(
        {
            "type": constructor_type,
            "optim_wrapper_cfg": optim_wrapper_cfg,
            "paramwise_cfg": paramwise_cfg,
        }
    )
    optim_wrapper = optim_wrapper_constructor(model)
    return optim_wrapper
