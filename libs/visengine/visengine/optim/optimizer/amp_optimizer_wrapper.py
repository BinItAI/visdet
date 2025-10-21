# ruff: noqa
# type: ignore
# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager

import torch
import torch.nn as nn

from visengine.registry import OPTIM_WRAPPERS

from .optimizer_wrapper import OptimWrapper

# updated from torch.cuda.amp -> torch.amp
# due to a deprecation warning
from torch.amp import GradScaler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from visengine.runner.amp import autocast


# There's also an APEX implementation that came before this,
# but the torch implementation (here) is recommended.
# https://discuss.pytorch.org/t/torch-cuda-amp-vs-nvidia-apex/74994
# https://github.com/open-mmlab/mmengine/tree/main/mmengine/optim/optimizer
@OPTIM_WRAPPERS.register_module(force=True)
class AmpOptimWrapper(OptimWrapper):
    """A subclass of :class:`OptimWrapper` that supports automatic mixed
    precision training based on torch.cuda.amp.

    ``AmpOptimWrapper`` provides a unified interface with
    ``OptimWrapper``, so ``AmpOptimWrapper`` can be used in the same way
    as ``OptimWrapper``.

    Warnings:
        ``AmpOptimWrapper`` requires PyTorch >= 1.6.

    Args:
        loss_scale (float or str or dict): The initial configuration of
            `torch.cuda.amp.GradScaler`. See more specific arguments
            introduction at `PyTorch AMP <https://pytorch.org/docs/stable/amp.html?highlight=gradscalertorch.cuda.amp.GradScaler>`_ # noqa: E501
            Defaults to ``dynamic``.

            - "dynamic": Initialize GradScale without any arguments.
            - float: Initialize GradScaler with ``init_scale``.
            - dict: Initialize GradScaler with more detail configuration.

        dtype (str or torch.dtype, optional): The data type to autocast in amp.
            If a ``str`` is given, it will be converted to ``torch.dtype``.
            Valid ``str`` format are `'float16'`, `'bfloat16'`, `'float32'` and
            `'float64'`. If set to ``None``, the default data type will be used.
            Defaults to None.
            `New in version 0.6.1.`
        use_fsdp (bool): Using ``ShardedGradScaler`` when it is True. It should
            be enabled when using ``FullyShardedDataParallel``.
            Defaults to False.
            `New in version 0.8.0.`
        **kwargs: Keyword arguments passed to OptimWrapper.

    Warnings:
        ``dtype`` argument is only available with PyTorch version >= 1.10.0. If
        you use PyTorch of an older version, it will be ignored.

    Note:
        If you use ``IterBasedRunner`` and enable gradient accumulation,
        the original `max_iters` should be multiplied by
        ``accumulative_counts``.
    """

    valid_dtypes = ("float16", "bfloat16", "float32", "float64")

    def __init__(
        self,
        loss_scale: str = "dynamic",
        dtype: str | torch.dtype = None,
        use_fsdp: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._scale_update_param = None

        if use_fsdp:
            scaler_type = ShardedGradScaler
        else:
            scaler_type = GradScaler

        if loss_scale == "dynamic":
            #  If loss_scale is a string, it must be 'dynamic', then dynamic
            #  loss scaling will be used.
            self.loss_scaler = scaler_type()
        elif isinstance(loss_scale, float):
            # Static loss scaling
            self._scale_update_param = loss_scale
            self.loss_scaler = scaler_type(init_scale=loss_scale)
        elif isinstance(loss_scale, dict):
            # More specific configuration.
            self.loss_scaler = scaler_type(**loss_scale)
        else:
            raise TypeError(f'loss_scale must be of type float, dict, or "dynamic", but got {loss_scale}')

        # convert string value to torch.dtype
        if isinstance(dtype, str):
            assert dtype in self.valid_dtypes, f"dtype should be any of {self.valid_dtypes}, got {dtype}"
            dtype = getattr(torch, dtype)

        assert dtype is None or isinstance(dtype, torch.dtype), (
            f"dtype should be None or instance of torch.dtype, got {dtype}"
        )
        self.cast_dtype = dtype

    def backward(self, loss: torch.Tensor, **kwargs):
        """Perform gradient back propagation with :attr:`loss_scaler`.

        Args:
            loss (torch.Tensor): The loss of current iteration.
            kwargs: Keyword arguments passed to :meth:`torch.Tensor.backward`
        """
        self.loss_scaler.scale(loss).backward(**kwargs)
        self._inner_count += 1

    def step(self, **kwargs):
        """Update parameters with :attr:`loss_scaler`.

        Args:
            kwargs: Keyword arguments passed to
                :meth:`torch.optim.Optimizer.step`.
        """
        if self.clip_grad_kwargs:
            self.loss_scaler.unscale_(self.optimizer)
            self._clip_grad()
        self.loss_scaler.step(self.optimizer, **kwargs)
        self.loss_scaler.update(self._scale_update_param)

    def state_dict(self) -> dict:
        """Get the state dictionary of :attr:`optimizer` and
        :attr:`loss_scaler`.

        Based on the state dictionary of the optimizer, the returned state
        dictionary will add a key named "loss_scaler".

        Returns:
            dict: The merged state dict of :attr:`loss_scaler` and
            :attr:`optimizer`.
        """
        # save state_dict of loss_scaler
        state_dict = super().state_dict()
        state_dict["loss_scaler"] = self.loss_scaler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict):
        """Load and parse the state dictionary of :attr:`optimizer` and
        :attr:`loss_scaler`.

        If state_dict contains "loss_scaler.", the :attr:`loss_scaler` will
        load the corresponding keys. Otherwise, only the :attr:`optimizer`
        will load the state dictionary.

        Args:
            state_dict (dict): The state dict of :attr:`optimizer` and
                :attr:`loss_scaler`
        """
        if "loss_scaler" in state_dict:
            self.loss_scaler.load_state_dict(state_dict.pop("loss_scaler"))

        if "base_param_settings" in state_dict:
            self.base_param_settings = state_dict.pop("base_param_settings")

        # load state_dict of optimizer
        self.optimizer.load_state_dict(state_dict)

    @contextmanager
    def optim_context(self, model: nn.Module):
        """Enables the context for mixed precision training, and enables the
        context for disabling gradient synchronization during gradient
        accumulation context.

        Args:
            model (nn.Module): The training model.
        """
        with super().optim_context(model), autocast(dtype=self.cast_dtype):
            yield
