"""Training configuration schemas.

This module provides schemas for optimizers, schedulers, and training loops.

Example:
    >>> from visdet.schemas.training import AdamWConfig, OneCycleLRConfig
    >>> optimizer = AdamWConfig(lr=1e-4, weight_decay=0.05)
    >>> scheduler = OneCycleLRConfig(max_lr=1e-3)
"""

from typing import Annotated, Literal, Optional, Union

from pydantic import Field

from visdet.schemas.base import ComponentConfig, OptionalConfig, VisdetBaseConfig


# =============================================================================
# Optimizer Configurations
# =============================================================================


class AdamWConfig(ComponentConfig):
    """AdamW optimizer configuration.

    AdamW with decoupled weight decay, recommended for transformers.

    Attributes:
        lr: Learning rate.
        betas: Adam beta parameters.
        weight_decay: Weight decay coefficient.

    Example:
        >>> cfg = AdamWConfig(lr=1e-4, weight_decay=0.05)
    """

    type: Literal["AdamW"] = "AdamW"
    lr: float = Field(default=1e-4, gt=0, description="Learning rate")
    betas: tuple[float, float] = Field(
        default=(0.9, 0.999), description="Adam beta parameters"
    )
    weight_decay: float = Field(
        default=0.05, ge=0, description="Weight decay coefficient"
    )
    eps: float = Field(default=1e-8, gt=0, description="Epsilon for numerical stability")


class SGDConfig(ComponentConfig):
    """SGD optimizer configuration.

    Standard SGD with optional momentum.

    Attributes:
        lr: Learning rate.
        momentum: Momentum factor.
        weight_decay: Weight decay (L2 penalty).
    """

    type: Literal["SGD"] = "SGD"
    lr: float = Field(default=0.02, gt=0, description="Learning rate")
    momentum: float = Field(default=0.9, ge=0, description="Momentum factor")
    weight_decay: float = Field(default=0.0001, ge=0, description="Weight decay")
    nesterov: bool = Field(default=False, description="Use Nesterov momentum")


class AdamConfig(ComponentConfig):
    """Adam optimizer configuration."""

    type: Literal["Adam"] = "Adam"
    lr: float = Field(default=1e-3, gt=0, description="Learning rate")
    betas: tuple[float, float] = Field(
        default=(0.9, 0.999), description="Adam beta parameters"
    )
    weight_decay: float = Field(default=0, ge=0, description="Weight decay")
    eps: float = Field(default=1e-8, gt=0, description="Epsilon for numerical stability")


# 8-bit optimizers (requires bitsandbytes)
class AdamW8bitConfig(ComponentConfig):
    """8-bit AdamW optimizer configuration.

    Memory-efficient 8-bit AdamW from bitsandbytes.
    Requires bitsandbytes package.
    """

    type: Literal["AdamW8bit"] = "AdamW8bit"
    lr: float = Field(default=1e-4, gt=0, description="Learning rate")
    betas: tuple[float, float] = Field(default=(0.9, 0.999), description="Beta parameters")
    weight_decay: float = Field(default=0.05, ge=0, description="Weight decay")
    eps: float = Field(default=1e-8, gt=0, description="Epsilon")


# =============================================================================
# Optimizer Wrapper Configuration
# =============================================================================


class OptimWrapperConfig(VisdetBaseConfig):
    """Optimizer wrapper configuration.

    Wraps an optimizer with additional features like gradient clipping.
    """

    type: Literal["OptimWrapper"] = Field(default="OptimWrapper", description="Wrapper type")
    optimizer: AdamWConfig | SGDConfig | AdamConfig | AdamW8bitConfig = Field(
        ..., description="Optimizer config"
    )
    clip_grad: OptionalConfig = Field(
        default=None, description="Gradient clipping config"
    )
    accumulative_counts: int = Field(
        default=1, ge=1, description="Gradient accumulation steps"
    )


# =============================================================================
# Learning Rate Scheduler Configurations
# =============================================================================


class OneCycleLRConfig(ComponentConfig):
    """OneCycle learning rate scheduler configuration.

    Fast.ai's 1cycle policy - warmup, then anneal.

    Attributes:
        max_lr: Maximum learning rate at peak.
        total_steps: Total training steps (or use epochs).
        pct_start: Percentage of cycle spent increasing LR.
    """

    type: Literal["OneCycleLR"] = "OneCycleLR"
    max_lr: float = Field(default=1e-3, gt=0, description="Maximum learning rate")
    total_steps: Optional[int] = Field(
        default=None, description="Total steps (auto-computed if None)"
    )
    pct_start: float = Field(
        default=0.3, ge=0, le=1, description="Fraction of cycle for warmup"
    )
    anneal_strategy: Literal["cos", "linear"] = Field(
        default="cos", description="Annealing strategy"
    )
    div_factor: float = Field(
        default=25.0, gt=0, description="Initial LR = max_lr / div_factor"
    )
    final_div_factor: float = Field(
        default=1e4, gt=0, description="Final LR = initial_lr / final_div_factor"
    )


class MultiStepLRConfig(ComponentConfig):
    """Multi-step learning rate scheduler configuration.

    Decays LR by gamma at specified milestones.
    """

    type: Literal["MultiStepLR"] = "MultiStepLR"
    milestones: list[int] = Field(
        ..., min_length=1, description="Epochs to decay LR"
    )
    gamma: float = Field(default=0.1, gt=0, description="Decay factor")


class CosineAnnealingLRConfig(ComponentConfig):
    """Cosine annealing learning rate scheduler configuration."""

    type: Literal["CosineAnnealingLR"] = "CosineAnnealingLR"
    T_max: int = Field(..., gt=0, description="Maximum number of iterations")
    eta_min: float = Field(default=0, ge=0, description="Minimum learning rate")


class LinearLRConfig(ComponentConfig):
    """Linear learning rate scheduler for warmup."""

    type: Literal["LinearLR"] = "LinearLR"
    start_factor: float = Field(
        default=0.001, ge=0, le=1, description="Starting factor"
    )
    end_factor: float = Field(default=1.0, ge=0, description="Ending factor")
    by_epoch: bool = Field(default=False, description="Step by epoch or iteration")
    begin: int = Field(default=0, ge=0, description="Begin epoch/iteration")
    end: int = Field(default=500, gt=0, description="End epoch/iteration")


# =============================================================================
# Training Loop Configurations
# =============================================================================


class EpochBasedTrainLoopConfig(ComponentConfig):
    """Epoch-based training loop configuration.

    Standard training loop that iterates by epochs.

    Attributes:
        max_epochs: Maximum number of training epochs.
        val_interval: Validate every N epochs.
    """

    type: Literal["EpochBasedTrainLoop"] = "EpochBasedTrainLoop"
    max_epochs: int = Field(default=12, ge=1, description="Maximum epochs")
    val_interval: int = Field(default=1, ge=1, description="Validation interval")


class IterBasedTrainLoopConfig(ComponentConfig):
    """Iteration-based training loop configuration.

    Training loop that iterates by iterations (steps).
    """

    type: Literal["IterBasedTrainLoop"] = "IterBasedTrainLoop"
    max_iters: int = Field(..., ge=1, description="Maximum iterations")
    val_interval: int = Field(default=5000, ge=1, description="Validation interval")


class ValLoopConfig(ComponentConfig):
    """Validation loop configuration."""

    type: Literal["ValLoop"] = "ValLoop"


class TestLoopConfig(ComponentConfig):
    """Test loop configuration."""

    type: Literal["TestLoop"] = "TestLoop"


# =============================================================================
# Hook Configurations
# =============================================================================


class CheckpointHookConfig(ComponentConfig):
    """Checkpoint saving hook configuration."""

    type: Literal["CheckpointHook"] = "CheckpointHook"
    interval: int = Field(default=1, ge=1, description="Save interval (epochs)")
    by_epoch: bool = Field(default=True, description="Interval by epoch or iteration")
    save_best: Optional[str] = Field(
        default=None, description="Metric to track for best model"
    )
    max_keep_ckpts: int = Field(default=3, ge=1, description="Max checkpoints to keep")


class LoggerHookConfig(ComponentConfig):
    """Logging hook configuration."""

    type: Literal["LoggerHook"] = "LoggerHook"
    interval: int = Field(default=50, ge=1, description="Log interval (iterations)")
    log_metric_by_epoch: bool = Field(default=True, description="Log by epoch")


# Type aliases
OptimizerType = Annotated[
    Union[AdamWConfig, SGDConfig, AdamConfig, AdamW8bitConfig],
    Field(discriminator="type"),
]
SchedulerType = Annotated[
    Union[OneCycleLRConfig, MultiStepLRConfig, CosineAnnealingLRConfig, LinearLRConfig],
    Field(discriminator="type"),
]
TrainLoopType = Annotated[
    Union[EpochBasedTrainLoopConfig, IterBasedTrainLoopConfig],
    Field(discriminator="type"),
]
