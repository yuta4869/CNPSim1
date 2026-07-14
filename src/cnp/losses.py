"""Loss functions for Conditional Neural Processes."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
from torch.distributions import Independent, Normal


def gaussian_nll(
    mean: Tensor,
    scale: Tensor,
    target_y: Tensor,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Tensor:
    """Negative log likelihood under a diagonal Gaussian output distribution."""

    if mean.shape != scale.shape or mean.shape != target_y.shape:
        raise ValueError("mean, scale, and target_y must have identical shapes")
    if mean.ndim < 2:
        raise ValueError("inputs must contain point and feature axes")
    if torch.any(scale <= 0):
        raise ValueError("scale must be strictly positive")

    loss = -Independent(Normal(mean, scale), 1).log_prob(target_y)
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"unsupported reduction: {reduction!r}")
