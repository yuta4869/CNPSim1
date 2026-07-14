"""Generic context/target sampling for sets of function observations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CNPBatch:
    """A sampled Conditional Neural Process training batch."""

    context_x: Tensor
    context_y: Tensor
    target_x: Tensor
    target_y: Tensor


def _gather_points(values: Tensor, indices: Tensor) -> Tensor:
    expanded = indices.unsqueeze(-1).expand(-1, -1, values.shape[-1])
    return torch.gather(values, dim=1, index=expanded)


def sample_context_target(
    x: Tensor,
    y: Tensor,
    num_context: int,
    num_target: int,
    *,
    generator: torch.Generator | None = None,
) -> CNPBatch:
    """Sample disjoint context and target points from each function in a batch.

    ``x`` and ``y`` must be shaped ``[B, N, D]`` (or ``[N, D]`` for one
    function). A separate random permutation is drawn for each batch item. If
    supplied, ``generator`` must target the same device type as ``x``.
    """

    x_was_unbatched = x.ndim == 2
    y_was_unbatched = y.ndim == 2
    if x_was_unbatched:
        x = x.unsqueeze(0)
    if y_was_unbatched:
        y = y.unsqueeze(0)
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("x and y must have rank 2 or 3")
    if x_was_unbatched != y_was_unbatched:
        raise ValueError("x and y must use the same batching convention")
    if x.shape[:2] != y.shape[:2]:
        raise ValueError("x and y must have matching batch and point axes")
    if x.device != y.device:
        raise ValueError("x and y must be on the same device")
    if generator is not None and generator.device.type != x.device.type:
        raise ValueError(
            "generator and x must use the same device type; "
            f"got generator={generator.device.type}, x={x.device.type}"
        )
    if num_context < 1 or num_target < 1:
        raise ValueError("num_context and num_target must both be positive")

    batch_size, point_count = x.shape[:2]
    if num_context + num_target > point_count:
        raise ValueError("num_context + num_target cannot exceed the point count")

    random_scores = torch.rand(
        batch_size,
        point_count,
        device=x.device,
        generator=generator,
    )
    permutation = random_scores.argsort(dim=-1)
    context_indices = permutation[:, :num_context]
    target_indices = permutation[:, num_context : num_context + num_target]

    batch = CNPBatch(
        context_x=_gather_points(x, context_indices),
        context_y=_gather_points(y, context_indices),
        target_x=_gather_points(x, target_indices),
        target_y=_gather_points(y, target_indices),
    )
    if not x_was_unbatched:
        return batch
    return CNPBatch(
        context_x=batch.context_x.squeeze(0),
        context_y=batch.context_y.squeeze(0),
        target_x=batch.target_x.squeeze(0),
        target_y=batch.target_y.squeeze(0),
    )
