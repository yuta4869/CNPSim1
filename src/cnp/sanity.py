"""Canonical Sin/Cos tasks for layered CNP sanity checks.

These tasks are intentionally simpler than :mod:`cnp.synthetic`.  They use
only two known, noiseless functions on ``x in [-pi, pi]``:

``sin(x)`` and ``cos(x)``.

Training a model on one function checks the optimizer/decoder path.  Training
one model on both functions additionally checks that predictions are actually
conditioned on the Context ``(X, Y)`` pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


CANONICAL_FUNCTION_NAMES = ("sin", "cos")
CANONICAL_X_RANGE = (-math.pi, math.pi)


@dataclass(frozen=True)
class CanonicalTrainingBatch:
    """One batch of canonical function episodes."""

    context_x: Tensor
    context_y: Tensor
    context_mask: Tensor
    target_x: Tensor
    target_y: Tensor
    function_indices: Tensor
    context_counts: Tensor


@dataclass(frozen=True)
class CanonicalEvaluationTask:
    """A deterministic Context/Query task on an interleaved time grid."""

    context_x: Tensor
    context_y: Tensor
    target_x: Tensor
    target_y: Tensor
    context_indices: Tensor
    target_indices: Tensor
    function_indices: Tensor
    context_count: int
    seed: int


def _validate_function_indices(function_indices: Tensor) -> Tensor:
    if function_indices.ndim == 0:
        function_indices = function_indices.unsqueeze(0)
    if function_indices.ndim != 1:
        raise ValueError("function_indices must be a scalar or rank-1 tensor")
    if function_indices.dtype == torch.bool or function_indices.is_floating_point():
        raise TypeError("function_indices must contain integer indices")
    if function_indices.numel() < 1:
        raise ValueError("at least one function index is required")
    if torch.any(function_indices < 0) or torch.any(
        function_indices >= len(CANONICAL_FUNCTION_NAMES)
    ):
        raise ValueError("canonical function index is out of range")
    return function_indices.to(dtype=torch.long)


def evaluate_canonical_functions(function_indices: Tensor, x: Tensor) -> Tensor:
    """Evaluate canonical functions at coordinates ``x``.

    ``function_indices`` is ``[B]`` (or a scalar). ``x`` may be ``[N, 1]``
    for a shared grid or ``[B, N, 1]`` for per-function coordinates.  The
    result is always batched and shaped ``[B, N, 1]``.
    """

    function_indices = _validate_function_indices(function_indices)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3 or x.shape[-1] != 1:
        raise ValueError("x must have shape [N, 1] or [B, N, 1]")
    if x.shape[0] == 1 and function_indices.numel() != 1:
        x = x.expand(function_indices.numel(), -1, -1)
    elif x.shape[0] != function_indices.numel():
        raise ValueError("x and function_indices must have compatible batch sizes")
    if x.device != function_indices.device:
        function_indices = function_indices.to(x.device)
    if not x.is_floating_point():
        raise TypeError("x must use a floating-point dtype")

    sine = torch.sin(x)
    cosine = torch.cos(x)
    selector = function_indices.view(-1, 1, 1) == 0
    return torch.where(selector, sine, cosine)


def sample_canonical_training_batch(
    allowed_function_indices: Tensor,
    *,
    batch_size: int,
    min_context: int,
    max_context: int,
    target_count: int,
    generator: torch.Generator,
    device: torch.device,
) -> CanonicalTrainingBatch:
    """Sample variable-size Context sets and independent Query coordinates."""

    allowed_function_indices = _validate_function_indices(allowed_function_indices)
    if allowed_function_indices.device.type != "cpu":
        raise ValueError("allowed_function_indices must remain on CPU")
    if generator.device.type != "cpu":
        raise ValueError("training sampling must use a CPU generator")
    if batch_size < 1 or target_count < 1:
        raise ValueError("batch_size and target_count must be positive")
    if not 1 <= min_context <= max_context:
        raise ValueError("expected 1 <= min_context <= max_context")

    selections = torch.randint(
        0,
        allowed_function_indices.numel(),
        (batch_size,),
        generator=generator,
    )
    function_indices = allowed_function_indices[selections]
    context_counts = torch.randint(
        min_context,
        max_context + 1,
        (batch_size,),
        generator=generator,
    )
    context_mask = (
        torch.arange(max_context).unsqueeze(0) < context_counts.unsqueeze(1)
    )
    coordinates = torch.rand(
        batch_size,
        max_context + target_count,
        1,
        generator=generator,
        dtype=torch.float32,
    )
    coordinates.mul_(CANONICAL_X_RANGE[1] - CANONICAL_X_RANGE[0]).add_(
        CANONICAL_X_RANGE[0]
    )
    context_x = coordinates[:, :max_context]
    target_x = coordinates[:, max_context:]
    context_y = evaluate_canonical_functions(function_indices, context_x)
    target_y = evaluate_canonical_functions(function_indices, target_x)

    return CanonicalTrainingBatch(
        context_x=context_x.to(device),
        context_y=context_y.to(device),
        context_mask=context_mask.to(device),
        target_x=target_x.to(device),
        target_y=target_y.to(device),
        function_indices=function_indices,
        context_counts=context_counts,
    )


def make_canonical_evaluation_task(
    function_indices: Tensor,
    *,
    context_count: int,
    grid_points: int,
    seed: int,
) -> CanonicalEvaluationTask:
    """Build a fixed task with disjoint and nested Context/Query coordinates.

    Context candidates occupy even positions and Query coordinates occupy odd
    positions of an interleaved grid. Reusing ``seed`` at several Context
    counts makes each smaller Context set a subset of the larger set.
    """

    function_indices = _validate_function_indices(function_indices)
    if function_indices.device.type != "cpu":
        raise ValueError("evaluation function indices must be on CPU")
    if context_count < 1:
        raise ValueError("context_count must be positive")
    if grid_points < 3 or grid_points % 2 == 0:
        raise ValueError("grid_points must be an odd integer of at least 3")
    if context_count > grid_points:
        raise ValueError("context_count exceeds available Context candidates")

    grid = torch.linspace(
        CANONICAL_X_RANGE[0],
        CANONICAL_X_RANGE[1],
        2 * grid_points - 1,
        dtype=torch.float32,
    )
    candidates = torch.arange(0, grid.numel(), 2)
    target_indices = torch.arange(1, grid.numel() - 1, 2)
    generator = torch.Generator().manual_seed(seed)
    scores = torch.rand(
        function_indices.numel(), candidates.numel(), generator=generator
    )
    priority = candidates[scores.argsort(dim=1)]
    context_indices = priority[:, :context_count].sort(dim=1).values
    expanded_target_indices = target_indices.view(1, -1).expand(
        function_indices.numel(), -1
    )
    expanded_x = grid.view(1, -1, 1).expand(function_indices.numel(), -1, -1)
    expanded_y = evaluate_canonical_functions(function_indices, expanded_x)

    def gather(values: Tensor, indices: Tensor) -> Tensor:
        return torch.gather(values, 1, indices.unsqueeze(-1))

    return CanonicalEvaluationTask(
        context_x=gather(expanded_x, context_indices),
        context_y=gather(expanded_y, context_indices),
        target_x=gather(expanded_x, expanded_target_indices),
        target_y=gather(expanded_y, expanded_target_indices),
        context_indices=context_indices,
        target_indices=expanded_target_indices,
        function_indices=function_indices,
        context_count=context_count,
        seed=seed,
    )
