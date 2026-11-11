"""Synthetic sinusoid tasks and metrics for CNP regression benchmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


ParameterName = Literal["amplitude", "angular_frequency", "phase", "offset"]
EvaluationRegime = Literal[
    "interpolation",
    "gap_interpolation",
    "extrapolation_stress",
]


@dataclass(frozen=True)
class SinusoidParameterRange:
    """Ranges for ``offset + amplitude * sin(frequency * x + phase)``."""

    amplitude: tuple[float, float] = (0.5, 2.0)
    angular_frequency: tuple[float, float] = (0.8, 2.0)
    phase: tuple[float, float] = (0.0, 2.0 * math.pi)
    offset: tuple[float, float] = (-0.5, 0.5)

    def __post_init__(self) -> None:
        for name in (
            "amplitude",
            "angular_frequency",
            "phase",
            "offset",
        ):
            lower, upper = getattr(self, name)
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise ValueError(f"{name} bounds must be finite")
            if lower >= upper:
                raise ValueError(f"{name} lower bound must be smaller than upper bound")
        if self.amplitude[0] <= 0:
            raise ValueError("amplitude must be strictly positive")
        if self.angular_frequency[0] <= 0:
            raise ValueError("angular frequency must be strictly positive")

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "amplitude": list(self.amplitude),
            "angular_frequency": list(self.angular_frequency),
            "phase": list(self.phase),
            "offset": list(self.offset),
        }


@dataclass(frozen=True)
class SyntheticTrainingBatch:
    """One batch of function episodes sampled from a fixed parameter bank."""

    context_x: Tensor
    context_y: Tensor
    context_mask: Tensor
    target_x: Tensor
    target_y: Tensor
    function_indices: Tensor
    context_counts: Tensor


@dataclass(frozen=True)
class SyntheticEvaluationTask:
    """A fixed Context/Query task for a bank of sinusoid functions."""

    context_x: Tensor
    context_y: Tensor
    target_x: Tensor
    target_y: Tensor
    context_indices: Tensor
    target_indices: Tensor
    regime: EvaluationRegime
    context_count: int


def sample_sinusoid_parameters(
    count: int,
    parameter_range: SinusoidParameterRange,
    *,
    generator: torch.Generator,
) -> Tensor:
    """Sample a reproducible CPU bank shaped ``[count, 4]``.

    Columns are amplitude, angular frequency, phase, and offset. Sampling is
    half-open at the upper bound because it is based on ``torch.rand``.
    """

    if count < 1:
        raise ValueError("count must be positive")
    if generator.device.type != "cpu":
        raise ValueError("sinusoid parameter banks must use a CPU generator")

    unit = torch.rand(count, 4, generator=generator, dtype=torch.float32)
    bounds = (
        parameter_range.amplitude,
        parameter_range.angular_frequency,
        parameter_range.phase,
        parameter_range.offset,
    )
    for column, (lower, upper) in enumerate(bounds):
        unit[:, column].mul_(upper - lower).add_(lower)
    return unit


def evaluate_sinusoids(parameters: Tensor, x: Tensor) -> Tensor:
    """Evaluate a parameter bank at coordinates ``x``.

    ``parameters`` must be ``[B, 4]`` (or ``[4]``). ``x`` may be ``[N, 1]``
    for a shared grid or ``[B, N, 1]`` for per-function coordinates.
    """

    parameters_were_unbatched = parameters.ndim == 1
    if parameters_were_unbatched:
        parameters = parameters.unsqueeze(0)
    if parameters.ndim != 2 or parameters.shape[-1] != 4:
        raise ValueError("parameters must have shape [4] or [B, 4]")

    x_was_unbatched = x.ndim == 2
    if x_was_unbatched:
        x = x.unsqueeze(0)
    if x.ndim != 3 or x.shape[-1] != 1:
        raise ValueError("x must have shape [N, 1] or [B, N, 1]")
    if x.shape[0] == 1 and parameters.shape[0] != 1:
        x = x.expand(parameters.shape[0], -1, -1)
    elif x.shape[0] != parameters.shape[0]:
        raise ValueError("x and parameters must have compatible batch sizes")
    if x.device != parameters.device:
        raise ValueError("x and parameters must be on the same device")
    if x.dtype != parameters.dtype:
        raise ValueError("x and parameters must have the same dtype")

    amplitude = parameters[:, 0].view(-1, 1, 1)
    frequency = parameters[:, 1].view(-1, 1, 1)
    phase = parameters[:, 2].view(-1, 1, 1)
    offset = parameters[:, 3].view(-1, 1, 1)
    values = offset + amplitude * torch.sin(frequency * x + phase)

    if parameters_were_unbatched and x_was_unbatched:
        return values.squeeze(0)
    return values


def sample_synthetic_training_batch(
    parameter_bank: Tensor,
    *,
    batch_size: int,
    min_context: int,
    max_context: int,
    target_count: int,
    x_range: tuple[float, float],
    generator: torch.Generator,
    device: torch.device,
) -> SyntheticTrainingBatch:
    """Sample variable-size Context sets and independent Query coordinates."""

    if parameter_bank.ndim != 2 or parameter_bank.shape[-1] != 4:
        raise ValueError("parameter_bank must have shape [functions, 4]")
    if parameter_bank.device.type != "cpu":
        raise ValueError("parameter_bank must remain on CPU for reproducible sampling")
    if generator.device.type != "cpu":
        raise ValueError("training sampling must use a CPU generator")
    if batch_size < 1 or target_count < 1:
        raise ValueError("batch_size and target_count must be positive")
    if not 1 <= min_context <= max_context:
        raise ValueError("expected 1 <= min_context <= max_context")
    x_min, x_max = x_range
    if not math.isfinite(x_min) or not math.isfinite(x_max) or x_min >= x_max:
        raise ValueError("x_range must contain increasing finite bounds")

    function_indices = torch.randint(
        0,
        parameter_bank.shape[0],
        (batch_size,),
        generator=generator,
    )
    parameters = parameter_bank[function_indices]
    context_counts = torch.randint(
        min_context,
        max_context + 1,
        (batch_size,),
        generator=generator,
    )
    context_mask = (
        torch.arange(max_context).unsqueeze(0) < context_counts.unsqueeze(1)
    )

    point_count = max_context + target_count
    coordinates = torch.rand(
        batch_size,
        point_count,
        1,
        generator=generator,
        dtype=parameter_bank.dtype,
    )
    coordinates.mul_(x_max - x_min).add_(x_min)
    context_x = coordinates[:, :max_context]
    target_x = coordinates[:, max_context:]
    context_y = evaluate_sinusoids(parameters, context_x)
    target_y = evaluate_sinusoids(parameters, target_x)

    return SyntheticTrainingBatch(
        context_x=context_x.to(device),
        context_y=context_y.to(device),
        context_mask=context_mask.to(device),
        target_x=target_x.to(device),
        target_y=target_y.to(device),
        function_indices=function_indices,
        context_counts=context_counts,
    )


def _gather_points(values: Tensor, indices: Tensor) -> Tensor:
    return torch.gather(values, 1, indices.unsqueeze(-1).expand(-1, -1, 1))


def _nested_context_indices(
    function_count: int,
    candidates: Tensor,
    anchors: tuple[int, int],
    context_count: int,
    *,
    generator: torch.Generator,
) -> Tensor:
    if context_count < 2:
        raise ValueError("evaluation requires at least two Context points")
    if anchors[0] == anchors[1]:
        raise ValueError("evaluation requires two distinct Context anchors")
    candidate_values = candidates.tolist()
    if anchors[0] not in candidate_values or anchors[1] not in candidate_values:
        raise ValueError("anchors must be included in the Context candidate set")
    remaining = candidates[
        (candidates != anchors[0]) & (candidates != anchors[1])
    ]
    if context_count > remaining.numel() + 2:
        raise ValueError("context_count exceeds available Context candidates")

    scores = torch.rand(function_count, remaining.numel(), generator=generator)
    order = scores.argsort(dim=1)
    shuffled = remaining[order]
    anchors_tensor = torch.tensor(anchors).view(1, 2).expand(function_count, -1)
    selected = torch.cat((anchors_tensor, shuffled[:, : context_count - 2]), dim=1)
    return selected.sort(dim=1).values


def make_synthetic_evaluation_task(
    parameters: Tensor,
    *,
    regime: EvaluationRegime,
    context_count: int,
    grid_points: int,
    x_range: tuple[float, float],
    seed: int,
    gap_half_width: float = 0.6,
    extrapolation_context_half_width: float = 1.0,
) -> SyntheticEvaluationTask:
    """Build a deterministic task with nested Context sets across counts.

    Reusing ``seed`` for several Context counts makes each smaller set a
    prefix/subset of the larger set. Context candidates occupy the even
    positions of an interleaved grid and fixed Query coordinates occupy the
    odd midpoints, so every Context count is compared at exactly the same
    Query locations without Context/Query overlap.
    """

    if parameters.ndim != 2 or parameters.shape[-1] != 4:
        raise ValueError("parameters must have shape [functions, 4]")
    if parameters.device.type != "cpu":
        raise ValueError("evaluation parameter banks must be on CPU")
    if grid_points < 9 or grid_points % 2 == 0:
        raise ValueError("grid_points must be an odd integer of at least 9")
    x_min, x_max = x_range
    if x_min >= x_max:
        raise ValueError("x_range bounds must be increasing")
    if gap_half_width <= 0:
        raise ValueError("gap_half_width must be positive")
    if extrapolation_context_half_width <= 0:
        raise ValueError("extrapolation_context_half_width must be positive")
    if regime == "gap_interpolation" and not (
        x_min < -gap_half_width < gap_half_width < x_max
    ):
        raise ValueError("gap interpolation requires both sides of a zero-centered gap")
    if regime == "extrapolation_stress" and not (
        x_min
        < -extrapolation_context_half_width
        < extrapolation_context_half_width
        < x_max
    ):
        raise ValueError(
            "extrapolation stress requires a zero-centered Context interval inside the domain"
        )

    generator = torch.Generator().manual_seed(seed)
    grid = torch.linspace(
        x_min,
        x_max,
        2 * grid_points - 1,
        dtype=parameters.dtype,
    )
    context_grid_indices = torch.arange(0, grid.shape[0], 2)
    query_grid_indices = torch.arange(1, grid.shape[0] - 1, 2)
    function_count = parameters.shape[0]
    expanded_x = grid.view(1, -1, 1).expand(function_count, -1, -1)
    expanded_y = evaluate_sinusoids(parameters, expanded_x)

    if regime == "interpolation":
        candidates = context_grid_indices
        anchors = (int(candidates[0].item()), int(candidates[-1].item()))
        target_candidates = query_grid_indices
    elif regime == "gap_interpolation":
        context_coordinates = grid[context_grid_indices]
        candidates = context_grid_indices[context_coordinates.abs() >= gap_half_width]
        left_candidates = candidates[grid[candidates] <= -gap_half_width]
        right_candidates = candidates[grid[candidates] >= gap_half_width]
        if left_candidates.numel() == 0 or right_candidates.numel() == 0:
            raise ValueError("grid does not contain Context candidates on both sides of gap")
        left = int(left_candidates[-1].item())
        right = int(right_candidates[0].item())
        anchors = (left, right)
        target_candidates = query_grid_indices[
            grid[query_grid_indices].abs() < gap_half_width
        ]
    elif regime == "extrapolation_stress":
        context_coordinates = grid[context_grid_indices]
        candidates = context_grid_indices[
            context_coordinates.abs() <= extrapolation_context_half_width
        ]
        if candidates.numel() < 2:
            raise ValueError(
                "grid does not contain two Context candidates inside the stress interval"
            )
        anchors = (int(candidates[0].item()), int(candidates[-1].item()))
        target_candidates = query_grid_indices[
            grid[query_grid_indices].abs() > extrapolation_context_half_width
        ]
    else:
        raise ValueError(f"unknown evaluation regime: {regime!r}")
    if target_candidates.numel() == 0:
        raise ValueError("evaluation regime produced no Query coordinates")

    context_indices = _nested_context_indices(
        function_count,
        candidates,
        anchors,
        context_count,
        generator=generator,
    )
    target_mask = torch.zeros(function_count, grid.shape[0], dtype=torch.bool)
    target_mask[:, target_candidates] = True
    target_counts = target_mask.sum(dim=1)
    if torch.any(target_counts == 0) or not torch.all(target_counts == target_counts[0]):
        raise ValueError("evaluation regime produced inconsistent target counts")
    target_indices = torch.nonzero(target_mask, as_tuple=False)[:, 1].view(
        function_count, -1
    )

    return SyntheticEvaluationTask(
        context_x=_gather_points(expanded_x, context_indices),
        context_y=_gather_points(expanded_y, context_indices),
        target_x=_gather_points(expanded_x, target_indices),
        target_y=_gather_points(expanded_y, target_indices),
        context_indices=context_indices,
        target_indices=target_indices,
        regime=regime,
        context_count=context_count,
    )


def deterministic_regression_metrics(prediction: Tensor, target: Tensor) -> dict[str, float | int]:
    """Return function-macro and point-micro deterministic metrics."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must have identical [B, T, D] shapes")
    error = prediction - target
    per_function_mae = error.abs().mean(dim=(1, 2))
    per_function_rmse = error.square().mean(dim=(1, 2)).sqrt()
    return {
        "function_macro_mae": float(per_function_mae.mean().item()),
        "function_macro_rmse": float(per_function_rmse.mean().item()),
        "point_micro_mae": float(error.abs().mean().item()),
        "point_micro_rmse": float(error.square().mean().sqrt().item()),
        "functions": prediction.shape[0],
        "points": prediction.shape[0] * prediction.shape[1],
    }


def gaussian_regression_metrics(
    mean: Tensor,
    scale: Tensor,
    target: Tensor,
) -> dict[str, float | int]:
    """Return pointwise Gaussian metrics plus deterministic mean errors."""

    if mean.shape != scale.shape or mean.shape != target.shape or mean.ndim != 3:
        raise ValueError("mean, scale, and target must have identical [B, T, D] shapes")
    if torch.any(scale <= 0):
        raise ValueError("scale must be strictly positive")
    metrics = deterministic_regression_metrics(mean, target)
    error = mean - target
    nll = 0.5 * math.log(2.0 * math.pi) + scale.log() + 0.5 * (error / scale).square()
    per_function_nll = nll.mean(dim=(1, 2))
    metrics.update(
        {
            "function_macro_nll": float(per_function_nll.mean().item()),
            "point_micro_nll": float(nll.mean().item()),
            "coverage_68": float((error.abs() <= scale).to(torch.float32).mean().item()),
            "coverage_95": float(
                (error.abs() <= 1.96 * scale).to(torch.float32).mean().item()
            ),
            "mean_interval_width_95": float((3.92 * scale).mean().item()),
        }
    )
    return metrics


def interpolation_baselines(
    context_x: Tensor,
    context_y: Tensor,
    target_x: Tensor,
) -> dict[str, Tensor]:
    """Context mean, nearest neighbour, and piecewise linear predictions."""

    if context_x.shape != context_y.shape or context_x.ndim != 3:
        raise ValueError("context_x and context_y must share a [B, C, 1] shape")
    if target_x.ndim != 3 or target_x.shape[0] != context_x.shape[0]:
        raise ValueError("target_x must have shape [B, T, 1]")
    if context_x.shape[-1] != 1 or target_x.shape[-1] != 1:
        raise ValueError("baseline helpers currently support scalar x/y only")
    if context_x.shape[1] < 2:
        raise ValueError("nearest and linear baselines require at least two Context points")

    order = context_x[..., 0].argsort(dim=1)
    x = torch.gather(context_x[..., 0], 1, order)
    y = torch.gather(context_y[..., 0], 1, order)
    query = target_x[..., 0]
    right = torch.searchsorted(x.contiguous(), query.contiguous()).clamp(
        1, x.shape[1] - 1
    )
    left = right - 1
    left_x = torch.gather(x, 1, left)
    right_x = torch.gather(x, 1, right)
    left_y = torch.gather(y, 1, left)
    right_y = torch.gather(y, 1, right)
    choose_right = (right_x - query).abs() < (query - left_x).abs()
    nearest = torch.where(choose_right, right_y, left_y)
    denominator = (right_x - left_x).clamp_min(torch.finfo(x.dtype).eps)
    linear = left_y + (query - left_x) / denominator * (right_y - left_y)
    context_mean = y.mean(dim=1, keepdim=True).expand_as(query)
    return {
        "context_mean": context_mean.unsqueeze(-1),
        "nearest": nearest.unsqueeze(-1),
        "linear": linear.unsqueeze(-1),
    }


def sinusoid_regression_baseline(
    context_x: Tensor,
    context_y: Tensor,
    target_x: Tensor,
    *,
    frequency_min: float = 0.5,
    frequency_max: float = 3.2,
    frequency_steps: int = 136,
    ridge: float = 1e-6,
) -> Tensor:
    """Fit ``b + alpha*sin(wx) + beta*cos(wx)`` over a frequency grid.

    This family-aware baseline is intentionally reported only for at least
    five Context points by the benchmark driver; with three free linear
    coefficients, smaller sets do not identify frequency reliably.
    """

    if context_x.shape != context_y.shape or context_x.ndim != 3:
        raise ValueError("context_x and context_y must share a [B, C, 1] shape")
    if target_x.ndim != 3 or target_x.shape[0] != context_x.shape[0]:
        raise ValueError("target_x must have shape [B, T, 1]")
    if context_x.shape[-1] != 1 or target_x.shape[-1] != 1:
        raise ValueError("sinusoid regression currently supports scalar x/y only")
    if frequency_steps < 2 or frequency_min <= 0 or frequency_min >= frequency_max:
        raise ValueError("invalid frequency search grid")
    if ridge <= 0:
        raise ValueError("ridge must be positive")

    original_dtype = context_x.dtype
    x = context_x[..., 0].to(torch.float64).cpu()
    y = context_y[..., 0].to(torch.float64).cpu()
    query = target_x[..., 0].to(torch.float64).cpu()
    frequencies = torch.linspace(
        frequency_min, frequency_max, frequency_steps, dtype=torch.float64
    )
    phase = x[:, None, :] * frequencies[None, :, None]
    design = torch.stack(
        (torch.ones_like(phase), torch.sin(phase), torch.cos(phase)), dim=-1
    )
    expanded_y = y[:, None, :, None].expand(-1, frequency_steps, -1, -1)
    transpose = design.transpose(-2, -1)
    gram = transpose @ design
    identity = torch.eye(3, dtype=torch.float64).view(1, 1, 3, 3)
    coefficients = torch.linalg.solve(
        gram + ridge * identity,
        transpose @ expanded_y,
    )
    fitted = (design @ coefficients).squeeze(-1)
    residual = (fitted - y[:, None, :]).square().mean(dim=-1)
    best_index = residual.argmin(dim=1)
    batch_index = torch.arange(x.shape[0])
    best_frequency = frequencies[best_index]
    best_coefficients = coefficients[batch_index, best_index, :, 0]

    target_phase = query * best_frequency[:, None]
    target_design = torch.stack(
        (
            torch.ones_like(target_phase),
            torch.sin(target_phase),
            torch.cos(target_phase),
        ),
        dim=-1,
    )
    prediction = (target_design * best_coefficients[:, None, :]).sum(dim=-1)
    return prediction.to(original_dtype).unsqueeze(-1)
