"""RBF-Gaussian-process tasks and reference predictions for CNP regression.

The benchmark in this module is deliberately finite and reproducible.  A
function episode is a zero-mean Gaussian-process draw plus independent
Gaussian observation noise on a fixed grid, using the squared-exponential
(RBF) latent covariance

``signal_scale**2 * exp(-0.5 * ||x - x'||**2 / length_scale**2)``.

Training code can precompute :func:`make_gp_cholesky` once and pass the factor
to :func:`sample_rbf_training_batch`.  Each call still draws fresh standard
normal coefficients, so every batch contains newly sampled functions without
repeating the expensive factorization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .synthetic import deterministic_regression_metrics, gaussian_regression_metrics


RBF_GP_X_RANGE = (-2.0, 2.0)
RBF_GP_GRID_POINTS = 129


@dataclass(frozen=True)
class RBFKernelConfig:
    """Hyperparameters of the squared-exponential GP prior.

    ``signal_scale`` and ``observation_noise`` are standard deviations, not
    variances.  ``jitter`` is a small diagonal variance used only for stable
    Cholesky factorizations.  Sampled function values are noisy observations,
    and :func:`gp_posterior` returns the matching observed-target predictive
    uncertainty.
    """

    length_scale: float = 0.4
    signal_scale: float = 1.0
    observation_noise: float = 0.02
    jitter: float = 1e-6

    def __post_init__(self) -> None:
        for name in ("length_scale", "signal_scale", "observation_noise", "jitter"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.length_scale <= 0:
            raise ValueError("length_scale must be positive")
        if self.signal_scale <= 0:
            raise ValueError("signal_scale must be positive")
        if self.observation_noise < 0:
            raise ValueError("observation_noise cannot be negative")
        if self.jitter <= 0:
            raise ValueError("jitter must be positive")

    def as_dict(self) -> dict[str, float]:
        return {
            "length_scale": float(self.length_scale),
            "signal_scale": float(self.signal_scale),
            "observation_noise": float(self.observation_noise),
            "jitter": float(self.jitter),
        }


@dataclass(frozen=True)
class RBFTrainingBatch:
    """One batch of freshly drawn RBF-GP function episodes.

    Model inputs and targets live on ``device``.  The bookkeeping tensors and
    full sampled functions remain on CPU so seeded sampling is independent of
    the selected accelerator.
    """

    context_x: Tensor
    context_y: Tensor
    context_mask: Tensor
    target_x: Tensor
    target_y: Tensor
    context_indices: Tensor
    target_indices: Tensor
    context_counts: Tensor
    function_values: Tensor


@dataclass(frozen=True)
class RBFEvaluationTask:
    """A deterministic held-out Context/Query task for a GP function bank."""

    context_x: Tensor
    context_y: Tensor
    target_x: Tensor
    target_y: Tensor
    context_indices: Tensor
    target_indices: Tensor
    grid: Tensor
    function_values: Tensor
    context_count: int
    seed: int


@dataclass(frozen=True)
class RBFReferencePredictions:
    """Analytic GP oracle and context-mean baseline for one task."""

    gp_mean: Tensor
    gp_scale: Tensor
    context_mean: Tensor


def make_gp_grid(
    grid_points: int = RBF_GP_GRID_POINTS,
    x_range: tuple[float, float] = RBF_GP_X_RANGE,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return a scalar-input grid shaped ``[grid_points, 1]``."""

    if grid_points < 3:
        raise ValueError("grid_points must be at least 3")
    x_min, x_max = x_range
    if not math.isfinite(x_min) or not math.isfinite(x_max) or x_min >= x_max:
        raise ValueError("x_range must contain increasing finite bounds")
    if not dtype.is_floating_point:
        raise TypeError("GP grids require a floating-point dtype")
    return torch.linspace(
        x_min,
        x_max,
        grid_points,
        dtype=dtype,
        device=device,
    ).unsqueeze(-1)


def _validate_coordinate_pair(x1: Tensor, x2: Tensor) -> None:
    if x1.ndim < 2 or x2.ndim < 2:
        raise ValueError("kernel coordinates must have shape [..., points, dimensions]")
    if x1.shape[-1] != x2.shape[-1]:
        raise ValueError("kernel coordinates must have the same feature dimension")
    if x1.device != x2.device:
        raise ValueError("kernel coordinates must be on the same device")
    if x1.dtype != x2.dtype:
        raise ValueError("kernel coordinates must have the same dtype")
    if not x1.is_floating_point():
        raise TypeError("kernel coordinates must use a floating-point dtype")


def evaluate_rbf_kernel(
    x1: Tensor,
    x2: Tensor,
    kernel: RBFKernelConfig = RBFKernelConfig(),
) -> Tensor:
    """Evaluate the RBF covariance between two (optionally batched) point sets."""

    _validate_coordinate_pair(x1, x2)
    squared_distance = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).square().sum(dim=-1)
    return (kernel.signal_scale**2) * torch.exp(
        -0.5 * squared_distance / (kernel.length_scale**2)
    )


def _validate_grid(grid: Tensor) -> None:
    if grid.ndim != 2 or grid.shape[0] < 3 or grid.shape[1] < 1:
        raise ValueError("grid must have shape [points, dimensions] with at least 3 points")
    if not grid.is_floating_point():
        raise TypeError("grid must use a floating-point dtype")
    if not torch.isfinite(grid).all():
        raise ValueError("grid coordinates must be finite")


def make_gp_cholesky(
    grid: Tensor,
    kernel: RBFKernelConfig = RBFKernelConfig(),
) -> Tensor:
    """Factor the finite-grid covariance of noisy GP observations."""

    _validate_grid(grid)
    covariance = evaluate_rbf_kernel(grid, grid, kernel)
    identity = torch.eye(grid.shape[0], dtype=grid.dtype, device=grid.device)
    diagonal_variance = kernel.observation_noise**2 + kernel.jitter
    return torch.linalg.cholesky(covariance + diagonal_variance * identity)


def sample_gp_functions(
    count: int,
    grid: Tensor,
    kernel: RBFKernelConfig = RBFKernelConfig(),
    *,
    generator: torch.Generator,
    cholesky: Tensor | None = None,
) -> Tensor:
    """Draw reproducible noisy GP observations shaped ``[count, grid_points, 1]``.

    Sampling is intentionally CPU-based.  This makes a seed produce the same
    held-out function bank whether training later runs on CPU, CUDA, or MPS.
    """

    if count < 1:
        raise ValueError("count must be positive")
    _validate_grid(grid)
    if grid.device.type != "cpu":
        raise ValueError("deterministic GP function sampling requires a CPU grid")
    if generator.device.type != "cpu":
        raise ValueError("deterministic GP function sampling requires a CPU generator")
    if cholesky is None:
        cholesky = make_gp_cholesky(grid, kernel)
    expected_shape = (grid.shape[0], grid.shape[0])
    if tuple(cholesky.shape) != expected_shape:
        raise ValueError(
            f"cholesky must have shape {expected_shape}, got {tuple(cholesky.shape)}"
        )
    if cholesky.device != grid.device or cholesky.dtype != grid.dtype:
        raise ValueError("cholesky must share the grid device and dtype")

    # Coordinate-major inverse-transform sampling keeps the coefficient prefix
    # stable when a GUI appends one exact Query coordinate to an existing
    # display grid. ``torch.randn`` may consume its generator differently for
    # different requested shapes, whereas ``torch.rand`` preserves this prefix.
    uniform = torch.rand(
        grid.shape[0],
        count,
        generator=generator,
        dtype=grid.dtype,
        device=grid.device,
    )
    epsilon = torch.finfo(grid.dtype).eps
    uniform = uniform.clamp(min=epsilon, max=1.0 - epsilon)
    coefficients = (math.sqrt(2.0) * torch.erfinv(2.0 * uniform - 1.0)).transpose(0, 1)
    return (coefficients @ cholesky.transpose(-1, -2)).unsqueeze(-1)


def _gather_points(values: Tensor, indices: Tensor) -> Tensor:
    if values.ndim != 3 or indices.ndim != 2 or values.shape[0] != indices.shape[0]:
        raise ValueError("values and indices must have compatible [B, N, D]/[B, K] shapes")
    return torch.gather(
        values,
        1,
        indices.unsqueeze(-1).expand(-1, -1, values.shape[-1]),
    )


def sample_rbf_training_batch(
    *,
    grid: Tensor,
    kernel: RBFKernelConfig,
    batch_size: int,
    min_context: int,
    max_context: int,
    target_count: int,
    generator: torch.Generator,
    device: torch.device,
    cholesky: Tensor | None = None,
) -> RBFTrainingBatch:
    """Sample fresh GP functions and disjoint variable-size Context/Query sets."""

    _validate_grid(grid)
    if grid.device.type != "cpu":
        raise ValueError("the training sampling grid must remain on CPU")
    if generator.device.type != "cpu":
        raise ValueError("training sampling requires a CPU generator")
    if batch_size < 1 or target_count < 1:
        raise ValueError("batch_size and target_count must be positive")
    if not 1 <= min_context <= max_context:
        raise ValueError("expected 1 <= min_context <= max_context")
    if max_context + target_count > grid.shape[0]:
        raise ValueError("Context and Query counts exceed the finite GP grid")

    function_values = sample_gp_functions(
        batch_size,
        grid,
        kernel,
        generator=generator,
        cholesky=cholesky,
    )
    context_counts = torch.randint(
        min_context,
        max_context + 1,
        (batch_size,),
        generator=generator,
    )
    context_mask = (
        torch.arange(max_context).unsqueeze(0) < context_counts.unsqueeze(1)
    )

    # A random priority over grid coordinates gives disjoint Context/Query
    # indices.  Invalid padded Context slots remain harmless behind the mask.
    priorities = torch.rand(batch_size, grid.shape[0], generator=generator).argsort(dim=1)
    context_indices = priorities[:, :max_context]
    target_indices = priorities[:, max_context : max_context + target_count]
    expanded_grid = grid.unsqueeze(0).expand(batch_size, -1, -1)
    context_x = _gather_points(expanded_grid, context_indices)
    target_x = _gather_points(expanded_grid, target_indices)
    context_y = _gather_points(function_values, context_indices)
    target_y = _gather_points(function_values, target_indices)

    return RBFTrainingBatch(
        context_x=context_x.to(device),
        context_y=context_y.to(device),
        context_mask=context_mask.to(device),
        target_x=target_x.to(device),
        target_y=target_y.to(device),
        context_indices=context_indices,
        target_indices=target_indices,
        context_counts=context_counts,
        function_values=function_values,
    )


def _validate_function_bank(function_values: Tensor, grid: Tensor) -> None:
    _validate_grid(grid)
    if function_values.ndim != 3 or function_values.shape[-1] != 1:
        raise ValueError("function_values must have shape [functions, grid_points, 1]")
    if function_values.shape[0] < 1 or function_values.shape[1] != grid.shape[0]:
        raise ValueError("function_values must contain at least one value on every grid point")
    if function_values.device != grid.device or function_values.dtype != grid.dtype:
        raise ValueError("function_values must share the grid device and dtype")
    if not torch.isfinite(function_values).all():
        raise ValueError("function_values must be finite")


def make_rbf_evaluation_task(
    function_values: Tensor,
    grid: Tensor,
    *,
    context_count: int,
    seed: int,
) -> RBFEvaluationTask:
    """Build a held-out task with nested Context sets and fixed Query points.

    Even grid positions are Context candidates and odd positions are Queries.
    Reusing ``seed`` across Context counts therefore makes smaller Context sets
    subsets of larger ones while preserving exactly the same Query set.
    """

    _validate_function_bank(function_values, grid)
    if context_count < 1:
        raise ValueError("context_count must be positive")
    context_candidates = torch.arange(0, grid.shape[0], 2, device=grid.device)
    target_candidates = torch.arange(1, grid.shape[0], 2, device=grid.device)
    if context_count > context_candidates.numel():
        raise ValueError("context_count exceeds the finite evaluation candidates")
    if target_candidates.numel() < 1:
        raise ValueError("the evaluation grid does not contain any Query points")
    if grid.device.type != "cpu":
        raise ValueError("deterministic evaluation task construction requires CPU tensors")

    generator = torch.Generator().manual_seed(seed)
    scores = torch.rand(
        function_values.shape[0],
        context_candidates.numel(),
        generator=generator,
    )
    priorities = context_candidates[scores.argsort(dim=1)]
    context_indices = priorities[:, :context_count].sort(dim=1).values
    target_indices = target_candidates.view(1, -1).expand(function_values.shape[0], -1)
    expanded_grid = grid.unsqueeze(0).expand(function_values.shape[0], -1, -1)

    return RBFEvaluationTask(
        context_x=_gather_points(expanded_grid, context_indices),
        context_y=_gather_points(function_values, context_indices),
        target_x=_gather_points(expanded_grid, target_indices),
        target_y=_gather_points(function_values, target_indices),
        context_indices=context_indices,
        target_indices=target_indices,
        grid=grid,
        function_values=function_values,
        context_count=context_count,
        seed=seed,
    )


def _with_batch(tensor: Tensor, name: str) -> tuple[Tensor, bool]:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0), True
    if tensor.ndim == 3:
        return tensor, False
    raise ValueError(f"{name} must have rank 2 or 3, got shape {tuple(tensor.shape)}")


def gp_posterior(
    context_x: Tensor,
    context_y: Tensor,
    target_x: Tensor,
    kernel: RBFKernelConfig = RBFKernelConfig(),
) -> tuple[Tensor, Tensor]:
    """Return analytic observed-target GP posterior ``(mean, scale)``.

    The returned variance includes independent target observation noise, so it
    is directly comparable to values from :func:`sample_gp_functions`.
    """

    context_x, context_unbatched = _with_batch(context_x, "context_x")
    context_y, y_unbatched = _with_batch(context_y, "context_y")
    target_x, target_unbatched = _with_batch(target_x, "target_x")
    if context_unbatched != y_unbatched or context_unbatched != target_unbatched:
        raise ValueError("Context and target tensors must use the same batching convention")
    if context_x.shape[:2] != context_y.shape[:2]:
        raise ValueError("context_x and context_y must share batch and Context axes")
    if context_x.shape[0] != target_x.shape[0]:
        raise ValueError("Context and target batch sizes must match")
    if context_x.shape[1] < 1 or context_y.shape[-1] < 1:
        raise ValueError("at least one Context point and one output are required")
    _validate_coordinate_pair(context_x, target_x)
    if context_y.device != context_x.device or context_y.dtype != context_x.dtype:
        raise ValueError("context_y must share the Context coordinate device and dtype")

    context_covariance = evaluate_rbf_kernel(context_x, context_x, kernel)
    target_context_covariance = evaluate_rbf_kernel(target_x, context_x, kernel)
    identity = torch.eye(
        context_x.shape[1],
        dtype=context_x.dtype,
        device=context_x.device,
    ).unsqueeze(0)
    regularization = kernel.observation_noise**2 + kernel.jitter
    cholesky = torch.linalg.cholesky(context_covariance + regularization * identity)
    alpha = torch.cholesky_solve(context_y, cholesky)
    mean = target_context_covariance @ alpha

    solved = torch.cholesky_solve(
        target_context_covariance.transpose(-1, -2),
        cholesky,
    )
    variance_reduction = (
        target_context_covariance * solved.transpose(-1, -2)
    ).sum(dim=-1)
    variance_floor = max(
        kernel.observation_noise**2 + kernel.jitter,
        torch.finfo(target_x.dtype).eps,
    )
    target_prior_variance = kernel.signal_scale**2 + kernel.observation_noise**2
    variance = (target_prior_variance - variance_reduction).clamp_min(variance_floor)
    scale = variance.sqrt().unsqueeze(-1).expand(-1, -1, context_y.shape[-1])

    if context_unbatched:
        return mean.squeeze(0), scale.squeeze(0)
    return mean, scale


def rbf_reference_predictions(
    task: RBFEvaluationTask,
    kernel: RBFKernelConfig = RBFKernelConfig(),
) -> RBFReferencePredictions:
    """Compute the exact-kernel oracle and a context-mean baseline."""

    gp_mean, gp_scale = gp_posterior(
        task.context_x,
        task.context_y,
        task.target_x,
        kernel,
    )
    context_mean = task.context_y.mean(dim=1, keepdim=True).expand_as(task.target_y)
    return RBFReferencePredictions(
        gp_mean=gp_mean,
        gp_scale=gp_scale,
        context_mean=context_mean,
    )


def rbf_regression_metrics(
    cnp_mean: Tensor,
    cnp_scale: Tensor,
    task: RBFEvaluationTask,
    kernel: RBFKernelConfig = RBFKernelConfig(),
) -> dict[str, dict[str, float | int]]:
    """Evaluate CNP predictions beside the GP oracle and context-mean baseline."""

    references = rbf_reference_predictions(task, kernel)
    return {
        "cnp": gaussian_regression_metrics(cnp_mean, cnp_scale, task.target_y),
        "gp_posterior": gaussian_regression_metrics(
            references.gp_mean,
            references.gp_scale,
            task.target_y,
        ),
        "context_mean": deterministic_regression_metrics(
            references.context_mean,
            task.target_y,
        ),
    }
