"""Inference-only runtime for the native RBF-GP CNP explorer.

The desktop UI deliberately stays outside this module.  This runtime owns the
scientifically important boundary: it reconstructs a held-out GP draw from a
seed, chooses Context pairs independently of the requested Query, and gives
the model only ``Context (X, Y)`` and ``Query X``.  Ground-truth Query Y is
retained exclusively for evaluation and display.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from .model import CNP
from .rbf_gp import (
    RBFKernelConfig,
    gp_posterior,
    make_gp_cholesky,
    make_gp_grid,
    make_rbf_evaluation_task,
    sample_gp_functions,
)


_KERNEL_FIELDS = (
    "length_scale",
    "signal_scale",
    "observation_noise",
    "jitter",
)


@dataclass(frozen=True)
class RBFGPPredictionResult:
    """CPU values needed to render one seeded RBF-GP prediction."""

    function_seed: int
    context_count: int
    context_seed: int
    training_max_context: int
    context_is_ood: bool
    context_x: Tensor
    context_y: Tensor
    target_x: Tensor
    truth_y: Tensor
    cnp_mean: Tensor
    cnp_scale: Tensor
    gp_mean: Tensor
    gp_scale: Tensor
    query_x_value: float
    truth_query_y_value: float
    cnp_query_y_value: float
    cnp_query_scale_value: float
    gp_query_y_value: float
    gp_query_scale_value: float

    @property
    def query_x(self) -> float:
        return self.query_x_value

    @property
    def truth_query_y(self) -> float:
        return self.truth_query_y_value

    @property
    def cnp_query_y(self) -> float:
        return self.cnp_query_y_value

    @property
    def cnp_query_scale(self) -> float:
        return self.cnp_query_scale_value

    @property
    def gp_query_y(self) -> float:
        return self.gp_query_y_value

    @property
    def gp_query_scale(self) -> float:
        return self.gp_query_scale_value

    @property
    def cnp_lower_95(self) -> float:
        return self.cnp_query_y - 1.96 * self.cnp_query_scale

    @property
    def cnp_upper_95(self) -> float:
        return self.cnp_query_y + 1.96 * self.cnp_query_scale

    @property
    def gp_lower_95(self) -> float:
        return self.gp_query_y - 1.96 * self.gp_query_scale

    @property
    def gp_upper_95(self) -> float:
        return self.gp_query_y + 1.96 * self.gp_query_scale

    @property
    def cnp_absolute_error(self) -> float:
        return abs(self.cnp_query_y - self.truth_query_y)

    @property
    def gp_absolute_error(self) -> float:
        return abs(self.gp_query_y - self.truth_query_y)

    @property
    def cnp_target_rmse(self) -> float:
        return float(torch.mean((self.cnp_mean - self.truth_y).square()).sqrt().item())

    @property
    def gp_target_rmse(self) -> float:
        return float(torch.mean((self.gp_mean - self.truth_y).square()).sqrt().item())

    @property
    def cnp_rmse(self) -> float:
        """Alias used by the desktop metric cards."""

        return self.cnp_target_rmse

    @property
    def gp_rmse(self) -> float:
        """Alias used by the desktop metric cards."""

        return self.gp_target_rmse

    @property
    def context_out_of_training_range(self) -> bool:
        return self.context_is_ood


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return device


def _load_trusted_checkpoint(path: Path, device: torch.device) -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint must contain a mapping: {path}")
    return checkpoint


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"checkpoint {name} must be a positive integer")
    return int(value)


def _seed(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value < 2**63:
        raise ValueError(f"{name} must be between 0 and 2**63 - 1")
    return int(value)


def _kernel_from_mapping(raw_kernel: Any) -> RBFKernelConfig:
    if not isinstance(raw_kernel, Mapping):
        raise ValueError("checkpoint kernel must be a mapping")
    missing = [name for name in _KERNEL_FIELDS if name not in raw_kernel]
    extra = sorted(set(raw_kernel).difference(_KERNEL_FIELDS))
    if missing or extra:
        raise ValueError(
            "checkpoint kernel field mismatch: "
            f"missing={missing}, unexpected={extra}"
        )
    values: dict[str, float] = {}
    for name in _KERNEL_FIELDS:
        value = raw_kernel[name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"checkpoint kernel {name} must be a real number")
        values[name] = float(value)
    try:
        return RBFKernelConfig(**values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid checkpoint RBF kernel: {error}") from error


class RBFGPDemoRuntime:
    """A validated RBF-GP checkpoint and deterministic GUI task builder."""

    def __init__(
        self,
        *,
        model: CNP,
        model_config: Mapping[str, Any],
        kernel: RBFKernelConfig,
        x_range: tuple[float, float],
        grid_points: int,
        step: int,
        training_config: Mapping[str, Any],
        training_max_context: int,
        gate_passed: bool | None,
        test_summary: Mapping[str, Any] | None,
        checkpoint_path: Path,
        device: torch.device,
    ) -> None:
        self.model = model
        self.model_config = dict(model_config)
        self.kernel = kernel
        self.x_range = x_range
        self.grid_points = grid_points
        self.step = step
        self.training_config = dict(training_config)
        self.training_max_context = training_max_context
        self.gate_passed = gate_passed
        self.test_summary = None if test_summary is None else dict(test_summary)
        self.checkpoint_path = checkpoint_path
        self.device = device

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
    ) -> "RBFGPDemoRuntime":
        """Load and validate a tensor/primitive ``rbf_gp`` checkpoint."""

        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"RBF-GP checkpoint not found: {checkpoint_path}")
        resolved_device = _resolve_device(device)
        checkpoint = _load_trusted_checkpoint(checkpoint_path, resolved_device)

        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported RBF-GP checkpoint format")
        if checkpoint.get("benchmark") != "rbf_gp":
            raise ValueError("checkpoint is not an rbf_gp benchmark")
        required = (
            "step",
            "model_config",
            "model_state_dict",
            "x_range",
            "grid_points",
            "kernel",
            "training_config",
        )
        missing = [name for name in required if name not in checkpoint]
        if missing:
            raise ValueError(f"checkpoint is missing required fields: {', '.join(missing)}")

        step = _positive_integer(checkpoint["step"], "step")
        raw_model_config = checkpoint["model_config"]
        if not isinstance(raw_model_config, Mapping):
            raise ValueError("checkpoint model_config must be a mapping")
        model_config = dict(raw_model_config)
        if model_config.get("x_dim") != 1 or model_config.get("y_dim") != 1:
            raise ValueError("the RBF-GP GUI requires one-dimensional X and Y models")

        raw_range = checkpoint["x_range"]
        if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
            raise ValueError("checkpoint x_range must contain two bounds")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw_range):
            raise ValueError("checkpoint x_range bounds must be real numbers")
        x_range = (float(raw_range[0]), float(raw_range[1]))
        if not all(math.isfinite(value) for value in x_range) or x_range[0] >= x_range[1]:
            raise ValueError("checkpoint x_range must contain increasing finite bounds")

        grid_points = _positive_integer(checkpoint["grid_points"], "grid_points")
        if grid_points < 3:
            raise ValueError("checkpoint grid_points must be at least 3")
        kernel = _kernel_from_mapping(checkpoint["kernel"])

        raw_training_config = checkpoint["training_config"]
        if not isinstance(raw_training_config, Mapping):
            raise ValueError("checkpoint training_config must be a mapping")
        training_config = dict(raw_training_config)
        if "max_context" not in training_config:
            raise ValueError("checkpoint training_config is missing max_context")
        training_max_context = _positive_integer(
            training_config["max_context"], "training_config.max_context"
        )

        raw_gate = checkpoint.get("gate_passed")
        if raw_gate is not None and not isinstance(raw_gate, bool):
            raise ValueError("checkpoint gate_passed must be boolean")
        raw_test_summary = checkpoint.get("test_summary")
        if raw_test_summary is not None and not isinstance(raw_test_summary, Mapping):
            raise ValueError("checkpoint test_summary must be a mapping")

        raw_state = checkpoint["model_state_dict"]
        if not isinstance(raw_state, Mapping):
            raise ValueError("checkpoint model_state_dict must be a mapping")
        try:
            model = CNP(**model_config).to(resolved_device)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid checkpoint model_config: {error}") from error
        try:
            model.load_state_dict(raw_state)
        except RuntimeError as error:
            raise ValueError(f"checkpoint model_state_dict is incompatible: {error}") from error
        model.eval()

        return cls(
            model=model,
            model_config=model_config,
            kernel=kernel,
            x_range=x_range,
            grid_points=grid_points,
            step=step,
            training_config=training_config,
            training_max_context=training_max_context,
            gate_passed=raw_gate,
            test_summary=raw_test_summary,
            checkpoint_path=checkpoint_path,
            device=resolved_device,
        )

    @torch.no_grad()
    def predict(
        self,
        function_seed: int,
        *,
        context_count: int,
        context_seed: int,
        query_x: float,
        display_grid_points: int = 401,
    ) -> RBFGPPredictionResult:
        """Condition the CNP and exact GP oracle on one reproducible function.

        The exact Query coordinate is appended *after* a fixed display grid
        before drawing the function.  A lower-triangular Cholesky factor means
        the first ``display_grid_points`` values depend only on the fixed grid
        and the first seeded coefficients.  Moving Query X therefore changes
        the exact Query truth without resampling the displayed curve or Context.
        """

        function_seed = _seed(function_seed, "function_seed")
        context_seed = _seed(context_seed, "context_seed")
        if isinstance(context_count, bool) or not isinstance(context_count, int):
            raise TypeError("context_count must be an integer")
        if context_count < 1:
            raise ValueError("context_count must be positive")
        if isinstance(display_grid_points, bool) or not isinstance(display_grid_points, int):
            raise TypeError("display_grid_points must be an integer")
        if display_grid_points < 3:
            raise ValueError("display_grid_points must be at least 3")
        context_candidates = (display_grid_points + 1) // 2
        if context_count > context_candidates:
            raise ValueError(
                "context_count exceeds the fixed display-grid Context candidates"
            )
        if isinstance(query_x, bool):
            raise TypeError("query_x must be a real number")
        try:
            requested_query_x = float(query_x)
        except (TypeError, ValueError) as error:
            raise TypeError("query_x must be a real number") from error
        if not math.isfinite(requested_query_x):
            raise ValueError("query_x must be finite")
        if not self.x_range[0] <= requested_query_x <= self.x_range[1]:
            raise ValueError(
                f"query_x must be inside [{self.x_range[0]}, {self.x_range[1]}]"
            )

        display_grid = make_gp_grid(
            display_grid_points,
            x_range=self.x_range,
            dtype=torch.float32,
            device="cpu",
        )
        exact_query_x = torch.tensor([[requested_query_x]], dtype=display_grid.dtype)
        joint_grid = torch.cat((display_grid, exact_query_x), dim=0)
        joint_cholesky = make_gp_cholesky(joint_grid, self.kernel)
        joint_values = sample_gp_functions(
            1,
            joint_grid,
            self.kernel,
            generator=torch.Generator().manual_seed(function_seed),
            cholesky=joint_cholesky,
        )
        display_values = joint_values[:, :display_grid_points]
        query_truth = joint_values[:, display_grid_points]

        task = make_rbf_evaluation_task(
            display_values,
            display_grid,
            context_count=context_count,
            seed=context_seed,
        )
        model_target_x = torch.cat((task.target_x, exact_query_x.view(1, 1, 1)), dim=1)
        all_cnp_mean, all_cnp_scale = self.model(
            task.context_x.to(self.device),
            task.context_y.to(self.device),
            model_target_x.to(self.device),
        )
        all_cnp_mean = all_cnp_mean.detach().cpu()
        all_cnp_scale = all_cnp_scale.detach().cpu()
        all_gp_mean, all_gp_scale = gp_posterior(
            task.context_x,
            task.context_y,
            model_target_x,
            self.kernel,
        )
        all_gp_mean = all_gp_mean.detach().cpu()
        all_gp_scale = all_gp_scale.detach().cpu()

        target_count = task.target_x.shape[1]
        cnp_mean = all_cnp_mean[:, :target_count]
        cnp_scale = all_cnp_scale[:, :target_count]
        gp_mean = all_gp_mean[:, :target_count]
        gp_scale = all_gp_scale[:, :target_count]
        cnp_query_mean = all_cnp_mean[:, target_count]
        cnp_query_scale = all_cnp_scale[:, target_count]
        gp_query_mean = all_gp_mean[:, target_count]
        gp_query_scale = all_gp_scale[:, target_count]

        expected_shape = task.target_y.shape
        predictions = (cnp_mean, cnp_scale, gp_mean, gp_scale)
        if any(value.shape != expected_shape for value in predictions):
            raise RuntimeError("CNP or GP returned an unexpected prediction shape")
        finite_values = predictions + (
            cnp_query_mean,
            cnp_query_scale,
            gp_query_mean,
            gp_query_scale,
            query_truth,
        )
        if any(not torch.isfinite(value).all() for value in finite_values):
            raise RuntimeError("CNP or GP returned a non-finite prediction")
        if torch.any(all_cnp_scale <= 0) or torch.any(all_gp_scale <= 0):
            raise RuntimeError("CNP or GP returned a non-positive predictive scale")

        return RBFGPPredictionResult(
            function_seed=function_seed,
            context_count=context_count,
            context_seed=context_seed,
            training_max_context=self.training_max_context,
            context_is_ood=context_count > self.training_max_context,
            context_x=task.context_x[0, :, 0].detach().cpu(),
            context_y=task.context_y[0, :, 0].detach().cpu(),
            target_x=task.target_x[0, :, 0].detach().cpu(),
            truth_y=task.target_y[0, :, 0].detach().cpu(),
            cnp_mean=cnp_mean[0, :, 0],
            cnp_scale=cnp_scale[0, :, 0],
            gp_mean=gp_mean[0, :, 0],
            gp_scale=gp_scale[0, :, 0],
            query_x_value=requested_query_x,
            truth_query_y_value=float(query_truth[0, 0].item()),
            cnp_query_y_value=float(cnp_query_mean[0, 0].item()),
            cnp_query_scale_value=float(cnp_query_scale[0, 0].item()),
            gp_query_y_value=float(gp_query_mean[0, 0].item()),
            gp_query_scale_value=float(gp_query_scale[0, 0].item()),
        )


__all__ = ["RBFGPDemoRuntime", "RBFGPPredictionResult"]
