"""Inference-only runtime for the canonical CNP sanity-check desktop app.

The GUI deliberately lives outside this module.  Keeping checkpoint parsing,
Context construction, and prediction here makes the important contract easy
to test without creating a Tk window: the selected function name and the
Query truth are used for evaluation only and are never model inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from .model import CNP
from .sanity import (
    CANONICAL_FUNCTION_NAMES,
    CANONICAL_X_RANGE,
    evaluate_canonical_functions,
    make_canonical_evaluation_task,
)


SUITE_KEYS = ("single_sin", "single_cos", "sin_cos_pair")


@dataclass(frozen=True)
class SanitySuite:
    """Metadata for one independently trained canonical-function model."""

    key: str
    step: int
    function_names: tuple[str, ...]
    gate_passed: bool | None = None


@dataclass(frozen=True)
class SanityPredictionResult:
    """CPU tensors and scalar values needed to render one GUI prediction."""

    suite: SanitySuite
    function_name: str
    function_index: int
    context_count: int
    context_seed: int
    context_x: Tensor
    context_y: Tensor
    target_x: Tensor
    truth_y: Tensor
    cnp_mean: Tensor
    cnp_scale: Tensor
    query_x_value: float
    truth_query_y_value: float
    cnp_query_y_value: float
    cnp_query_scale_value: float

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
    def lower_95(self) -> float:
        return self.cnp_query_y - 1.96 * self.cnp_query_scale

    @property
    def upper_95(self) -> float:
        return self.cnp_query_y + 1.96 * self.cnp_query_scale

    @property
    def absolute_error(self) -> float:
        return abs(self.cnp_query_y - self.truth_query_y)

    @property
    def target_rmse(self) -> float:
        return float(torch.mean((self.cnp_mean - self.truth_y).square()).sqrt().item())


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


def _canonical_index(function_name: str) -> int:
    try:
        return tuple(CANONICAL_FUNCTION_NAMES).index(function_name)
    except ValueError as error:
        raise ValueError(
            f"checkpoint references unknown canonical function: {function_name!r}"
        ) from error


class SanityDemoRuntime:
    """Validated models and deterministic task construction for the GUI."""

    def __init__(
        self,
        *,
        models: Mapping[str, CNP],
        suites: tuple[SanitySuite, ...],
        model_config: Mapping[str, Any],
        x_range: tuple[float, float],
        checkpoint_path: Path,
        device: torch.device,
    ) -> None:
        if tuple(suite.key for suite in suites) != SUITE_KEYS:
            raise ValueError(f"checkpoint suites must be ordered as {SUITE_KEYS}")
        if set(models) != set(SUITE_KEYS):
            raise ValueError(f"checkpoint must contain exactly these suites: {SUITE_KEYS}")
        self.models = dict(models)
        self.suites = suites
        self.model_config = dict(model_config)
        self.x_range = x_range
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._suite_by_key = {suite.key: suite for suite in suites}

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
    ) -> "SanityDemoRuntime":
        """Load the three canonical models from a tensor/primitive checkpoint."""

        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"canonical sanity checkpoint not found: {checkpoint_path}")
        resolved_device = _resolve_device(device)
        checkpoint = _load_trusted_checkpoint(checkpoint_path, resolved_device)

        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported canonical sanity checkpoint format")
        if checkpoint.get("benchmark") != "canonical_sanity":
            raise ValueError("checkpoint is not a canonical_sanity benchmark")
        required = ("model_config", "models", "x_range")
        missing = [name for name in required if name not in checkpoint]
        if missing:
            raise ValueError(f"checkpoint is missing required fields: {', '.join(missing)}")

        model_config = dict(checkpoint["model_config"])
        if model_config.get("x_dim") != 1 or model_config.get("y_dim") != 1:
            raise ValueError("the sanity GUI requires one-dimensional X and Y models")

        raw_range = checkpoint["x_range"]
        if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
            raise ValueError("checkpoint x_range must contain two bounds")
        x_range = (float(raw_range[0]), float(raw_range[1]))
        if not all(math.isfinite(value) for value in x_range) or x_range[0] >= x_range[1]:
            raise ValueError("checkpoint x_range must contain increasing finite bounds")
        if not all(
            math.isclose(actual, expected)
            for actual, expected in zip(x_range, CANONICAL_X_RANGE)
        ):
            raise ValueError(
                "canonical sanity checkpoint x_range does not match the "
                "implemented canonical functions"
            )

        raw_models = checkpoint["models"]
        if not isinstance(raw_models, Mapping):
            raise ValueError("checkpoint models must be a mapping")
        missing_suites = [key for key in SUITE_KEYS if key not in raw_models]
        extra_suites = sorted(set(raw_models).difference(SUITE_KEYS))
        if missing_suites or extra_suites:
            raise ValueError(
                "checkpoint suite mismatch: "
                f"missing={missing_suites}, unexpected={extra_suites}"
            )

        models: dict[str, CNP] = {}
        suites: list[SanitySuite] = []
        for key in SUITE_KEYS:
            entry = raw_models[key]
            if not isinstance(entry, Mapping):
                raise ValueError(f"suite {key!r} must be a mapping")
            entry_missing = [
                name
                for name in ("step", "model_state_dict", "function_names")
                if name not in entry
            ]
            if entry_missing:
                raise ValueError(
                    f"suite {key!r} is missing fields: {', '.join(entry_missing)}"
                )
            function_names = tuple(str(name) for name in entry["function_names"])
            if not function_names or len(set(function_names)) != len(function_names):
                raise ValueError(f"suite {key!r} needs unique function names")
            for function_name in function_names:
                _canonical_index(function_name)

            step = int(entry["step"])
            if step < 1:
                raise ValueError(f"suite {key!r} has an invalid training step")
            raw_gate = entry.get("gate_passed")
            if raw_gate is not None and not isinstance(raw_gate, bool):
                raise ValueError(f"suite {key!r} gate_passed must be boolean")
            model = CNP(**model_config).to(resolved_device)
            model.load_state_dict(entry["model_state_dict"])
            model.eval()
            models[key] = model
            suites.append(
                SanitySuite(
                    key=key,
                    step=step,
                    function_names=function_names,
                    gate_passed=raw_gate,
                )
            )

        return cls(
            models=models,
            suites=tuple(suites),
            model_config=model_config,
            x_range=x_range,
            checkpoint_path=checkpoint_path,
            device=resolved_device,
        )

    def suite(self, key: str) -> SanitySuite:
        try:
            return self._suite_by_key[key]
        except KeyError as error:
            raise ValueError(f"unknown sanity suite: {key!r}") from error

    @torch.no_grad()
    def predict(
        self,
        suite_key: str,
        function_name: str,
        *,
        context_count: int,
        query_x: float,
        context_seed: int,
        grid_points: int = 129,
    ) -> SanityPredictionResult:
        """Predict one canonical function from Context pairs and Query X values."""

        suite = self.suite(suite_key)
        if function_name not in suite.function_names:
            raise ValueError(
                f"function {function_name!r} is not available in suite {suite_key!r}"
            )
        if isinstance(context_count, bool) or not isinstance(context_count, int):
            raise TypeError("context_count must be an integer")
        if not 1 <= context_count < grid_points:
            raise ValueError("context_count must be positive and smaller than grid_points")
        if isinstance(context_seed, bool) or not isinstance(context_seed, int):
            raise TypeError("context_seed must be an integer")
        requested_query_x = float(query_x)
        if not math.isfinite(requested_query_x):
            raise ValueError("query_x must be finite")
        if not self.x_range[0] <= requested_query_x <= self.x_range[1]:
            raise ValueError(
                f"query_x must be inside [{self.x_range[0]}, {self.x_range[1]}]"
            )

        function_index = _canonical_index(function_name)
        function_indices = torch.tensor([function_index], dtype=torch.long)
        task = make_canonical_evaluation_task(
            function_indices,
            context_count=context_count,
            grid_points=grid_points,
            seed=context_seed,
        )
        model = self.models[suite_key]
        exact_query_x = torch.tensor(
            [[[requested_query_x]]], dtype=task.target_x.dtype
        )
        model_target_x = torch.cat((task.target_x, exact_query_x), dim=1)
        all_mean, all_scale = model(
            task.context_x.to(self.device),
            task.context_y.to(self.device),
            model_target_x.to(self.device),
        )
        all_mean = all_mean.detach().cpu()
        all_scale = all_scale.detach().cpu()
        target_count = task.target_x.shape[1]
        mean = all_mean[:, :target_count]
        scale = all_scale[:, :target_count]
        query_mean = all_mean[:, target_count]
        query_scale = all_scale[:, target_count]
        query_truth = evaluate_canonical_functions(
            function_indices, exact_query_x
        )[:, 0]
        truth = task.target_y.detach().cpu()
        target_x = task.target_x.detach().cpu()
        context_x = task.context_x.detach().cpu()
        context_y = task.context_y.detach().cpu()
        expected_shape = truth.shape
        if mean.shape != expected_shape or scale.shape != expected_shape:
            raise RuntimeError("model returned an unexpected prediction shape")
        if (
            not torch.isfinite(all_mean).all()
            or not torch.isfinite(all_scale).all()
            or not torch.isfinite(query_truth).all()
        ):
            raise RuntimeError("model returned a non-finite prediction")
        if torch.any(all_scale <= 0):
            raise RuntimeError("model returned a non-positive predictive scale")

        return SanityPredictionResult(
            suite=suite,
            function_name=function_name,
            function_index=function_index,
            context_count=context_count,
            context_seed=context_seed,
            context_x=context_x[0, :, 0],
            context_y=context_y[0, :, 0],
            target_x=target_x[0, :, 0],
            truth_y=truth[0, :, 0],
            cnp_mean=mean[0, :, 0],
            cnp_scale=scale[0, :, 0],
            query_x_value=requested_query_x,
            truth_query_y_value=float(query_truth[0, 0].item()),
            cnp_query_y_value=float(query_mean[0, 0].item()),
            cnp_query_scale_value=float(query_scale[0, 0].item()),
        )


__all__ = [
    "SUITE_KEYS",
    "SanityDemoRuntime",
    "SanityPredictionResult",
    "SanitySuite",
]
