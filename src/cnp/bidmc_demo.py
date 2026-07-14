"""Inference helpers for the interactive BIDMC heart-rate CNP demo.

This module deliberately has no GUI dependency.  It owns the important part
of the demo contract: the selected query and its surrounding artificial gap
are removed from the context before inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from .datasets import (
    BIDMCWindow,
    HeartRateStandardizer,
    load_bidmc_records,
    make_bidmc_windows,
)
from .model import CNP


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class WindowEntry:
    """One selectable BIDMC window and its checkpoint split."""

    key: str
    split: str
    label: str
    window: BIDMCWindow


@dataclass(frozen=True)
class PredictionResult:
    """All values needed to render one query without running inference again."""

    entry: WindowEntry
    query_index: int
    gap_start: int
    gap_stop: int
    context_indices: Tensor
    seconds: Tensor
    truth_bpm: Tensor
    cnp_mean_bpm: Tensor
    cnp_scale_bpm: Tensor
    linear_bpm: Tensor

    @property
    def gap_width(self) -> int:
        return self.gap_stop - self.gap_start

    @property
    def cnp_query_bpm(self) -> float:
        return float(self.cnp_mean_bpm[self.query_index].item())

    @property
    def cnp_query_scale_bpm(self) -> float:
        return float(self.cnp_scale_bpm[self.query_index].item())

    @property
    def lower_95_bpm(self) -> float:
        return self.cnp_query_bpm - 1.96 * self.cnp_query_scale_bpm

    @property
    def upper_95_bpm(self) -> float:
        return self.cnp_query_bpm + 1.96 * self.cnp_query_scale_bpm

    @property
    def truth_query_bpm(self) -> float:
        return float(self.truth_bpm[self.query_index].item())

    @property
    def absolute_error_bpm(self) -> float:
        return abs(self.cnp_query_bpm - self.truth_query_bpm)

    @property
    def linear_query_bpm(self) -> float:
        return float(self.linear_bpm[self.query_index].item())


def centered_gap_indices(
    point_count: int,
    query_index: int,
    gap_width: int,
    *,
    margin: int = 5,
) -> tuple[Tensor, Tensor, int, int]:
    """Return context/gap indices for a fixed-width interpolation gap.

    The query always belongs to the gap and ``margin`` samples are retained at
    both ends.  Near a boundary the gap shifts rather than shrinking.
    """

    values = (point_count, query_index, gap_width, margin)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("point_count, query_index, gap_width, and margin must be integers")
    if point_count < 3:
        raise ValueError("an interpolation window needs at least three points")
    if margin < 1 or 2 * margin >= point_count:
        raise ValueError("margin must leave at least one possible gap point")
    first_query = margin
    last_query = point_count - margin - 1
    if not first_query <= query_index <= last_query:
        raise ValueError(
            f"query_index must be between {first_query} and {last_query}, got {query_index}"
        )
    max_width = point_count - 2 * margin
    if not 1 <= gap_width <= max_width:
        raise ValueError(f"gap_width must be between 1 and {max_width}, got {gap_width}")

    desired_start = query_index - (gap_width - 1) // 2
    first_start = margin
    last_start = point_count - margin - gap_width
    start = max(first_start, min(desired_start, last_start))
    stop = start + gap_width

    gap_indices = torch.arange(start, stop, dtype=torch.long)
    is_context = torch.ones(point_count, dtype=torch.bool)
    is_context[gap_indices] = False
    context_indices = torch.nonzero(is_context, as_tuple=False).flatten()
    if not bool((gap_indices == query_index).any()):
        raise RuntimeError("internal error: query was not included in the artificial gap")
    return context_indices, gap_indices, start, stop


def linear_interpolate(context_x: Tensor, context_y: Tensor, target_x: Tensor) -> Tensor:
    """Piecewise-linear baseline for one-dimensional, sorted or unsorted context."""

    if context_x.ndim != 2 or context_y.ndim != 2 or target_x.ndim != 2:
        raise ValueError("context_x, context_y, and target_x must have rank 2")
    if context_x.shape != context_y.shape or context_x.shape[-1] != 1:
        raise ValueError("this baseline expects matching [points, 1] context tensors")
    if target_x.shape[-1] != 1:
        raise ValueError("target_x must have shape [points, 1]")
    if context_x.shape[0] < 2:
        raise ValueError("linear interpolation needs at least two context points")

    x = context_x.flatten().detach().cpu()
    y = context_y.flatten().detach().cpu()
    query = target_x.flatten().detach().cpu()
    order = torch.argsort(x)
    x, y = x[order], y[order]
    if torch.any(x[1:] <= x[:-1]):
        raise ValueError("context_x values must be distinct")

    right = torch.searchsorted(x, query).clamp(1, x.numel() - 1)
    left = right - 1
    weight = (query - x[left]) / (x[right] - x[left])
    result = y[left] + weight * (y[right] - y[left])
    return result.reshape(target_x.shape).to(dtype=context_y.dtype)


@torch.no_grad()
def predict_window(
    model: CNP,
    entry: WindowEntry,
    standardizer: HeartRateStandardizer,
    query_index: int,
    gap_width: int,
    *,
    device: torch.device | str = "cpu",
    margin: int = 5,
) -> PredictionResult:
    """Predict a complete curve after hiding a gap containing ``query_index``."""

    window = entry.window
    point_count = int(window.time.shape[0])
    if window.time.shape != window.heart_rate.shape or window.time.shape[-1] != 1:
        raise ValueError("BIDMC demo windows must have matching [points, 1] tensors")
    context_indices, _gap_indices, start, stop = centered_gap_indices(
        point_count, query_index, gap_width, margin=margin
    )

    device = torch.device(device)
    context_x = window.time[context_indices]
    context_y = window.heart_rate[context_indices]
    model.eval()
    mean, scale = model(
        context_x.to(device),
        context_y.to(device),
        window.time.to(device),
    )
    mean = mean.detach().cpu()
    scale = scale.detach().cpu()
    if mean.shape != window.heart_rate.shape or scale.shape != window.heart_rate.shape:
        raise RuntimeError("model returned an unexpected prediction shape")
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
        raise RuntimeError("model returned a non-finite prediction")
    if torch.any(scale <= 0):
        raise RuntimeError("model returned a non-positive predictive scale")

    linear = linear_interpolate(context_x, context_y, window.time)
    return PredictionResult(
        entry=entry,
        query_index=query_index,
        gap_start=start,
        gap_stop=stop,
        context_indices=context_indices,
        seconds=torch.arange(point_count, dtype=torch.float32),
        truth_bpm=standardizer.denormalize(window.heart_rate.detach().cpu()).flatten(),
        cnp_mean_bpm=standardizer.denormalize(mean).flatten(),
        cnp_scale_bpm=standardizer.scale_to_bpm(scale).flatten(),
        linear_bpm=standardizer.denormalize(linear).flatten(),
    )


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
    """Load a tensor/primitive-only checkpoint without arbitrary pickle globals."""

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint must contain a mapping: {path}")
    return checkpoint


class BIDMCDemoRuntime:
    """Loaded model, normalization statistics, and checkpoint-bound windows."""

    def __init__(
        self,
        *,
        model: CNP,
        standardizer: HeartRateStandardizer,
        entries: tuple[WindowEntry, ...],
        device: torch.device,
        checkpoint_path: Path,
        checkpoint_step: int,
        margin: int = 5,
    ) -> None:
        if not entries:
            raise ValueError("the BIDMC demo needs at least one window")
        point_counts = {int(entry.window.time.shape[0]) for entry in entries}
        if len(point_counts) != 1:
            raise ValueError("all BIDMC demo windows must have the same number of points")
        point_count = next(iter(point_counts))
        if margin < 1 or 2 * margin >= point_count:
            raise ValueError("margin must leave at least one query point in every window")
        self.model = model
        self.standardizer = standardizer
        self.entries = entries
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.checkpoint_step = checkpoint_step
        self.margin = margin
        self._by_key = {entry.key: entry for entry in entries}
        if len(self._by_key) != len(entries):
            raise ValueError("duplicate window keys were generated")

    @classmethod
    def load(
        cls,
        data_dir: str | Path,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        margin: int = 5,
    ) -> "BIDMCDemoRuntime":
        """Load the exact split and normalization stored in a trusted checkpoint."""

        data_dir = Path(data_dir).expanduser().resolve()
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"BIDMC checkpoint not found: {checkpoint_path}")
        resolved_device = _resolve_device(device)
        checkpoint = _load_trusted_checkpoint(checkpoint_path, resolved_device)

        required = ("model_config", "model_state_dict", "normalization", "split")
        missing = [name for name in required if name not in checkpoint]
        if missing:
            raise ValueError(f"checkpoint is missing required fields: {', '.join(missing)}")
        model_config = dict(checkpoint["model_config"])
        if model_config.get("x_dim") != 1 or model_config.get("y_dim") != 1:
            raise ValueError("the BIDMC GUI requires a one-dimensional X and Y model")
        model = CNP(**model_config).to(resolved_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        normalization = checkpoint["normalization"]
        standardizer = HeartRateStandardizer(
            mean=float(normalization["mean_bpm"]),
            std=float(normalization["std_bpm_population"]),
        )
        training_config = checkpoint.get("training_config", {})
        window_size = int(training_config.get("window_size", 120))

        records = load_bidmc_records(data_dir)
        records_by_id = {record.record_id: record for record in records}
        split_manifest = checkpoint["split"]
        seen_record_ids: set[str] = set()
        patient_ids_by_split: dict[str, set[str]] = {}
        entries: list[WindowEntry] = []
        for split_name in SPLIT_NAMES:
            if split_name not in split_manifest:
                raise ValueError(f"checkpoint split is missing {split_name!r}")
            record_ids = tuple(split_manifest[split_name]["record_ids"])
            duplicates = seen_record_ids.intersection(record_ids)
            if duplicates:
                raise ValueError(f"records occur in multiple splits: {sorted(duplicates)}")
            seen_record_ids.update(record_ids)
            unknown = sorted(set(record_ids).difference(records_by_id))
            if unknown:
                raise ValueError(f"checkpoint references missing BIDMC records: {unknown}")
            split_records = tuple(records_by_id[record_id] for record_id in record_ids)
            actual_patient_ids = {record.patient_id for record in split_records}
            manifest_patient_ids = set(split_manifest[split_name].get("patient_ids", ()))
            if actual_patient_ids != manifest_patient_ids:
                raise ValueError(
                    f"checkpoint patient IDs do not match resolved records for {split_name}: "
                    f"manifest={sorted(manifest_patient_ids)}, actual={sorted(actual_patient_ids)}"
                )
            for previous_split, previous_ids in patient_ids_by_split.items():
                overlap = actual_patient_ids.intersection(previous_ids)
                if overlap:
                    raise ValueError(
                        f"patients occur in both {previous_split} and {split_name}: "
                        f"{sorted(overlap)}"
                    )
            patient_ids_by_split[split_name] = actual_patient_ids
            windows = make_bidmc_windows(
                split_records,
                standardizer,
                window_size=window_size,
            )
            for window in windows:
                key = f"{split_name}/{window.record_id}/{window.window_index}"
                label = (
                    f"{window.record_id} | patient {window.patient_id} | "
                    f"window {window.window_index + 1} (local 0–{window_size - 1} s)"
                )
                entries.append(
                    WindowEntry(
                        key=key,
                        split=split_name,
                        label=label,
                        window=window,
                    )
                )

        if not entries:
            raise ValueError("no complete BIDMC windows were produced")
        return cls(
            model=model,
            standardizer=standardizer,
            entries=tuple(entries),
            device=resolved_device,
            checkpoint_path=checkpoint_path,
            checkpoint_step=int(checkpoint.get("step", -1)),
            margin=margin,
        )

    @property
    def point_count(self) -> int:
        return int(self.entries[0].window.time.shape[0])

    @property
    def query_bounds(self) -> tuple[int, int]:
        return self.margin, self.point_count - self.margin - 1

    def entries_for_split(self, split: str) -> tuple[WindowEntry, ...]:
        if split not in SPLIT_NAMES:
            raise ValueError(f"unknown split: {split!r}")
        return tuple(entry for entry in self.entries if entry.split == split)

    def window_choices(self, split: str) -> list[tuple[str, str]]:
        return [(entry.label, entry.key) for entry in self.entries_for_split(split)]

    def predict(self, key: str, query_index: int, gap_width: int) -> PredictionResult:
        try:
            entry = self._by_key[key]
        except KeyError as error:
            raise ValueError(f"unknown BIDMC window key: {key!r}") from error
        return predict_window(
            self.model,
            entry,
            self.standardizer,
            query_index,
            gap_width,
            device=self.device,
            margin=self.margin,
        )


__all__ = [
    "BIDMCDemoRuntime",
    "PredictionResult",
    "SPLIT_NAMES",
    "WindowEntry",
    "centered_gap_indices",
    "linear_interpolate",
    "predict_window",
]
