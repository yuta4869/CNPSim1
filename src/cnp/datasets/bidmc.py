"""BIDMC heart-rate loading, subject splits, and CNP windows."""

from __future__ import annotations

import csv
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor


_NUMERICS_RE = re.compile(r"^(bidmc_\d{2})_Numerics\.csv$")
_PATIENT_ID_KEY = "MIMIC II matched wdb ID"


@dataclass(frozen=True, eq=False)
class BIDMCRecord:
    """One BIDMC recording before windowing.

    ``time`` and ``heart_rate`` are float32 tensors with shape ``[N, 1]``.
    Missing HR entries are retained as IEEE NaNs so callers can decide how to
    handle them; :func:`make_bidmc_windows` rejects any affected window.
    """

    record_id: str
    patient_id: str
    time: Tensor
    heart_rate: Tensor
    metadata: Mapping[str, str]


@dataclass(frozen=True, eq=False)
class BIDMCWindow:
    """A fixed-length function observation ready for CNP sampling."""

    record_id: str
    patient_id: str
    window_index: int
    time: Tensor
    heart_rate: Tensor

    @property
    def x(self) -> Tensor:
        return self.time

    @property
    def y(self) -> Tensor:
        return self.heart_rate


@dataclass(frozen=True)
class BIDMCSplit:
    """A patient-disjoint collection of train/validation/test records."""

    train: tuple[BIDMCRecord, ...]
    validation: tuple[BIDMCRecord, ...]
    test: tuple[BIDMCRecord, ...]

    @property
    def val(self) -> tuple[BIDMCRecord, ...]:
        return self.validation

    @property
    def train_patient_ids(self) -> tuple[str, ...]:
        return tuple(sorted({record.patient_id for record in self.train}))

    @property
    def validation_patient_ids(self) -> tuple[str, ...]:
        return tuple(sorted({record.patient_id for record in self.validation}))

    @property
    def test_patient_ids(self) -> tuple[str, ...]:
        return tuple(sorted({record.patient_id for record in self.test}))


@dataclass(frozen=True)
class HeartRateStandardizer:
    """Training-set heart-rate statistics, expressed in BPM."""

    mean: float
    std: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean):
            raise ValueError("mean must be finite")
        if not math.isfinite(self.std) or self.std <= 0:
            raise ValueError("std must be finite and positive")

    def normalize(self, heart_rate: Tensor) -> Tensor:
        return (heart_rate - self.mean) / self.std

    def denormalize(self, normalized_heart_rate: Tensor) -> Tensor:
        return normalized_heart_rate * self.std + self.mean

    def scale_to_bpm(self, normalized_scale: Tensor) -> Tensor:
        """Convert a predicted normalized standard deviation to BPM."""

        return normalized_scale * self.std


def _raw_directory(root: Path) -> Path:
    nested = root / "raw"
    return nested if nested.is_dir() else root


def _read_fix(path: Path, expected_record_id: str) -> tuple[str, Mapping[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    nonempty = [line.strip() for line in lines if line.strip()]
    if not nonempty:
        raise ValueError(f"Empty BIDMC Fix file: {path}")
    record_id = nonempty[0]
    if record_id != expected_record_id:
        raise ValueError(
            f"Record ID mismatch in {path}: expected {expected_record_id!r}, got {record_id!r}"
        )

    metadata: dict[str, str] = {}
    for line_number, line in enumerate(nonempty[1:], start=2):
        if ":" not in line:
            raise ValueError(f"Malformed metadata at {path}:{line_number}: {line!r}")
        key, value = line.split(":", maxsplit=1)
        key, value = key.strip(), value.strip()
        if not key:
            raise ValueError(f"Empty metadata key at {path}:{line_number}")
        metadata[key] = value

    patient_id = metadata.get(_PATIENT_ID_KEY, "").strip()
    if not patient_id:
        raise ValueError(f"Missing {_PATIENT_ID_KEY!r} in {path}")
    return patient_id, MappingProxyType(metadata)


def _number(value: str | None, *, missing_as_nan: bool, path: Path, line: int) -> float:
    stripped = "" if value is None else value.strip()
    if not stripped and missing_as_nan:
        return math.nan
    try:
        return float(stripped)
    except ValueError as error:
        raise ValueError(f"Invalid numeric value {value!r} at {path}:{line}") from error


def _read_numerics(path: Path) -> tuple[Tensor, Tensor]:
    times: list[float] = []
    heart_rates: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        # Explicit stripping also handles files whose whitespace is not after
        # the delimiter (and makes fixture files follow the same path).
        fieldnames = {field.strip(): field for field in reader.fieldnames if field is not None}
        try:
            time_key = fieldnames["Time [s]"]
            heart_rate_key = fieldnames["HR"]
        except KeyError as error:
            raise ValueError(f"Expected 'Time [s]' and 'HR' columns in {path}") from error

        for line_number, row in enumerate(reader, start=2):
            time_value = _number(
                row.get(time_key), missing_as_nan=False, path=path, line=line_number
            )
            if not math.isfinite(time_value):
                raise ValueError(f"Non-finite time at {path}:{line_number}")
            times.append(time_value)
            heart_rates.append(
                _number(row.get(heart_rate_key), missing_as_nan=True, path=path, line=line_number)
            )

    if not times:
        raise ValueError(f"No numeric observations in {path}")
    if any(current <= previous for previous, current in zip(times, times[1:])):
        raise ValueError(f"Time must be strictly increasing in {path}")
    return (
        torch.tensor(times, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(heart_rates, dtype=torch.float32).unsqueeze(-1),
    )


def load_bidmc_records(root: str | Path) -> tuple[BIDMCRecord, ...]:
    """Load every paired ``Numerics.csv``/``Fix.txt`` record under *root*.

    *root* may point either to ``data/bidmc`` or directly to its ``raw``
    directory. Records are returned in numeric file-name order.
    """

    raw = _raw_directory(Path(root).expanduser())
    numerics_by_id: dict[str, Path] = {}
    for path in raw.glob("bidmc_*_Numerics.csv"):
        match = _NUMERICS_RE.fullmatch(path.name)
        if match is not None:
            numerics_by_id[match.group(1)] = path
    if not numerics_by_id:
        raise FileNotFoundError(f"No BIDMC Numerics files found under {raw}")

    records: list[BIDMCRecord] = []
    for record_id in sorted(numerics_by_id):
        numerics_path = numerics_by_id[record_id]
        fix_path = raw / f"{record_id}_Fix.txt"
        if not fix_path.is_file():
            raise FileNotFoundError(f"Missing Fix file for {record_id}: {fix_path}")
        patient_id, metadata = _read_fix(fix_path, record_id)
        time, heart_rate = _read_numerics(numerics_path)
        records.append(
            BIDMCRecord(
                record_id=record_id,
                patient_id=patient_id,
                time=time,
                heart_rate=heart_rate,
                metadata=metadata,
            )
        )
    return tuple(records)


def split_records_by_subject(
    records: Sequence[BIDMCRecord],
    *,
    seed: int = 0,
    train_subjects: int = 32,
    validation_subjects: int = 7,
    test_subjects: int = 7,
) -> BIDMCSplit:
    """Make a reproducible patient-disjoint 32/7/7 split by default."""

    counts = (train_subjects, validation_subjects, test_subjects)
    if any(count < 0 for count in counts) or sum(counts) == 0:
        raise ValueError("split subject counts must be non-negative and not all zero")

    grouped: dict[str, list[BIDMCRecord]] = {}
    for record in records:
        grouped.setdefault(record.patient_id, []).append(record)
    expected_subjects = sum(counts)
    if len(grouped) != expected_subjects:
        raise ValueError(
            f"Split requests {expected_subjects} subjects but received {len(grouped)} unique subjects"
        )

    patient_ids = sorted(grouped)
    random.Random(seed).shuffle(patient_ids)
    train_end = train_subjects
    validation_end = train_end + validation_subjects
    train_ids = patient_ids[:train_end]
    validation_ids = patient_ids[train_end:validation_end]
    test_ids = patient_ids[validation_end:]

    def collect(ids: Sequence[str]) -> tuple[BIDMCRecord, ...]:
        return tuple(
            record
            for patient_id in ids
            for record in sorted(grouped[patient_id], key=lambda item: item.record_id)
        )

    return BIDMCSplit(
        train=collect(train_ids),
        validation=collect(validation_ids),
        test=collect(test_ids),
    )


def fit_heart_rate_standardizer(
    records: Iterable[BIDMCRecord | BIDMCWindow],
    *,
    max_points: int = 480,
) -> HeartRateStandardizer:
    """Fit population mean/std from training records only.

    The default truncation mirrors :func:`make_bidmc_windows`, ensuring that
    the unused 481st sample cannot influence training statistics. NaNs are
    ignored here; complete-window filtering is performed during windowing.
    """

    if max_points < 1:
        raise ValueError("max_points must be positive")
    finite_values: list[Tensor] = []
    for record in records:
        values = record.heart_rate[:max_points].reshape(-1)
        values = values[torch.isfinite(values)]
        if values.numel():
            finite_values.append(values.to(dtype=torch.float64, device="cpu"))
    if not finite_values:
        raise ValueError("Cannot fit heart-rate statistics without finite training values")

    combined = torch.cat(finite_values)
    mean = float(combined.mean().item())
    std = float(combined.std(correction=0).item())
    return HeartRateStandardizer(mean=mean, std=std)


def make_bidmc_windows(
    records: Iterable[BIDMCRecord],
    standardizer: HeartRateStandardizer | None = None,
    *,
    window_size: int = 120,
    max_points: int = 480,
) -> tuple[BIDMCWindow, ...]:
    """Create non-overlapping CNP functions from each record's first 480 points.

    Each window has its own local time domain, with its first and last sample
    mapped to -1 and 1. A window is omitted if either time or HR contains a
    non-finite value. HR stays in BPM unless *standardizer* is supplied.
    """

    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if max_points < window_size:
        raise ValueError("max_points must be at least window_size")

    window_count = max_points // window_size
    windows: list[BIDMCWindow] = []
    for record in records:
        available = min(record.time.shape[0], record.heart_rate.shape[0], max_points)
        record_window_count = min(window_count, available // window_size)
        for window_index in range(record_window_count):
            start = window_index * window_size
            stop = start + window_size
            time = record.time[start:stop]
            heart_rate = record.heart_rate[start:stop]
            if not torch.isfinite(time).all() or not torch.isfinite(heart_rate).all():
                continue
            span = time[-1] - time[0]
            if span.numel() != 1 or not torch.isfinite(span).all() or span.item() <= 0:
                continue
            normalized_time = (2.0 * (time - time[0]) / span - 1.0).to(torch.float32)
            output_heart_rate = heart_rate
            if standardizer is not None:
                output_heart_rate = standardizer.normalize(output_heart_rate)
            windows.append(
                BIDMCWindow(
                    record_id=record.record_id,
                    patient_id=record.patient_id,
                    window_index=window_index,
                    time=normalized_time.contiguous(),
                    heart_rate=output_heart_rate.to(torch.float32).contiguous(),
                )
            )
    return tuple(windows)


__all__ = [
    "BIDMCRecord",
    "BIDMCSplit",
    "BIDMCWindow",
    "HeartRateStandardizer",
    "fit_heart_rate_standardizer",
    "load_bidmc_records",
    "make_bidmc_windows",
    "split_records_by_subject",
]
