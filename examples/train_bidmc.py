"""Train and evaluate CNP heart-rate interpolation on BIDMC numerics."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor

from cnp import CNP, gaussian_nll
from cnp.datasets import (
    BIDMCSplit,
    BIDMCWindow,
    HeartRateStandardizer,
    fit_heart_rate_standardizer,
    load_bidmc_records,
    make_bidmc_windows,
    split_records_by_subject,
)


@dataclass(frozen=True)
class MaskRegime:
    name: str
    kind: str
    amount: float


RANDOM_REGIMES = (
    MaskRegime("random_10pct", "random", 0.10),
    MaskRegime("random_30pct", "random", 0.30),
    MaskRegime("random_50pct", "random", 0.50),
)
BLOCK_REGIMES = (
    MaskRegime("gap_5s", "block", 5),
    MaskRegime("gap_15s", "block", 15),
    MaskRegime("gap_30s", "block", 30),
)
ALL_REGIMES = RANDOM_REGIMES + BLOCK_REGIMES


def default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mask_indices(
    point_count: int,
    regime: MaskRegime,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Return sorted context/target indices for a pure interpolation task."""

    if point_count < 3:
        raise ValueError("an interpolation window needs at least three points")
    if regime.kind == "random":
        target_count = max(1, round(point_count * regime.amount))
        candidates = torch.arange(1, point_count - 1)
        if target_count > candidates.numel():
            raise ValueError("random target count leaves no endpoint context")
        order = torch.randperm(candidates.numel(), generator=generator)
        target_indices = candidates[order[:target_count]].sort().values
    elif regime.kind == "block":
        target_count = int(regime.amount)
        margin = 5
        first_start = margin
        last_start = point_count - margin - target_count
        if last_start < first_start:
            raise ValueError("block and margins do not fit inside the window")
        start = int(
            torch.randint(
                first_start,
                last_start + 1,
                size=(),
                generator=generator,
            ).item()
        )
        target_indices = torch.arange(start, start + target_count)
    else:
        raise ValueError(f"unknown mask kind: {regime.kind!r}")

    is_context = torch.ones(point_count, dtype=torch.bool)
    is_context[target_indices] = False
    context_indices = torch.nonzero(is_context, as_tuple=False).flatten()
    return context_indices, target_indices


def group_windows_by_subject(
    windows: Iterable[BIDMCWindow],
) -> dict[str, tuple[BIDMCWindow, ...]]:
    grouped: dict[str, list[BIDMCWindow]] = defaultdict(list)
    for window in windows:
        grouped[window.patient_id].append(window)
    return {
        patient_id: tuple(
            sorted(items, key=lambda item: (item.record_id, item.window_index))
        )
        for patient_id, items in sorted(grouped.items())
    }


def choose_training_regime(rng: random.Random) -> MaskRegime:
    if rng.random() < 0.70:
        return rng.choice(RANDOM_REGIMES)
    return rng.choice(BLOCK_REGIMES)


def sample_training_batch(
    windows_by_subject: dict[str, tuple[BIDMCWindow, ...]],
    regime: MaskRegime,
    batch_size: int,
    python_rng: random.Random,
    torch_generator: torch.Generator,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sample subjects uniformly, then one window within each subject."""

    patient_ids = tuple(windows_by_subject)
    if not patient_ids:
        raise ValueError("no training windows were provided")

    context_x: list[Tensor] = []
    context_y: list[Tensor] = []
    target_x: list[Tensor] = []
    target_y: list[Tensor] = []
    for _ in range(batch_size):
        patient_id = python_rng.choice(patient_ids)
        window = python_rng.choice(windows_by_subject[patient_id])
        context_indices, target_indices = mask_indices(
            window.time.shape[0], regime, torch_generator
        )
        context_x.append(window.time[context_indices])
        context_y.append(window.heart_rate[context_indices])
        target_x.append(window.time[target_indices])
        target_y.append(window.heart_rate[target_indices])

    return (
        torch.stack(context_x).to(device),
        torch.stack(context_y).to(device),
        torch.stack(target_x).to(device),
        torch.stack(target_y).to(device),
    )


def interpolation_baselines(
    context_x: Tensor,
    context_y: Tensor,
    target_x: Tensor,
) -> dict[str, Tensor]:
    """Return normalized-HR baseline predictions at target coordinates."""

    x = context_x.flatten().cpu()
    y = context_y.flatten().cpu()
    query = target_x.flatten().cpu()
    order = torch.argsort(x)
    x, y = x[order], y[order]

    right = torch.searchsorted(x, query).clamp(1, x.numel() - 1)
    left = right - 1
    choose_right = (x[right] - query).abs() < (query - x[left]).abs()
    nearest = torch.where(choose_right, y[right], y[left])

    denominator = (x[right] - x[left]).clamp_min(torch.finfo(x.dtype).eps)
    weight = (query - x[left]) / denominator
    linear = y[left] + weight * (y[right] - y[left])

    return {
        "train_mean": torch.zeros_like(query),
        "context_mean": torch.full_like(query, float(y.mean().item())),
        "nearest": nearest,
        "linear": linear,
    }


def _empty_accumulator() -> dict[str, float]:
    return {
        "absolute_error": 0.0,
        "squared_error": 0.0,
        "count": 0.0,
        "nll": 0.0,
        "covered_68": 0.0,
        "covered_95": 0.0,
        "interval_width_95": 0.0,
    }


def _update_accumulator(
    accumulator: dict[str, float],
    prediction_bpm: Tensor,
    target_bpm: Tensor,
    *,
    scale_bpm: Tensor | None = None,
    nll_standardized: Tensor | None = None,
) -> None:
    error = prediction_bpm - target_bpm
    accumulator["absolute_error"] += float(error.abs().sum().item())
    accumulator["squared_error"] += float(error.square().sum().item())
    accumulator["count"] += float(error.numel())
    if scale_bpm is not None and nll_standardized is not None:
        accumulator["nll"] += float(nll_standardized.sum().item())
        accumulator["covered_68"] += float((error.abs() <= scale_bpm).sum().item())
        accumulator["covered_95"] += float(
            (error.abs() <= 1.96 * scale_bpm).sum().item()
        )
        accumulator["interval_width_95"] += float((3.92 * scale_bpm).sum().item())


def _summarize_subjects(
    by_subject: dict[str, dict[str, float]],
    *,
    probabilistic: bool,
) -> dict[str, float]:
    if not by_subject:
        raise ValueError("cannot summarize empty evaluation results")
    total_abs = total_sq = total_count = 0.0
    subject_mae: list[float] = []
    subject_rmse: list[float] = []
    subject_nll: list[float] = []
    subject_coverage_68: list[float] = []
    subject_coverage_95: list[float] = []
    subject_width_95: list[float] = []

    for values in by_subject.values():
        count = values["count"]
        total_abs += values["absolute_error"]
        total_sq += values["squared_error"]
        total_count += count
        subject_mae.append(values["absolute_error"] / count)
        subject_rmse.append(math.sqrt(values["squared_error"] / count))
        if probabilistic:
            subject_nll.append(values["nll"] / count)
            subject_coverage_68.append(values["covered_68"] / count)
            subject_coverage_95.append(values["covered_95"] / count)
            subject_width_95.append(values["interval_width_95"] / count)

    summary = {
        "subject_macro_mae_bpm": sum(subject_mae) / len(subject_mae),
        "subject_macro_rmse_bpm": sum(subject_rmse) / len(subject_rmse),
        "point_micro_mae_bpm": total_abs / total_count,
        "point_micro_rmse_bpm": math.sqrt(total_sq / total_count),
        "points": int(total_count),
        "subjects": len(by_subject),
    }
    if probabilistic:
        summary.update(
            {
                "subject_macro_nll_standardized": sum(subject_nll) / len(subject_nll),
                "subject_macro_coverage_68": sum(subject_coverage_68)
                / len(subject_coverage_68),
                "subject_macro_coverage_95": sum(subject_coverage_95)
                / len(subject_coverage_95),
                "subject_macro_interval_width_95_bpm": sum(subject_width_95)
                / len(subject_width_95),
            }
        )
    return summary


@torch.no_grad()
def evaluate_regime(
    model: CNP,
    windows: tuple[BIDMCWindow, ...],
    standardizer: HeartRateStandardizer,
    regime: MaskRegime,
    *,
    repeats: int,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Evaluate CNP and simple interpolation baselines for one mask regime."""

    model.eval()
    generator = torch.Generator().manual_seed(seed)
    methods = ("cnp", "train_mean", "context_mean", "nearest", "linear")
    accumulators: dict[str, dict[str, dict[str, float]]] = {
        method: defaultdict(_empty_accumulator) for method in methods
    }

    for _ in range(repeats):
        for window in windows:
            context_indices, target_indices = mask_indices(
                window.time.shape[0], regime, generator
            )
            context_x = window.time[context_indices]
            context_y = window.heart_rate[context_indices]
            target_x = window.time[target_indices]
            target_y = window.heart_rate[target_indices]

            mean, scale = model(
                context_x.to(device),
                context_y.to(device),
                target_x.to(device),
            )
            mean, scale = mean.cpu(), scale.cpu()
            nll = gaussian_nll(mean, scale, target_y, reduction="none")
            prediction_bpm = standardizer.denormalize(mean).flatten()
            target_bpm = standardizer.denormalize(target_y).flatten()
            scale_bpm = standardizer.scale_to_bpm(scale).flatten()
            _update_accumulator(
                accumulators["cnp"][window.patient_id],
                prediction_bpm,
                target_bpm,
                scale_bpm=scale_bpm,
                nll_standardized=nll.flatten(),
            )

            for name, baseline in interpolation_baselines(
                context_x, context_y, target_x
            ).items():
                _update_accumulator(
                    accumulators[name][window.patient_id],
                    standardizer.denormalize(baseline),
                    target_bpm,
                )

    return {
        method: _summarize_subjects(
            dict(accumulators[method]), probabilistic=method == "cnp"
        )
        for method in methods
    }


def evaluate_all_regimes(
    model: CNP,
    windows: tuple[BIDMCWindow, ...],
    standardizer: HeartRateStandardizer,
    *,
    repeats: int,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, dict[str, float]]]:
    return {
        regime.name: evaluate_regime(
            model,
            windows,
            standardizer,
            regime,
            repeats=repeats,
            seed=seed + index * 100_003,
            device=device,
        )
        for index, regime in enumerate(ALL_REGIMES)
    }


def mixed_validation_nll(
    metrics: dict[str, dict[str, dict[str, float]]],
) -> float:
    return sum(
        regime_metrics["cnp"]["subject_macro_nll_standardized"]
        for regime_metrics in metrics.values()
    ) / len(metrics)


def split_manifest(split: BIDMCSplit) -> dict[str, Any]:
    def describe(records: Iterable[Any]) -> dict[str, list[str]]:
        records = tuple(records)
        return {
            "patient_ids": sorted({record.patient_id for record in records}),
            "record_ids": sorted(record.record_id for record in records),
        }

    return {
        "train": describe(split.train),
        "validation": describe(split.validation),
        "test": describe(split.test),
    }


def plot_example(
    model: CNP,
    window: BIDMCWindow,
    standardizer: HeartRateStandardizer,
    output: Path,
    device: torch.device,
    seed: int,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    regime = BLOCK_REGIMES[-1]
    generator = torch.Generator().manual_seed(seed)
    context_indices, target_indices = mask_indices(
        window.time.shape[0], regime, generator
    )
    context_x = window.time[context_indices]
    context_y = window.heart_rate[context_indices]
    with torch.no_grad():
        mean, scale = model(
            context_x.to(device),
            context_y.to(device),
            window.time.to(device),
        )
    mean, scale = mean.cpu(), scale.cpu()
    time_seconds = torch.linspace(0, window.time.shape[0] - 1, window.time.shape[0])
    truth = standardizer.denormalize(window.heart_rate).flatten()
    prediction = standardizer.denormalize(mean).flatten()
    uncertainty = standardizer.scale_to_bpm(scale).flatten()
    linear = standardizer.denormalize(
        interpolation_baselines(context_x, context_y, window.time)["linear"]
    )

    plt.figure(figsize=(10, 5))
    plt.plot(time_seconds, truth, color="black", linewidth=1.5, label="Observed truth")
    plt.plot(time_seconds, prediction, color="tab:blue", label="CNP mean")
    plt.fill_between(
        time_seconds,
        prediction - 1.96 * uncertainty,
        prediction + 1.96 * uncertainty,
        color="tab:blue",
        alpha=0.15,
        label="CNP 95% interval",
    )
    plt.plot(time_seconds, linear, color="tab:orange", linestyle="--", label="Linear")
    plt.scatter(
        time_seconds[target_indices],
        truth[target_indices],
        color="tab:red",
        s=18,
        label="Artificially missing",
        zorder=3,
    )
    plt.xlabel("Seconds within 120-second window")
    plt.ylabel("Heart rate (bpm)")
    plt.title(
        f"BIDMC {window.record_id}, held-out patient {window.patient_id}, 30 s gap"
    )
    plt.legend(ncol=2)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)
    plt.close()
    return True


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_root / "data" / "bidmc")
    parser.add_argument(
        "--output-dir", type=Path, default=project_root / "outputs" / "bidmc"
    )
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--min-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--representation-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--selection-metric",
        choices=("mae", "nll"),
        default="mae",
        help="validation metric used for checkpoint selection and early stopping",
    )
    parser.add_argument("--test-repeats", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "steps",
        "min_steps",
        "batch_size",
        "hidden_dim",
        "representation_dim",
        "validation_interval",
        "patience",
        "test_repeats",
        "window_size",
    )
    for name in positive_names:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.min_steps > args.steps:
        raise ValueError("--min-steps cannot exceed --steps")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.window_size <= max(int(regime.amount) for regime in BLOCK_REGIMES) + 10:
        raise ValueError("--window-size is too short for the largest block mask")


def _json_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch.manual_seed(args.seed)
    device = default_device() if args.device == "auto" else torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_bidmc_records(args.data_dir)
    split = split_records_by_subject(records, seed=args.seed)
    standardizer = fit_heart_rate_standardizer(split.train)
    train_windows = make_bidmc_windows(
        split.train, standardizer, window_size=args.window_size
    )
    validation_windows = make_bidmc_windows(
        split.validation, standardizer, window_size=args.window_size
    )
    test_windows = make_bidmc_windows(
        split.test, standardizer, window_size=args.window_size
    )
    train_by_subject = group_windows_by_subject(train_windows)

    print(
        "dataset="
        f"{len(records)} records/{len({record.patient_id for record in records})} patients "
        f"windows(train/val/test)={len(train_windows)}/{len(validation_windows)}/{len(test_windows)}"
    )
    print(f"train_hr_mean={standardizer.mean:.6f} train_hr_std={standardizer.std:.6f}")
    print(f"device={device}")

    model_config = {
        "x_dim": 1,
        "y_dim": 1,
        "hidden_dim": args.hidden_dim,
        "representation_dim": args.representation_dim,
    }
    model = CNP(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    python_rng = random.Random(args.seed + 1)
    torch_generator = torch.Generator().manual_seed(args.seed + 2)
    checkpoint_path = args.output_dir / "best_model.pt"
    history: list[dict[str, float | int | str]] = []
    best_validation_score = float("inf")
    validations_without_improvement = 0
    final_step = 0

    for step in range(1, args.steps + 1):
        final_step = step
        regime = choose_training_regime(python_rng)
        context_x, context_y, target_x, target_y = sample_training_batch(
            train_by_subject,
            regime,
            args.batch_size,
            python_rng,
            torch_generator,
            device,
        )
        mean, scale = model(context_x, context_y, target_x)
        loss = gaussian_nll(mean, scale, target_y)
        optimizer.zero_grad()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % args.validation_interval == 0 or step == args.steps:
            validation = evaluate_all_regimes(
                model,
                validation_windows,
                standardizer,
                repeats=1,
                seed=args.seed + 10_000,
                device=device,
            )
            validation_nll = mixed_validation_nll(validation)
            validation_mae = sum(
                values["cnp"]["subject_macro_mae_bpm"]
                for values in validation.values()
            ) / len(validation)
            validation_score = (
                validation_mae
                if args.selection_metric == "mae"
                else validation_nll
            )
            row: dict[str, float | int | str] = {
                "step": step,
                "train_nll": float(loss.item()),
                "gradient_norm": float(gradient_norm),
                "mask_regime": regime.name,
                "validation_nll": validation_nll,
                "validation_mae_bpm": validation_mae,
            }
            history.append(row)
            print(
                f"step={step:04d} train_nll={loss.item():.4f} "
                f"val_nll={validation_nll:.4f} val_mae={validation_mae:.3f} bpm "
                f"mask={regime.name}"
            )

            if validation_score < best_validation_score:
                best_validation_score = validation_score
                validations_without_improvement = 0
                torch.save(
                    {
                        "format_version": 1,
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "model_config": model_config,
                        "training_config": _json_args(args),
                        "normalization": {
                            "mean_bpm": standardizer.mean,
                            "std_bpm_population": standardizer.std,
                        },
                        "split": split_manifest(split),
                        "selection_metric": args.selection_metric,
                        "best_validation_score": best_validation_score,
                        "validation_nll": validation_nll,
                        "validation_mae_bpm": validation_mae,
                        "python_rng_state": python_rng.getstate(),
                        "torch_generator_state": torch_generator.get_state(),
                    },
                    checkpoint_path,
                )
            else:
                validations_without_improvement += 1

            if (
                step >= args.min_steps
                and validations_without_improvement >= args.patience
            ):
                print(f"early_stop step={step} patience={args.patience}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_all_regimes(
        model,
        test_windows,
        standardizer,
        repeats=args.test_repeats,
        seed=args.seed + 20_000,
        device=device,
    )

    results = {
        "dataset": {
            "name": "BIDMC PPG and Respiration Dataset",
            "version": "1.0.0",
            "records": len(records),
            "patients": len({record.patient_id for record in records}),
            "windows": {
                "train": len(train_windows),
                "validation": len(validation_windows),
                "test": len(test_windows),
            },
            "window_time_normalization": "each window mapped to [-1, 1]",
            "natural_missing_hr_policy": "drop any 120-point window containing NaN HR",
        },
        "config": _json_args(args),
        "model_config": model_config,
        "normalization": {
            "mean_bpm": standardizer.mean,
            "std_bpm_population": standardizer.std,
            "fit_scope": "training patients, first 480 points per record",
        },
        "split": split_manifest(split),
        "training": {
            "final_step": final_step,
            "best_step": int(checkpoint["step"]),
            "selection_metric": checkpoint["selection_metric"],
            "best_validation_score": float(checkpoint["best_validation_score"]),
            "validation_nll_at_best": float(checkpoint["validation_nll"]),
            "validation_mae_bpm_at_best": float(checkpoint["validation_mae_bpm"]),
            "history": history,
        },
        "test": test_metrics,
    }
    results_path = args.output_dir / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\nTest subject-macro MAE (bpm)")
    print(f"{'mask':<16} {'CNP':>8} {'linear':>8} {'nearest':>8} {'ctx-mean':>10}")
    for regime in ALL_REGIMES:
        values = test_metrics[regime.name]
        print(
            f"{regime.name:<16} "
            f"{values['cnp']['subject_macro_mae_bpm']:8.3f} "
            f"{values['linear']['subject_macro_mae_bpm']:8.3f} "
            f"{values['nearest']['subject_macro_mae_bpm']:8.3f} "
            f"{values['context_mean']['subject_macro_mae_bpm']:10.3f}"
        )
    print(f"checkpoint={checkpoint_path}")
    print(f"results={results_path}")

    if not args.no_plot:
        plot_path = args.output_dir / "example_gap30.png"
        if plot_example(
            model,
            test_windows[0],
            standardizer,
            plot_path,
            device,
            seed=args.seed + 30_000,
        ):
            print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
