"""Train and evaluate a CNP on fixed banks of synthetic Sin/Cos functions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor

from cnp import CNP, gaussian_nll
from cnp.synthetic import (
    SinusoidParameterRange,
    deterministic_regression_metrics,
    evaluate_sinusoids,
    gaussian_regression_metrics,
    interpolation_baselines,
    make_synthetic_evaluation_task,
    sample_sinusoid_parameters,
    sample_synthetic_training_batch,
    sinusoid_regression_baseline,
)


CONTEXT_COUNTS = (3, 5, 10, 20)
MAIN_REGIMES = ("interpolation", "gap_interpolation")
ALL_REGIMES = MAIN_REGIMES + ("extrapolation_stress",)
ID_RANGE = SinusoidParameterRange()
FREQUENCY_OOD_RANGE = SinusoidParameterRange(angular_frequency=(2.2, 3.0))


def default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parameter_fingerprint(parameters: Tensor) -> str:
    """Return a stable fingerprint without depending on a binary file format."""

    payload = json.dumps(parameters.tolist(), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_parameter_banks(args: argparse.Namespace) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Create disjoint train/validation/test parameter banks."""

    specifications = {
        "train": (args.train_functions, args.seed + 101, ID_RANGE, None, None),
        "validation": (
            args.validation_functions,
            args.seed + 211,
            ID_RANGE,
            None,
            None,
        ),
        "id_test": (args.test_functions, args.seed + 307, ID_RANGE, None, None),
        "frequency_ood_test": (
            args.test_functions,
            args.seed + 401,
            FREQUENCY_OOD_RANGE,
            None,
            None,
        ),
        "pure_sin_test": (
            args.anchor_functions,
            args.seed + 503,
            ID_RANGE,
            0.0,
            0.0,
        ),
        "pure_cos_test": (
            args.anchor_functions,
            args.seed + 601,
            ID_RANGE,
            math.pi / 2.0,
            0.0,
        ),
    }
    banks: dict[str, Tensor] = {}
    manifest: dict[str, Any] = {}
    for name, (
        count,
        seed,
        parameter_range,
        fixed_phase,
        fixed_offset,
    ) in specifications.items():
        bank = sample_sinusoid_parameters(
            count,
            parameter_range,
            generator=torch.Generator().manual_seed(seed),
        )
        if fixed_phase is not None:
            bank[:, 2] = fixed_phase
        if fixed_offset is not None:
            bank[:, 3] = fixed_offset
        banks[name] = bank
        manifest[name] = {
            "functions": count,
            "seed": seed,
            "parameter_range": parameter_range.as_dict(),
            "fixed_phase": fixed_phase,
            "fixed_offset": fixed_offset,
            "sha256": parameter_fingerprint(bank),
        }

    # Exact tuple overlap would leak a function across fixed banks. It is
    # overwhelmingly unlikely for random float32 tuples, but fail loudly.
    row_sets = {
        name: {tuple(float(value) for value in row) for row in bank.tolist()}
        for name, bank in banks.items()
    }
    names = tuple(banks)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = row_sets[left_name] & row_sets[right_name]
            if overlap:
                raise RuntimeError(
                    f"parameter banks {left_name} and {right_name} overlap by "
                    f"{len(overlap)} function(s)"
                )
    manifest["overlap_check"] = "all exact parameter tuples are disjoint"
    return banks, manifest


@torch.no_grad()
def evaluate_parameter_bank(
    model: CNP,
    parameters: Tensor,
    *,
    context_counts: Iterable[int],
    regimes: Iterable[str],
    grid_points: int,
    x_range: tuple[float, float],
    seed: int,
    device: torch.device,
    include_baselines: bool,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Evaluate one fixed function bank without advancing training RNG state."""

    was_training = model.training
    model.eval()
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for regime_index, regime in enumerate(regimes):
        regime_seed = seed + regime_index * 100_003
        by_context: dict[str, dict[str, Any]] = {}
        for context_count in context_counts:
            task = make_synthetic_evaluation_task(
                parameters,
                regime=regime,  # type: ignore[arg-type]
                context_count=context_count,
                grid_points=grid_points,
                x_range=x_range,
                seed=regime_seed,
            )
            mean, scale = model(
                task.context_x.to(device),
                task.context_y.to(device),
                task.target_x.to(device),
            )
            condition_results: dict[str, Any] = {
                "cnp": gaussian_regression_metrics(
                    mean.cpu(), scale.cpu(), task.target_y
                )
            }
            if include_baselines:
                for name, prediction in interpolation_baselines(
                    task.context_x, task.context_y, task.target_x
                ).items():
                    condition_results[name] = deterministic_regression_metrics(
                        prediction, task.target_y
                    )
                if context_count >= 5:
                    sinusoid_prediction = sinusoid_regression_baseline(
                        task.context_x, task.context_y, task.target_x
                    )
                    condition_results["sincos_regression"] = (
                        deterministic_regression_metrics(
                            sinusoid_prediction, task.target_y
                        )
                    )
            by_context[str(context_count)] = condition_results
        results[regime] = by_context
    if was_training:
        model.train()
    return results


def average_cnp_metric(
    evaluation: dict[str, dict[str, dict[str, Any]]],
    metric: str,
) -> float:
    values = [
        float(condition["cnp"][metric])
        for by_context in evaluation.values()
        for condition in by_context.values()
    ]
    if not values:
        raise ValueError("cannot average an empty evaluation")
    return sum(values) / len(values)


def _json_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def plot_learning_curve(history: list[dict[str, Any]], output: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    steps = [int(row["step"]) for row in history]
    train_nll = [float(row["train_nll_mean"]) for row in history]
    validation_nll = [float(row["validation_nll"]) for row in history]
    validation_rmse = [float(row["validation_rmse"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(steps, train_nll, label="Train NLL", color="tab:blue")
    axes[0].plot(steps, validation_nll, label="Validation NLL", color="tab:orange")
    axes[0].set_xlabel("Optimizer step")
    axes[0].set_ylabel("Gaussian NLL")
    axes[0].set_title("Learning objective")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(steps, validation_rmse, color="tab:green")
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("Function-macro RMSE")
    axes[1].set_title("ID validation interpolation")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return True


def plot_context_sweep(
    id_results: dict[str, dict[str, dict[str, Any]]],
    output: Path,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    methods = (
        ("cnp", "CNP", "tab:blue", "o"),
        ("linear", "Linear (incl. extrap.)", "tab:orange", "s"),
        ("nearest", "Nearest", "tab:green", "^"),
        ("context_mean", "Context mean", "tab:red", "D"),
        ("sincos_regression", "Sin/Cos fit", "tab:purple", "x"),
    )
    titles = {
        "interpolation": "All-domain interpolation",
        "gap_interpolation": "Central-gap interpolation",
        "extrapolation_stress": "Extrapolation stress test",
    }
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=False)
    for axis, regime in zip(axes, ALL_REGIMES):
        by_context = id_results[regime]
        counts = sorted(int(value) for value in by_context)
        for key, label, color, marker in methods:
            method_counts = [
                count for count in counts if key in by_context[str(count)]
            ]
            if not method_counts:
                continue
            values = [
                by_context[str(count)][key]["function_macro_rmse"]
                for count in method_counts
            ]
            axis.plot(
                method_counts,
                values,
                label=label,
                color=color,
                marker=marker,
            )
        axis.set_title(titles[regime])
        axis.set_xlabel("Context points")
        axis.set_xticks(CONTEXT_COUNTS)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Function-macro RMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return True


@torch.no_grad()
def plot_anchor_predictions(
    model: CNP,
    output: Path,
    *,
    grid_points: int,
    x_range: tuple[float, float],
    seed: int,
    device: torch.device,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    anchors = torch.tensor(
        [
            [1.25, 1.4, 0.0, 0.0],
            [1.25, 1.4, math.pi / 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = ("Pure Sin (phase 0)", "Pure Cos (phase pi/2)")
    query = torch.linspace(*x_range, 401).view(1, -1, 1).expand(2, -1, -1)
    truth = evaluate_sinusoids(anchors, query)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    was_training = model.training
    model.eval()
    for column, context_count in enumerate((3, 20)):
        task = make_synthetic_evaluation_task(
            anchors,
            regime="gap_interpolation",
            context_count=context_count,
            grid_points=grid_points,
            x_range=x_range,
            seed=seed,
        )
        mean, scale = model(
            task.context_x.to(device),
            task.context_y.to(device),
            query.to(device),
        )
        mean, scale = mean.cpu(), scale.cpu()
        for row in range(2):
            axis = axes[row, column]
            axis.axvspan(-0.6, 0.6, color="grey", alpha=0.10, label="Missing gap")
            axis.plot(query[row, :, 0], truth[row, :, 0], color="black", label="Truth")
            axis.plot(query[row, :, 0], mean[row, :, 0], color="tab:blue", label="CNP mean")
            axis.fill_between(
                query[row, :, 0],
                mean[row, :, 0] - 1.96 * scale[row, :, 0],
                mean[row, :, 0] + 1.96 * scale[row, :, 0],
                color="tab:blue",
                alpha=0.15,
                label="Pointwise 95% interval",
            )
            axis.scatter(
                task.context_x[row, :, 0],
                task.context_y[row, :, 0],
                color="tab:red",
                s=24,
                zorder=4,
                label="Context",
            )
            axis.set_title(f"{labels[row]}, Context={context_count}")
            axis.grid(alpha=0.2)
    if was_training:
        model.train()
    axes[1, 0].set_xlabel("x (normalized time)")
    axes[1, 1].set_xlabel("x (normalized time)")
    axes[0, 0].set_ylabel("y")
    axes[1, 0].set_ylabel("y")
    handles, labels_for_legend = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels_for_legend, loc="lower center", ncol=4)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return True


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=project_root / "outputs" / "sincos"
    )
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--min-steps", type=int, default=2_500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--targets", type=int, default=32)
    parser.add_argument("--min-context", type=int, default=3)
    parser.add_argument("--max-context", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--representation-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--train-functions", type=int, default=4_096)
    parser.add_argument("--validation-functions", type=int, default=256)
    parser.add_argument("--test-functions", type=int, default=256)
    parser.add_argument("--anchor-functions", type=int, default=128)
    parser.add_argument("--grid-points", type=int, default=129)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "steps",
        "min_steps",
        "batch_size",
        "targets",
        "min_context",
        "max_context",
        "hidden_dim",
        "representation_dim",
        "validation_interval",
        "patience",
        "train_functions",
        "validation_functions",
        "test_functions",
        "anchor_functions",
        "grid_points",
    )
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.min_steps > args.steps:
        raise ValueError("--min-steps cannot exceed --steps")
    if args.min_context > args.max_context:
        raise ValueError("--min-context cannot exceed --max-context")
    if args.max_context < max(CONTEXT_COUNTS):
        raise ValueError("--max-context must cover all evaluation Context counts")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.grid_points < 9 or args.grid_points % 2 == 0:
        raise ValueError("--grid-points must be an odd integer of at least 9")
    context_grid = torch.linspace(-2.0, 2.0, args.grid_points)
    candidate_counts = (
        args.grid_points,
        int((context_grid.abs() >= 0.6).sum().item()),
        int((context_grid.abs() <= 1.0).sum().item()),
    )
    if min(candidate_counts) < max(CONTEXT_COUNTS):
        raise ValueError(
            "--grid-points does not provide enough Context candidates for all regimes"
        )


def print_test_summary(test: dict[str, Any]) -> None:
    print("\nGap-interpolation function-macro RMSE")
    print(f"{'split':<20} {'C=3':>9} {'C=5':>9} {'C=10':>9} {'C=20':>9}")
    for split_name, split_results in test.items():
        values = split_results["gap_interpolation"]
        row = [values[str(count)]["cnp"]["function_macro_rmse"] for count in CONTEXT_COUNTS]
        print(f"{split_name:<20} " + " ".join(f"{value:9.4f}" for value in row))

    id_gap = test["id_test"]["gap_interpolation"]["20"]
    print("\nID gap interpolation at Context=20")
    for method in ("cnp", "linear", "nearest", "context_mean", "sincos_regression"):
        if method in id_gap:
            print(f"{method:<20} rmse={id_gap[method]['function_macro_rmse']:.5f}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    start_time = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = default_device() if args.device == "auto" else torch.device(args.device)
    x_range = (-2.0, 2.0)

    torch.manual_seed(args.seed)
    banks, bank_manifest = make_parameter_banks(args)
    model_config = {
        "x_dim": 1,
        "y_dim": 1,
        "hidden_dim": args.hidden_dim,
        "representation_dim": args.representation_dim,
    }
    model = CNP(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_generator = torch.Generator().manual_seed(args.seed + 1_001)
    evaluation_seeds = {
        "validation": args.seed + 10_000,
        "test": args.seed + 20_000,
        "anchor_plot": args.seed + 30_000,
    }
    usage_counts = torch.zeros(args.train_functions, dtype=torch.int64)
    history: list[dict[str, Any]] = []
    checkpoint_path = args.output_dir / "best_model.pt"
    best_score = float("inf")
    best_step = 0
    checkpoint_written = False
    validations_without_improvement = 0
    interval_loss_sum = 0.0
    interval_gradient_sum = 0.0
    interval_steps = 0
    context_observations = 0
    target_loss_points = 0
    final_step = 0

    print(
        f"device={device} train/validation/test functions="
        f"{args.train_functions}/{args.validation_functions}/{args.test_functions}"
    )
    print(
        "family=y=offset+amplitude*sin(angular_frequency*x+phase), "
        "cos=phase pi/2"
    )

    for step in range(1, args.steps + 1):
        final_step = step
        batch = sample_synthetic_training_batch(
            banks["train"],
            batch_size=args.batch_size,
            min_context=args.min_context,
            max_context=args.max_context,
            target_count=args.targets,
            x_range=x_range,
            generator=train_generator,
            device=device,
        )
        mean, scale = model(
            batch.context_x,
            batch.context_y,
            batch.target_x,
            batch.context_mask,
        )
        loss = gaussian_nll(mean, scale, batch.target_y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at step {step}")
        optimizer.zero_grad()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        optimizer.step()

        usage_counts += torch.bincount(
            batch.function_indices, minlength=args.train_functions
        )
        context_observations += int(batch.context_counts.sum().item())
        target_loss_points += args.batch_size * args.targets
        interval_loss_sum += float(loss.item())
        interval_gradient_sum += float(gradient_norm)
        interval_steps += 1

        if step == 1 or step % args.validation_interval == 0 or step == args.steps:
            validation = evaluate_parameter_bank(
                model,
                banks["validation"],
                context_counts=CONTEXT_COUNTS,
                regimes=MAIN_REGIMES,
                grid_points=args.grid_points,
                x_range=x_range,
                seed=evaluation_seeds["validation"],
                device=device,
                include_baselines=False,
            )
            validation_nll = average_cnp_metric(validation, "function_macro_nll")
            validation_rmse = average_cnp_metric(validation, "function_macro_rmse")
            if not math.isfinite(validation_nll) or not math.isfinite(validation_rmse):
                raise FloatingPointError(f"non-finite validation metric at step {step}")
            row = {
                "step": step,
                "train_nll_mean": interval_loss_sum / interval_steps,
                "gradient_norm_mean_before_clip": interval_gradient_sum / interval_steps,
                "validation_nll": validation_nll,
                "validation_rmse": validation_rmse,
            }
            history.append(row)
            print(
                f"step={step:05d} train_nll={row['train_nll_mean']:.4f} "
                f"val_nll={validation_nll:.4f} val_rmse={validation_rmse:.4f}"
            )
            interval_loss_sum = 0.0
            interval_gradient_sum = 0.0
            interval_steps = 0

            if validation_nll < best_score:
                best_score = validation_nll
                best_step = step
                validations_without_improvement = 0
                torch.save(
                    {
                        "format_version": 1,
                        "benchmark": "sincos",
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "model_config": model_config,
                        "training_config": _json_args(args),
                        "x_range": list(x_range),
                        "parameter_banks": bank_manifest,
                        "evaluation_seeds": evaluation_seeds,
                        "selection_metric": "mean ID validation NLL over interpolation regimes",
                        "best_validation_score": best_score,
                        "validation_rmse_at_best": validation_rmse,
                        "validation_at_best": validation,
                        "train_generator_state": train_generator.get_state(),
                        "train_function_usage_counts": usage_counts.clone(),
                        "context_observations_at_best": context_observations,
                        "target_loss_points_at_best": target_loss_points,
                    },
                    checkpoint_path,
                )
                checkpoint_written = True
            else:
                validations_without_improvement += 1

            if (
                step >= args.min_steps
                and validations_without_improvement >= args.patience
            ):
                print(f"early_stop step={step} patience={args.patience}")
                break

    if not checkpoint_written:
        raise RuntimeError("training completed without writing a finite checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test = {
        name: evaluate_parameter_bank(
            model,
            bank,
            context_counts=CONTEXT_COUNTS,
            regimes=ALL_REGIMES,
            grid_points=args.grid_points,
            x_range=x_range,
            seed=evaluation_seeds["test"],
            device=device,
            include_baselines=True,
        )
        for name, bank in banks.items()
        if name not in ("train", "validation")
    }

    duration = time.perf_counter() - start_time
    best_usage = checkpoint["train_function_usage_counts"]
    results = {
        "benchmark": {
            "name": "Sin/Cos synthetic function regression",
            "formula": "y = offset + amplitude * sin(angular_frequency * x + phase)",
            "cosine_identity": "cos(theta) = sin(theta + pi/2)",
            "x_range": list(x_range),
            "observation_noise_std": 0.0,
            "context_counts": list(CONTEXT_COUNTS),
            "regimes": {
                "interpolation": "domain endpoints are Context; all Query points lie inside their hull",
                "gap_interpolation": "Context lies outside [-0.6, 0.6]; Query is the central missing gap",
                "extrapolation_stress": "Context lies in [-1, 1]; Query lies outside it; not used for checkpoint selection",
            },
        },
        "config": _json_args(args),
        "model_config": model_config,
        "parameter_banks": bank_manifest,
        "evaluation_seeds": evaluation_seeds,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": str(device),
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
        },
        "training": {
            "final_step": final_step,
            "best_step": int(checkpoint["step"]),
            "selection_metric": checkpoint["selection_metric"],
            "best_validation_nll": float(checkpoint["best_validation_score"]),
            "validation_rmse_at_best": float(checkpoint["validation_rmse_at_best"]),
            "function_episodes_final": final_step * args.batch_size,
            "context_observations_final": context_observations,
            "target_loss_points_final": target_loss_points,
            "unique_train_functions_seen_at_best": int((best_usage > 0).sum().item()),
            "function_episodes_at_best": int(best_usage.sum().item()),
            "context_observations_at_best": int(
                checkpoint["context_observations_at_best"]
            ),
            "target_loss_points_at_best": int(checkpoint["target_loss_points_at_best"]),
            "history": history,
            "wall_time_seconds": duration,
        },
        "validation_at_best": checkpoint["validation_at_best"],
        "test": test,
        "interpretation_notes": [
            "All reported Gaussian intervals are pointwise, not simultaneous function bands.",
            "The extrapolation stress test also changes Context layout and is not a heart-rate future model.",
            "The family-aware Sin/Cos regression baseline is omitted for Context=3 because frequency is underidentified.",
            "Test banks are evaluated only after selecting and reloading the best validation checkpoint.",
        ],
    }
    results_path = args.output_dir / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print_test_summary(test)
    print(f"checkpoint={checkpoint_path}")
    print(f"results={results_path}")
    print(f"best_step={best_step} duration_seconds={duration:.2f}")

    if not args.no_plot:
        plots = (
            ("learning_curve.png", plot_learning_curve(history, args.output_dir / "learning_curve.png")),
            ("context_sweep.png", plot_context_sweep(test["id_test"], args.output_dir / "context_sweep.png")),
            (
                "anchor_predictions.png",
                plot_anchor_predictions(
                    model,
                    args.output_dir / "anchor_predictions.png",
                    grid_points=args.grid_points,
                    x_range=x_range,
                    seed=evaluation_seeds["anchor_plot"],
                    device=device,
                ),
            ),
        )
        for name, created in plots:
            if created:
                print(f"plot={args.output_dir / name}")


if __name__ == "__main__":
    main()
