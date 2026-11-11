"""Train and evaluate a CNP on functions sampled from an RBF Gaussian Process."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor

from cnp import CNP, gaussian_nll
from cnp.rbf_gp import (
    RBFKernelConfig,
    gp_posterior,
    make_gp_cholesky,
    make_gp_grid,
    make_rbf_evaluation_task,
    sample_gp_functions,
    sample_rbf_training_batch,
)
from cnp.synthetic import (
    deterministic_regression_metrics,
    gaussian_regression_metrics,
)


CONTEXT_COUNTS = (3, 5, 10, 20)
GATE_IMPROVEMENT_C3_TO_C20 = 0.20
GATE_C10_GAIN_OVER_CONTEXT_MEAN = 0.25


def default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _json_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _state_dict_on_cpu(model: CNP) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def tensor_fingerprint(value: Tensor) -> str:
    """Return a reproducible SHA-256 for a fixed CPU tensor bank."""

    payload = json.dumps(
        value.detach().cpu().tolist(), separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@torch.no_grad()
def evaluate_model(
    model: CNP,
    function_values: Tensor,
    grid: Tensor,
    kernel: RBFKernelConfig,
    *,
    context_counts: Iterable[int],
    seed: int,
    device: torch.device,
    include_references: bool,
) -> dict[str, dict[str, Any]]:
    """Evaluate held-out GP draws on unseen Query points.

    The same seed is used for every Context count so that the Context sets are
    nested. Context positions are excluded from each condition's Query set.
    """

    was_training = model.training
    model.eval()
    results: dict[str, dict[str, Any]] = {}
    for context_count in context_counts:
        task = make_rbf_evaluation_task(
            function_values,
            grid,
            context_count=context_count,
            seed=seed,
        )
        mean, scale = model(
            task.context_x.to(device),
            task.context_y.to(device),
            task.target_x.to(device),
        )
        condition: dict[str, Any] = {
            "cnp": gaussian_regression_metrics(
                mean.cpu(), scale.cpu(), task.target_y
            )
        }
        if include_references:
            gp_mean, gp_scale = gp_posterior(
                task.context_x,
                task.context_y,
                task.target_x,
                kernel,
            )
            context_mean = task.context_y.mean(dim=1, keepdim=True).expand_as(
                task.target_y
            )
            condition["gp_posterior_oracle"] = gaussian_regression_metrics(
                gp_mean, gp_scale, task.target_y
            )
            condition["context_mean"] = deterministic_regression_metrics(
                context_mean, task.target_y
            )
        results[str(context_count)] = condition
    if was_training:
        model.train()
    return results


def average_cnp_metric(evaluation: dict[str, dict[str, Any]], metric: str) -> float:
    values = [float(condition["cnp"][metric]) for condition in evaluation.values()]
    if not values:
        raise ValueError("cannot average an empty evaluation")
    return sum(values) / len(values)


def final_test_gate(evaluation: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply thresholds declared independently of the held-out test values."""

    c3 = float(evaluation["3"]["cnp"]["function_macro_rmse"])
    c10 = float(evaluation["10"]["cnp"]["function_macro_rmse"])
    c20 = float(evaluation["20"]["cnp"]["function_macro_rmse"])
    mean_c10 = float(
        evaluation["10"]["context_mean"]["function_macro_rmse"]
    )
    checks = {
        "c20_rmse_improves_at_least_20_percent_over_c3": (
            c20 <= (1.0 - GATE_IMPROVEMENT_C3_TO_C20) * c3
        ),
        "c10_rmse_beats_context_mean_by_at_least_25_percent": (
            c10
            <= (1.0 - GATE_C10_GAIN_OVER_CONTEXT_MEAN) * mean_c10
        ),
    }
    return {
        "criteria_declared_before_final_evaluation": True,
        "thresholds": {
            "minimum_c3_to_c20_relative_rmse_improvement": (
                GATE_IMPROVEMENT_C3_TO_C20
            ),
            "minimum_c10_relative_rmse_gain_over_context_mean": (
                GATE_C10_GAIN_OVER_CONTEXT_MEAN
            ),
        },
        "observed": {
            "c3_cnp_rmse": c3,
            "c10_cnp_rmse": c10,
            "c20_cnp_rmse": c20,
            "c10_context_mean_rmse": mean_c10,
            "c3_to_c20_relative_improvement": 1.0 - c20 / c3,
            "c10_relative_gain_over_context_mean": 1.0 - c10 / mean_c10,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "note": (
            "The analytic GP posterior is an oracle that knows the generating "
            "kernel; it is reported as a reference and is not used as a pass gate."
        ),
    }


def plot_learning_curve(history: list[dict[str, Any]], output: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    steps = [int(row["step"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    axes[0].plot(
        steps,
        [float(row["train_nll_mean"]) for row in history],
        color="tab:blue",
        label="Train NLL",
    )
    axes[0].plot(
        steps,
        [float(row["validation_nll"]) for row in history],
        color="tab:orange",
        label="Validation NLL",
    )
    axes[0].set(
        title="Learning objective",
        xlabel="Optimizer step",
        ylabel="Gaussian NLL",
    )
    axes[0].legend()
    axes[1].plot(
        steps,
        [float(row["validation_rmse"]) for row in history],
        color="tab:green",
    )
    axes[1].set(
        title="Held-out GP validation",
        xlabel="Optimizer step",
        ylabel="Mean function-macro RMSE",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return True


def plot_context_sweep(
    evaluation: dict[str, dict[str, Any]], output: Path
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    methods = (
        ("cnp", "CNP", "tab:blue", "o"),
        ("gp_posterior_oracle", "Analytic GP posterior", "tab:orange", "s"),
        ("context_mean", "Context mean", "tab:grey", "D"),
    )
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    for key, label, color, marker in methods:
        values = [
            float(evaluation[str(count)][key]["function_macro_rmse"])
            for count in CONTEXT_COUNTS
        ]
        axis.plot(
            CONTEXT_COUNTS,
            values,
            color=color,
            marker=marker,
            label=label,
        )
    axis.set(
        title="RBF-GP held-out function reconstruction",
        xlabel="Context points",
        ylabel="Unseen-Query function-macro RMSE",
        xticks=CONTEXT_COUNTS,
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return True


@torch.no_grad()
def plot_predictions(
    model: CNP,
    function_values: Tensor,
    grid: Tensor,
    kernel: RBFKernelConfig,
    output: Path,
    *,
    context_count: int,
    seed: int,
    device: torch.device,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    sample_count = min(4, function_values.shape[0])
    selected = function_values[:sample_count]
    task = make_rbf_evaluation_task(
        selected,
        grid,
        context_count=context_count,
        seed=seed,
    )
    dense_x = grid.view(1, -1, 1).expand(sample_count, -1, -1)
    was_training = model.training
    model.eval()
    cnp_mean, cnp_scale = model(
        task.context_x.to(device),
        task.context_y.to(device),
        dense_x.to(device),
    )
    cnp_mean, cnp_scale = cnp_mean.cpu(), cnp_scale.cpu()
    gp_mean, gp_scale = gp_posterior(
        task.context_x,
        task.context_y,
        dense_x,
        kernel,
    )

    figure, axes = plt.subplots(2, 2, figsize=(11.6, 7.6), sharex=True)
    flat_axes = axes.ravel()
    x = grid[:, 0] if grid.ndim == 2 else grid
    for index, axis in enumerate(flat_axes):
        if index >= sample_count:
            axis.axis("off")
            continue
        axis.plot(x, selected[index, :, 0], color="black", label="Sampled truth")
        axis.plot(x, cnp_mean[index, :, 0], color="tab:blue", label="CNP mean")
        axis.fill_between(
            x,
            cnp_mean[index, :, 0] - 1.96 * cnp_scale[index, :, 0],
            cnp_mean[index, :, 0] + 1.96 * cnp_scale[index, :, 0],
            color="tab:blue",
            alpha=0.13,
            label="CNP 95% interval",
        )
        axis.plot(
            x,
            gp_mean[index, :, 0],
            color="tab:orange",
            linestyle="--",
            label="GP posterior mean",
        )
        axis.fill_between(
            x,
            gp_mean[index, :, 0] - 1.96 * gp_scale[index, :, 0],
            gp_mean[index, :, 0] + 1.96 * gp_scale[index, :, 0],
            color="tab:orange",
            alpha=0.09,
            label="GP posterior 95% interval",
        )
        axis.scatter(
            task.context_x[index, :, 0],
            task.context_y[index, :, 0],
            color="tab:red",
            edgecolor="white",
            linewidth=0.5,
            s=28,
            zorder=4,
            label="Context",
        )
        axis.set_title(f"Held-out GP draw {index + 1}, Context={context_count}")
        axis.grid(alpha=0.2)
    if was_training:
        model.train()
    axes[1, 0].set_xlabel("x")
    axes[1, 1].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[1, 0].set_ylabel("y")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3)
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return True


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=project_root / "outputs" / "rbf_gp"
    )
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--min-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--targets", type=int, default=32)
    parser.add_argument("--min-context", type=int, default=3)
    parser.add_argument("--max-context", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--representation-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--validation-functions", type=int, default=128)
    parser.add_argument("--test-functions", type=int, default=256)
    parser.add_argument("--grid-points", type=int, default=129)
    parser.add_argument("--x-min", type=float, default=-2.0)
    parser.add_argument("--x-max", type=float, default=2.0)
    parser.add_argument("--length-scale", type=float, default=0.4)
    parser.add_argument(
        "--signal-scale", "--signal-std", dest="signal_scale", type=float, default=1.0
    )
    parser.add_argument("--observation-noise", type=float, default=0.02)
    parser.add_argument("--jitter", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
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
        "validation_functions",
        "test_functions",
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
        raise ValueError("--max-context must cover every evaluation Context count")
    if args.max_context + args.targets > args.grid_points:
        raise ValueError(
            "--grid-points must cover disjoint maximum Context and target sets"
        )
    if args.grid_points < 21 or args.grid_points % 2 == 0:
        raise ValueError("--grid-points must be an odd integer of at least 21")
    if not math.isfinite(args.x_min) or not math.isfinite(args.x_max):
        raise ValueError("x bounds must be finite")
    if args.x_min >= args.x_max:
        raise ValueError("--x-min must be smaller than --x-max")
    if args.length_scale <= 0 or args.signal_scale <= 0 or args.jitter <= 0:
        raise ValueError("RBF kernel parameters must be positive")
    if args.observation_noise < 0:
        raise ValueError("--observation-noise cannot be negative")
    if args.learning_rate <= 0 or args.gradient_clip <= 0:
        raise ValueError("learning rate and gradient clip must be positive")


def print_test_summary(test: dict[str, dict[str, Any]], gate: dict[str, Any]) -> None:
    print("\nHeld-out RBF-GP unseen-Query function-macro RMSE")
    print(f"{'method':<24} {'C=3':>9} {'C=5':>9} {'C=10':>9} {'C=20':>9}")
    for key, label in (
        ("cnp", "CNP"),
        ("gp_posterior_oracle", "GP posterior (oracle)"),
        ("context_mean", "Context mean"),
    ):
        values = [test[str(count)][key]["function_macro_rmse"] for count in CONTEXT_COUNTS]
        print(f"{label:<24} " + " ".join(f"{value:9.4f}" for value in values))
    print(f"gate: {'PASS' if gate['passed'] else 'FAIL'}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = default_device() if args.device == "auto" else torch.device(args.device)
    x_range = (args.x_min, args.x_max)
    kernel = RBFKernelConfig(
        length_scale=args.length_scale,
        signal_scale=args.signal_scale,
        observation_noise=args.observation_noise,
        jitter=args.jitter,
    )
    grid = make_gp_grid(args.grid_points, x_range=x_range)
    gp_cholesky = make_gp_cholesky(grid, kernel)

    torch.manual_seed(args.seed)
    validation_seed = args.seed + 10_000
    test_seed = args.seed + 20_000
    validation_functions = sample_gp_functions(
        args.validation_functions,
        grid,
        kernel,
        generator=torch.Generator().manual_seed(validation_seed),
    )
    test_functions = sample_gp_functions(
        args.test_functions,
        grid,
        kernel,
        generator=torch.Generator().manual_seed(test_seed),
    )
    if tensor_fingerprint(validation_functions) == tensor_fingerprint(test_functions):
        raise RuntimeError("validation and test GP banks unexpectedly match")

    model_config = {
        "x_dim": 1,
        "y_dim": 1,
        "hidden_dim": args.hidden_dim,
        "representation_dim": args.representation_dim,
    }
    model = CNP(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_generator = torch.Generator().manual_seed(args.seed + 1_001)
    history: list[dict[str, Any]] = []
    checkpoint_path = args.output_dir / "best_model.pt"
    best_score = float("inf")
    best_step = 0
    validations_without_improvement = 0
    interval_loss_sum = 0.0
    interval_gradient_sum = 0.0
    interval_steps = 0
    context_observations = 0
    target_loss_points = 0
    final_step = 0
    start = time.perf_counter()

    print(
        f"device={device} RBF-GP length_scale={args.length_scale:g} "
        f"signal_scale={args.signal_scale:g} grid={args.grid_points}"
    )
    print(
        f"dynamic train episodes; fixed validation/test draws="
        f"{args.validation_functions}/{args.test_functions}"
    )

    for step in range(1, args.steps + 1):
        final_step = step
        batch = sample_rbf_training_batch(
            grid=grid,
            kernel=kernel,
            batch_size=args.batch_size,
            min_context=args.min_context,
            max_context=args.max_context,
            target_count=args.targets,
            generator=train_generator,
            device=device,
            cholesky=gp_cholesky,
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
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        optimizer.step()

        context_observations += int(batch.context_counts.sum().item())
        target_loss_points += args.batch_size * args.targets
        interval_loss_sum += float(loss.item())
        interval_gradient_sum += float(gradient_norm.item())
        interval_steps += 1

        if step == 1 or step % args.validation_interval == 0 or step == args.steps:
            validation = evaluate_model(
                model,
                validation_functions,
                grid,
                kernel,
                context_counts=CONTEXT_COUNTS,
                seed=validation_seed + 1,
                device=device,
                include_references=False,
            )
            validation_nll = average_cnp_metric(
                validation, "function_macro_nll"
            )
            validation_rmse = average_cnp_metric(
                validation, "function_macro_rmse"
            )
            if not math.isfinite(validation_nll) or not math.isfinite(
                validation_rmse
            ):
                raise FloatingPointError(
                    f"non-finite validation metric at step {step}"
                )
            row = {
                "step": step,
                "train_nll_mean": interval_loss_sum / interval_steps,
                "gradient_norm_mean_before_clip": (
                    interval_gradient_sum / interval_steps
                ),
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

            if validation_rmse < best_score:
                best_score = validation_rmse
                best_step = step
                validations_without_improvement = 0
                torch.save(
                    {
                        "format_version": 1,
                        "benchmark": "rbf_gp",
                        "step": step,
                        "model_state_dict": _state_dict_on_cpu(model),
                        "model_config": model_config,
                        "training_config": _json_args(args),
                        "x_range": list(x_range),
                        "grid_points": args.grid_points,
                        "kernel": asdict(kernel),
                        "evaluation_seeds": {
                            "validation_function_bank": validation_seed,
                            "validation_context": validation_seed + 1,
                            "test_function_bank": test_seed,
                            "test_context": test_seed + 1,
                        },
                        "bank_fingerprints": {
                            "validation": tensor_fingerprint(validation_functions),
                            "test": tensor_fingerprint(test_functions),
                        },
                        "selection_metric": (
                            "mean validation function-macro RMSE over Context "
                            "3/5/10/20"
                        ),
                        "best_validation_score": best_score,
                        "validation_nll_at_best": validation_nll,
                        "validation_at_best": validation,
                        "train_generator_state": train_generator.get_state(),
                        "function_episodes_at_best": step * args.batch_size,
                        "context_observations_at_best": context_observations,
                        "target_loss_points_at_best": target_loss_points,
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

    if not checkpoint_path.exists():
        raise RuntimeError("training completed without a finite checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test = evaluate_model(
        model,
        test_functions,
        grid,
        kernel,
        context_counts=CONTEXT_COUNTS,
        seed=test_seed + 1,
        device=device,
        include_references=True,
    )
    gate = final_test_gate(test)
    checkpoint["gate_passed"] = bool(gate["passed"])
    checkpoint["test_gate"] = gate
    checkpoint["test_summary"] = {
        str(count): {
            key: {
                "function_macro_rmse": float(
                    test[str(count)][key]["function_macro_rmse"]
                )
            }
            for key in ("cnp", "gp_posterior_oracle", "context_mean")
        }
        for count in CONTEXT_COUNTS
    }
    torch.save(checkpoint, checkpoint_path)
    duration = time.perf_counter() - start

    plot_status = {
        "learning_curve": False,
        "context_sweep": False,
        "predictions": False,
    }
    if not args.no_plot:
        plot_status = {
            "learning_curve": plot_learning_curve(
                history, args.output_dir / "learning_curve.png"
            ),
            "context_sweep": plot_context_sweep(
                test, args.output_dir / "context_sweep.png"
            ),
            "predictions": plot_predictions(
                model,
                test_functions,
                grid,
                kernel,
                args.output_dir / "predictions.png",
                context_count=10,
                seed=test_seed + 1,
                device=device,
            ),
        }

    results = {
        "format_version": 1,
        "benchmark": "rbf_gp",
        "purpose": (
            "Validate that the CNP learns a distribution over smooth functions "
            "and conditions on Context Y before applying it to heart-rate data."
        ),
        "data_generation": {
            "distribution": "zero-mean Gaussian Process",
            "kernel": "squared-exponential / RBF",
            "formula": (
                "k(x,x') = signal_scale^2 * "
                "exp(-(x-x')^2 / (2 * length_scale^2))"
            ),
            "kernel_parameters": asdict(kernel),
            "x_range": list(x_range),
            "grid_points": args.grid_points,
            "observation_noise_std": kernel.observation_noise,
            "sampling_note": (
                "Observation noise is included as diagonal covariance. Jitter is "
                "additional numerical stabilization. A fresh noisy GP realization "
                "is drawn for every training episode."
            ),
        },
        "context_counts": list(CONTEXT_COUNTS),
        "config": _json_args(args),
        "model_config": model_config,
        "fixed_banks": {
            "validation": {
                "functions": args.validation_functions,
                "seed": validation_seed,
                "sha256": tensor_fingerprint(validation_functions),
            },
            "test": {
                "functions": args.test_functions,
                "seed": test_seed,
                "sha256": tensor_fingerprint(test_functions),
            },
            "separation": (
                "Independent RNG seeds; test is evaluated only after reloading "
                "the validation-selected checkpoint."
            ),
        },
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
            "best_validation_rmse": float(checkpoint["best_validation_score"]),
            "validation_nll_at_best": float(
                checkpoint["validation_nll_at_best"]
            ),
            "function_episodes_final": final_step * args.batch_size,
            "context_observations_final": context_observations,
            "target_loss_points_final": target_loss_points,
            "function_episodes_at_best": int(
                checkpoint["function_episodes_at_best"]
            ),
            "context_observations_at_best": int(
                checkpoint["context_observations_at_best"]
            ),
            "target_loss_points_at_best": int(
                checkpoint["target_loss_points_at_best"]
            ),
            "history": history,
            "wall_time_seconds": duration,
        },
        "validation_at_best": checkpoint["validation_at_best"],
        "test": test,
        "gate": gate,
        "plots": plot_status,
        "interpretation_notes": [
            "The analytic GP posterior is privileged: it knows the exact kernel and hyperparameters.",
            "CNP intervals are pointwise predictive intervals, not simultaneous function bands.",
            "All reported test Query points are disjoint from Context positions.",
            "This is an implementation/distribution-learning test, not evidence of heart-rate accuracy.",
            "Future heart-rate forecasting is outside the scope of this interpolation benchmark.",
        ],
    }
    results_path = args.output_dir / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print_test_summary(test, gate)
    print(f"best_step={best_step} checkpoint={checkpoint_path}")
    print(f"results={results_path}")


if __name__ == "__main__":
    main()
