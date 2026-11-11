"""Run layered fixed Sin/Cos sanity checks for the standalone CNP."""

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
from cnp.sanity import (
    CANONICAL_FUNCTION_NAMES,
    CANONICAL_X_RANGE,
    evaluate_canonical_functions,
    make_canonical_evaluation_task,
    sample_canonical_training_batch,
)


CONTEXT_COUNTS = (3, 5, 10, 20)
GATE_CONTEXT_COUNT = 10
SUITE_FUNCTIONS = {
    "single_sin": (0,),
    "single_cos": (1,),
    "sin_cos_pair": (0, 1),
}


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


@torch.no_grad()
def evaluate_model(
    model: CNP,
    function_indices: Tensor,
    *,
    context_counts: Iterable[int],
    seeds: Iterable[int],
    grid_points: int,
    device: torch.device,
    include_wrong_context: bool,
) -> dict[str, Any]:
    """Evaluate fixed dense-grid Queries across several Context draws."""

    function_indices = function_indices.to(dtype=torch.long, device="cpu")
    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values:
        raise ValueError("evaluation needs at least one Context seed")
    was_training = model.training
    model.eval()
    results: dict[str, Any] = {}
    for context_count in context_counts:
        rmse_rows: list[Tensor] = []
        mae_rows: list[Tensor] = []
        nll_rows: list[Tensor] = []
        coverage_rows: list[Tensor] = []
        width_rows: list[Tensor] = []
        wrong_rmse_rows: list[Tensor] = []
        for seed in seed_values:
            task = make_canonical_evaluation_task(
                function_indices,
                context_count=context_count,
                grid_points=grid_points,
                seed=seed,
            )
            mean, scale = model(
                task.context_x.to(device),
                task.context_y.to(device),
                task.target_x.to(device),
            )
            mean = mean.cpu()
            scale = scale.cpu()
            error = mean - task.target_y
            rmse_rows.append(error.square().mean(dim=(1, 2)).sqrt())
            mae_rows.append(error.abs().mean(dim=(1, 2)))
            nll = (
                0.5 * math.log(2.0 * math.pi)
                + scale.log()
                + 0.5 * (error / scale).square()
            )
            nll_rows.append(nll.mean(dim=(1, 2)))
            coverage_rows.append(
                (error.abs() <= 1.96 * scale).to(torch.float32).mean(dim=(1, 2))
            )
            width_rows.append((3.92 * scale).mean(dim=(1, 2)))

            if include_wrong_context:
                if set(function_indices.tolist()) != {0, 1}:
                    raise ValueError("wrong-Context evaluation requires Sin and Cos")
                opposite = 1 - function_indices
                wrong_context_y = evaluate_canonical_functions(
                    opposite, task.context_x
                )
                wrong_mean, _ = model(
                    task.context_x.to(device),
                    wrong_context_y.to(device),
                    task.target_x.to(device),
                )
                wrong_error = wrong_mean.cpu() - task.target_y
                wrong_rmse_rows.append(
                    wrong_error.square().mean(dim=(1, 2)).sqrt()
                )

        rmse = torch.stack(rmse_rows)
        mae = torch.stack(mae_rows)
        nll = torch.stack(nll_rows)
        coverage = torch.stack(coverage_rows)
        width = torch.stack(width_rows)
        per_function: dict[str, Any] = {}
        for column, function_index in enumerate(function_indices.tolist()):
            name = CANONICAL_FUNCTION_NAMES[function_index]
            metrics: dict[str, float] = {
                "mean_rmse": float(rmse[:, column].mean().item()),
                "p95_rmse": float(
                    torch.quantile(rmse[:, column], 0.95).item()
                ),
                "mean_mae": float(mae[:, column].mean().item()),
                "mean_nll": float(nll[:, column].mean().item()),
                "coverage_95": float(coverage[:, column].mean().item()),
                "mean_interval_width_95": float(width[:, column].mean().item()),
            }
            if wrong_rmse_rows:
                wrong_rmse = torch.stack(wrong_rmse_rows)[:, column]
                metrics["wrong_context_mean_rmse"] = float(
                    wrong_rmse.mean().item()
                )
                metrics["conditioning_gap"] = (
                    metrics["wrong_context_mean_rmse"] - metrics["mean_rmse"]
                )
            per_function[name] = metrics
        results[str(context_count)] = {
            "context_draws": len(seed_values),
            "query_points_per_function": int(task.target_x.shape[1]),
            "macro_mean_rmse": float(rmse.mean().item()),
            "macro_p95_rmse": float(
                torch.stack(
                    [
                        torch.quantile(rmse[:, column], 0.95)
                        for column in range(rmse.shape[1])
                    ]
                ).mean().item()
            ),
            "per_function": per_function,
        }
    if was_training:
        model.train()
    return results


def training_gate(suite_name: str, evaluation: dict[str, Any]) -> dict[str, Any]:
    """Apply pre-declared hard gates to the Context=10 fixed evaluation."""

    condition = evaluation[str(GATE_CONTEXT_COUNT)]
    checks: dict[str, bool] = {}
    for name, metrics in condition["per_function"].items():
        checks[f"{name}_mean_rmse_below_0.04"] = metrics["mean_rmse"] < 0.04
        checks[f"{name}_p95_rmse_below_0.05"] = metrics["p95_rmse"] < 0.05
        if suite_name == "sin_cos_pair":
            checks[f"{name}_conditioning_gap_above_0.8"] = (
                metrics["conditioning_gap"] > 0.8
            )
    return {
        "context_count": GATE_CONTEXT_COUNT,
        "criteria_declared_before_final_evaluation": True,
        "checks": checks,
        "passed": all(checks.values()),
        "note": (
            "Single-function gates test the optimizer/decoder path only. "
            "The pair conditioning-gap gate tests use of Context Y."
        ),
    }


def train_suite(
    suite_name: str,
    function_indices: Tensor,
    *,
    steps: int,
    args: argparse.Namespace,
    model_config: dict[str, Any],
    device: torch.device,
    suite_seed: int,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Train one diagnostic suite and return results plus best state."""

    torch.manual_seed(suite_seed)
    model = CNP(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(suite_seed + 1)
    validation_seeds = range(suite_seed + 10_000, suite_seed + 10_008)
    test_seeds = range(suite_seed + 20_000, suite_seed + 20_000 + args.test_draws)
    include_wrong_context = suite_name == "sin_cos_pair"
    initial = evaluate_model(
        model,
        function_indices,
        context_counts=(GATE_CONTEXT_COUNT,),
        seeds=validation_seeds,
        grid_points=args.grid_points,
        device=device,
        include_wrong_context=include_wrong_context,
    )
    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_step = 0
    best_state = _state_dict_on_cpu(model)
    interval_loss = 0.0
    interval_steps = 0
    context_observations = 0
    target_loss_points = 0
    start = time.perf_counter()

    model.train()
    for step in range(1, steps + 1):
        batch = sample_canonical_training_batch(
            function_indices,
            batch_size=args.batch_size,
            min_context=args.min_context,
            max_context=args.max_context,
            target_count=args.targets,
            generator=generator,
            device=device,
        )
        mean, scale = model(
            batch.context_x,
            batch.context_y,
            batch.target_x,
            batch.context_mask,
        )
        loss = gaussian_nll(mean, scale, batch.target_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip
        )
        optimizer.step()
        if not torch.isfinite(loss):
            raise RuntimeError(f"{suite_name} produced non-finite loss at step {step}")
        interval_loss += float(loss.item())
        interval_steps += 1
        context_observations += int(batch.context_counts.sum().item())
        target_loss_points += args.batch_size * args.targets

        should_validate = step % args.validation_interval == 0 or step == steps
        if should_validate:
            validation = evaluate_model(
                model,
                function_indices,
                context_counts=(GATE_CONTEXT_COUNT,),
                seeds=validation_seeds,
                grid_points=args.grid_points,
                device=device,
                include_wrong_context=include_wrong_context,
            )
            score = float(validation[str(GATE_CONTEXT_COUNT)]["macro_mean_rmse"])
            row = {
                "step": step,
                "train_nll_mean": interval_loss / interval_steps,
                "validation_macro_rmse_c10": score,
                "gradient_norm": float(gradient_norm.item()),
            }
            history.append(row)
            interval_loss = 0.0
            interval_steps = 0
            if score < best_score:
                best_score = score
                best_step = step
                best_state = _state_dict_on_cpu(model)
            print(
                f"{suite_name:<13} step={step:5d}/{steps} "
                f"train_nll={row['train_nll_mean']:+.5f} "
                f"val_rmse={score:.5f} best={best_score:.5f}@{best_step}"
            )

    model.load_state_dict(best_state)
    final_evaluation = evaluate_model(
        model,
        function_indices,
        context_counts=CONTEXT_COUNTS,
        seeds=test_seeds,
        grid_points=args.grid_points,
        device=device,
        include_wrong_context=include_wrong_context,
    )
    result = {
        "function_names": [
            CANONICAL_FUNCTION_NAMES[index] for index in function_indices.tolist()
        ],
        "seed": suite_seed,
        "steps_requested": steps,
        "best_step": best_step,
        "best_validation_macro_rmse_c10": best_score,
        "duration_seconds": time.perf_counter() - start,
        "function_episodes": steps * args.batch_size,
        "context_observations": context_observations,
        "target_loss_points": target_loss_points,
        "initial_evaluation": initial,
        "history": history,
        "test_evaluation": final_evaluation,
        "gate": training_gate(suite_name, final_evaluation),
    }
    return result, best_state


def plot_learning_curves(suites: dict[str, Any], output: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    labels = {
        "single_sin": "Single Sin",
        "single_cos": "Single Cos",
        "sin_cos_pair": "One CNP: Sin/Cos",
    }
    for name, suite in suites.items():
        steps = [row["step"] for row in suite["history"]]
        axes[0].plot(
            steps,
            [row["train_nll_mean"] for row in suite["history"]],
            label=labels[name],
        )
        axes[1].plot(
            steps,
            [row["validation_macro_rmse_c10"] for row in suite["history"]],
            label=labels[name],
        )
    axes[0].set(title="Training objective", xlabel="Optimizer step", ylabel="Gaussian NLL")
    axes[1].set(title="Fixed validation", xlabel="Optimizer step", ylabel="RMSE at Context=10")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return True


def plot_context_sweep(suites: dict[str, Any], output: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    figure, axis = plt.subplots(figsize=(7.8, 4.8))
    labels = {
        "single_sin": "Single Sin",
        "single_cos": "Single Cos",
        "sin_cos_pair": "One CNP: Sin/Cos",
    }
    for name, suite in suites.items():
        evaluation = suite["test_evaluation"]
        axis.plot(
            CONTEXT_COUNTS,
            [evaluation[str(count)]["macro_mean_rmse"] for count in CONTEXT_COUNTS],
            marker="o",
            label=labels[name],
        )
    axis.axhline(0.04, color="tab:red", linestyle="--", linewidth=1, label="Mean gate (0.04)")
    axis.set(
        title="Canonical Sin/Cos sanity check",
        xlabel="Context points",
        ylabel="Dense-grid RMSE",
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
    model_config: dict[str, Any],
    states: dict[str, dict[str, Tensor]],
    output: Path,
    *,
    device: torch.device,
    seed: int,
    grid_points: int,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True, sharey=True)
    suite_columns = ("single", "sin_cos_pair")
    function_indices = torch.tensor([0, 1])
    query = torch.linspace(*CANONICAL_X_RANGE, 401).view(1, -1, 1)
    for row, function_index in enumerate(function_indices.tolist()):
        task = make_canonical_evaluation_task(
            torch.tensor([function_index]),
            context_count=GATE_CONTEXT_COUNT,
            grid_points=grid_points,
            seed=seed,
        )
        truth = evaluate_canonical_functions(torch.tensor([function_index]), query)[0]
        for column, suite_kind in enumerate(suite_columns):
            suite_name = (
                f"single_{CANONICAL_FUNCTION_NAMES[function_index]}"
                if suite_kind == "single"
                else suite_kind
            )
            model = CNP(**model_config).to(device)
            model.load_state_dict(states[suite_name])
            model.eval()
            mean, scale = model(
                task.context_x.to(device),
                task.context_y.to(device),
                query.to(device),
            )
            mean, scale = mean.cpu()[0], scale.cpu()[0]
            axis = axes[row, column]
            axis.plot(query[0, :, 0], truth[:, 0], color="black", label="Truth")
            axis.plot(query[0, :, 0], mean[:, 0], color="tab:blue", label="CNP mean")
            axis.fill_between(
                query[0, :, 0],
                mean[:, 0] - 1.96 * scale[:, 0],
                mean[:, 0] + 1.96 * scale[:, 0],
                color="tab:blue",
                alpha=0.15,
                label="95% interval",
            )
            axis.scatter(
                task.context_x[0, :, 0],
                task.context_y[0, :, 0],
                color="tab:red",
                s=24,
                zorder=4,
                label="Context",
            )
            axis.set_title(
                f"{CANONICAL_FUNCTION_NAMES[function_index].title()} / "
                f"{'single model' if column == 0 else 'shared Sin/Cos model'}"
            )
            axis.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Synthetic coordinate x (radians)")
    axes[1, 1].set_xlabel("Synthetic coordinate x (radians)")
    axes[0, 0].set_ylabel("y")
    axes[1, 0].set_ylabel("y")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return True


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=project_root / "outputs" / "sanity"
    )
    parser.add_argument("--single-steps", type=int, default=10_000)
    parser.add_argument("--pair-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--targets", type=int, default=32)
    parser.add_argument("--min-context", type=int, default=3)
    parser.add_argument("--max-context", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--representation-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--grid-points", type=int, default=129)
    parser.add_argument("--test-draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "single_steps",
        "pair_steps",
        "batch_size",
        "targets",
        "min_context",
        "max_context",
        "hidden_dim",
        "representation_dim",
        "validation_interval",
        "grid_points",
        "test_draws",
    )
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.min_context > args.max_context:
        raise ValueError("--min-context cannot exceed --max-context")
    if args.max_context < max(CONTEXT_COUNTS):
        raise ValueError("--max-context must cover every evaluation Context count")
    if args.grid_points < 3 or args.grid_points % 2 == 0:
        raise ValueError("--grid-points must be an odd integer of at least 3")
    if args.grid_points < args.max_context:
        raise ValueError("--grid-points must provide every requested Context point")
    if args.learning_rate <= 0 or args.gradient_clip <= 0:
        raise ValueError("learning rate and gradient clip must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = default_device() if args.device == "auto" else torch.device(args.device)
    model_config = {
        "x_dim": 1,
        "y_dim": 1,
        "hidden_dim": args.hidden_dim,
        "representation_dim": args.representation_dim,
    }
    suite_results: dict[str, Any] = {}
    suite_states: dict[str, dict[str, Tensor]] = {}
    suite_checkpoint_entries: dict[str, Any] = {}
    total_start = time.perf_counter()
    for suite_offset, (suite_name, indices) in enumerate(SUITE_FUNCTIONS.items()):
        steps = args.pair_steps if suite_name == "sin_cos_pair" else args.single_steps
        suite_seed = args.seed + suite_offset * 1_000
        function_indices = torch.tensor(indices, dtype=torch.long)
        result, state = train_suite(
            suite_name,
            function_indices,
            steps=steps,
            args=args,
            model_config=model_config,
            device=device,
            suite_seed=suite_seed,
        )
        suite_results[suite_name] = result
        suite_states[suite_name] = state
        suite_checkpoint_entries[suite_name] = {
            "step": result["best_step"],
            "function_names": result["function_names"],
            "model_state_dict": state,
            "gate_passed": result["gate"]["passed"],
        }

    checkpoint = {
        "format_version": 1,
        "benchmark": "canonical_sanity",
        "model_config": model_config,
        "x_range": list(CANONICAL_X_RANGE),
        "function_definitions": {
            "sin": "sin(x)",
            "cos": "cos(x)",
        },
        "models": suite_checkpoint_entries,
        "training_config": _json_args(args),
        "selection_metric": "fixed Context=10 dense-grid macro RMSE",
    }
    checkpoint_path = args.output_dir / "best_models.pt"
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    plot_status = {
        "learning_curves": False,
        "context_sweep": False,
        "predictions": False,
    }
    if not args.no_plot:
        plot_status = {
            "learning_curves": plot_learning_curves(
                suite_results, args.output_dir / "learning_curves.png"
            ),
            "context_sweep": plot_context_sweep(
                suite_results, args.output_dir / "context_sweep.png"
            ),
            "predictions": plot_predictions(
                model_config,
                suite_states,
                args.output_dir / "predictions.png",
                device=device,
                seed=args.seed + 90_000,
                grid_points=args.grid_points,
            ),
        }
    results = {
        "format_version": 1,
        "benchmark": "canonical_sanity",
        "purpose": (
            "Layered implementation sanity check; not evidence for heart-rate "
            "performance or unseen-function generalization."
        ),
        "function_definitions": checkpoint["function_definitions"],
        "x_range": list(CANONICAL_X_RANGE),
        "context_counts": list(CONTEXT_COUNTS),
        "gate_context_count": GATE_CONTEXT_COUNT,
        "model_config": model_config,
        "training_config": _json_args(args),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": str(device),
        },
        "duration_seconds": time.perf_counter() - total_start,
        "checkpoint": {
            "file": checkpoint_path.name,
            "sha256": checkpoint_sha256,
        },
        "all_gates_passed": all(
            suite["gate"]["passed"] for suite in suite_results.values()
        ),
        "suites": suite_results,
        "plots": plot_status,
    }
    results_path = args.output_dir / "results.json"
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\nCanonical Sin/Cos fixed test")
    print(f"{'suite':<15} {'C10 RMSE':>10} {'gate':>8} {'best step':>10}")
    for name, suite in suite_results.items():
        value = suite["test_evaluation"][str(GATE_CONTEXT_COUNT)][
            "macro_mean_rmse"
        ]
        print(
            f"{name:<15} {value:10.5f} "
            f"{('PASS' if suite['gate']['passed'] else 'FAIL'):>8} "
            f"{suite['best_step']:10d}"
        )
    print(f"all gates: {'PASS' if results['all_gates_passed'] else 'FAIL'}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"results:    {results_path}")


if __name__ == "__main__":
    main()
