"""Native macOS GUI for canonical Sin/Cos and RBF-GP CNP checks."""

from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Union

# Keep this import order.  On the supported conda environment, initializing
# TkAgg before PyTorch avoids competing OpenMP runtimes during Finder launch.
import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from cnp.rbf_gp_demo import RBFGPDemoRuntime, RBFGPPredictionResult
from cnp.sanity import CANONICAL_FUNCTION_NAMES, CANONICAL_X_RANGE
from cnp.sanity_demo import (
    SUITE_KEYS,
    SanityDemoRuntime,
    SanityPredictionResult,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODE_CANONICAL = "canonical_sanity"
MODE_RBF_GP = "rbf_gp"
MODE_LABELS = {
    MODE_CANONICAL: "Canonical Sin / Cos",
    MODE_RBF_GP: "RBF-GP（未知関数）",
}
MODE_KEYS_BY_LABEL = {label: key for key, label in MODE_LABELS.items()}

SUITE_LABELS = {
    "single_sin": "Sin only（Sinだけで学習）",
    "single_cos": "Cos only（Cosだけで学習）",
    "sin_cos_pair": "Sin + Cos（共有モデル）",
}
SUITE_KEYS_BY_LABEL = {label: key for key, label in SUITE_LABELS.items()}

QUERY_STEP = 0.001

METRICS = (
    ("cnp_mean", "CNP Prediction"),
    ("cnp_scale", "CNP SD"),
    ("truth", "Query truth"),
    ("error", "Absolute error"),
    ("rmse", "All-Query RMSE"),
)

PredictionResult = Union[SanityPredictionResult, RBFGPPredictionResult]


def _default_suite_functions() -> dict[str, tuple[str, ...]]:
    names = tuple(str(name) for name in CANONICAL_FUNCTION_NAMES)
    sin_name = next(
        (name for name in names if "sin" in name.lower() and "cos" not in name.lower()),
        names[0] if names else "sin",
    )
    cos_name = next(
        (name for name in names if "cos" in name.lower()),
        names[1] if len(names) > 1 else sin_name,
    )
    return {
        "single_sin": (sin_name,),
        "single_cos": (cos_name,),
        "sin_cos_pair": tuple(dict.fromkeys((sin_name, cos_name))),
    }


def metric_values(result: PredictionResult) -> dict[str, float]:
    if isinstance(result, RBFGPPredictionResult):
        return {
            "cnp_mean": result.cnp_query_y,
            "cnp_scale": result.cnp_query_scale,
            "truth": result.truth_query_y,
            "error": result.cnp_absolute_error,
            "rmse": result.cnp_target_rmse,
        }
    return {
        "cnp_mean": result.cnp_query_y,
        "cnp_scale": result.cnp_query_scale,
        "truth": result.truth_query_y,
        "error": result.absolute_error,
        "rmse": result.target_rmse,
    }


def status_text(result: PredictionResult, rbf_runtime: RBFGPDemoRuntime | None = None) -> str:
    if isinstance(result, RBFGPPredictionResult):
        gate_value = None if rbf_runtime is None else rbf_runtime.gate_passed
        gate = (
            "PASS"
            if gate_value is True
            else "FAIL"
            if gate_value is False
            else "not recorded"
        )
        step = "?" if rbf_runtime is None else str(rbf_runtime.step)
        kernel = "" if rbf_runtime is None else (
            f"ell={rbf_runtime.kernel.length_scale:g}, "
            f"sigma={rbf_runtime.kernel.signal_scale:g}, "
            f"noise={rbf_runtime.kernel.observation_noise:g}"
        )
        return (
            f"RBF-GP draw seed {result.function_seed} / checkpoint step {step} / "
            f"fixed-test gate {gate} / {kernel}\n"
            f"Input: {result.context_count} Context pairs (X,Y), seed {result.context_seed}. "
            f"Query X={result.query_x:.4f} -> CNP Y={result.cnp_query_y:.4f}, "
            f"analytic GP Y={result.gp_query_y:.4f}, truth={result.truth_query_y:.4f}\n"
            f"Dense Query RMSE: CNP={result.cnp_target_rmse:.4f}, "
            f"analytic GP={result.gp_target_rmse:.4f}. "
            "function seed・kernel・正解YはCNP入力に渡していません。"
        )

    gate = (
        "PASS"
        if result.suite.gate_passed is True
        else "FAIL"
        if result.suite.gate_passed is False
        else "not recorded"
    )
    return (
        f"Suite: {SUITE_LABELS[result.suite.key]} / function: {result.function_name} / "
        f"checkpoint step {result.suite.step} / fixed-test gate {gate}\n"
        f"Input: {result.context_count} Context pairs (X,Y), seed {result.context_seed}. "
        f"Query X={result.query_x:.4f} → "
        f"Prediction Y={result.cnp_query_y:.4f}\n"
        "モデル入力はContext (X,Y)とQuery Xだけです。選択したfunctionラベルと"
        "Queryの正解Yは入力していません。"
    )


def draw_prediction(axis, result: PredictionResult) -> None:
    if isinstance(result, RBFGPPredictionResult):
        _draw_rbf_prediction(axis, result)
        return

    target_x = result.target_x.numpy()
    truth = result.truth_y.numpy()
    mean = result.cnp_mean.numpy()
    scale = result.cnp_scale.numpy()
    context_x = result.context_x.numpy()
    context_y = result.context_y.numpy()

    axis.clear()
    axis.plot(
        target_x,
        truth,
        color="#111827",
        linewidth=1.8,
        alpha=0.72,
        label="Ground truth (display only)",
    )
    axis.fill_between(
        target_x,
        mean - 1.96 * scale,
        mean + 1.96 * scale,
        color="#2563eb",
        alpha=0.16,
        label="CNP pointwise 95% interval",
    )
    axis.plot(target_x, mean, color="#1d4ed8", linewidth=2.3, label="CNP mean")
    axis.scatter(
        context_x,
        context_y,
        s=36,
        color="#15803d",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="Context given to CNP",
    )
    axis.axvline(
        result.query_x,
        color="#a21caf",
        linewidth=1.5,
        linestyle=":",
        label=f"Query X={result.query_x:.3f}",
    )
    axis.scatter(
        [result.query_x],
        [result.cnp_query_y],
        s=74,
        color="#a21caf",
        edgecolor="white",
        linewidth=1.2,
        zorder=7,
        label="Query prediction",
    )
    axis.scatter(
        [result.query_x],
        [result.truth_query_y],
        s=52,
        facecolor="white",
        edgecolor="#111827",
        linewidth=1.4,
        zorder=6,
        label="Query truth (display only)",
    )
    axis.set_xlim(*CANONICAL_X_RANGE)
    axis.set_xlabel("Synthetic coordinate X (radians)")
    axis.set_ylabel("Y")
    axis.set_title(
        f"{result.function_name}: Context={result.context_count}, "
        f"seed={result.context_seed}"
    )
    axis.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.55)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        frameon=False,
        fontsize=8,
    )


def _draw_rbf_prediction(axis, result: RBFGPPredictionResult) -> None:
    target_x = result.target_x.numpy()
    truth = result.truth_y.numpy()
    cnp_mean = result.cnp_mean.numpy()
    cnp_scale = result.cnp_scale.numpy()
    gp_mean = result.gp_mean.numpy()
    gp_scale = result.gp_scale.numpy()
    context_x = result.context_x.numpy()
    context_y = result.context_y.numpy()

    axis.clear()
    axis.plot(
        target_x,
        truth,
        color="#111827",
        linewidth=1.6,
        alpha=0.72,
        label="Sampled RBF-GP truth (display only)",
    )
    axis.fill_between(
        target_x,
        gp_mean - 1.96 * gp_scale,
        gp_mean + 1.96 * gp_scale,
        color="#d97706",
        alpha=0.10,
        label="Analytic GP 95% interval",
    )
    axis.plot(
        target_x,
        gp_mean,
        color="#d97706",
        linewidth=1.8,
        linestyle="--",
        label="Analytic GP posterior",
    )
    axis.fill_between(
        target_x,
        cnp_mean - 1.96 * cnp_scale,
        cnp_mean + 1.96 * cnp_scale,
        color="#2563eb",
        alpha=0.16,
        label="CNP pointwise 95% interval",
    )
    axis.plot(target_x, cnp_mean, color="#1d4ed8", linewidth=2.3, label="CNP mean")
    axis.scatter(
        context_x,
        context_y,
        s=38,
        color="#15803d",
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
        label="Context given to both models",
    )
    axis.axvline(
        result.query_x,
        color="#a21caf",
        linewidth=1.5,
        linestyle=":",
        label=f"Exact Query X={result.query_x:.3f}",
    )
    axis.scatter(
        [result.query_x],
        [result.cnp_query_y],
        s=76,
        color="#a21caf",
        edgecolor="white",
        linewidth=1.2,
        zorder=8,
        label="CNP Query prediction",
    )
    axis.scatter(
        [result.query_x],
        [result.gp_query_y],
        s=58,
        marker="s",
        color="#d97706",
        edgecolor="white",
        linewidth=1.0,
        zorder=7,
        label="GP Query prediction",
    )
    axis.scatter(
        [result.query_x],
        [result.truth_query_y],
        s=52,
        facecolor="white",
        edgecolor="#111827",
        linewidth=1.4,
        zorder=7,
        label="Query truth (display only)",
    )
    axis.set_xlim(-2.0, 2.0)
    axis.set_xlabel("Synthetic coordinate X")
    axis.set_ylabel("Y")
    axis.set_title(
        f"RBF-GP draw {result.function_seed}: Context={result.context_count}, "
        f"Context seed={result.context_seed}"
    )
    axis.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.55)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        frameon=False,
        fontsize=8,
    )


class SanityDesktopApp:
    """Tkinter shell for canonical and RBF-GP models."""

    def __init__(
        self,
        root: tk.Tk,
        runtime: SanityDemoRuntime | None,
        *,
        checkpoint_path: Path,
        load_error: str | None = None,
        rbf_runtime: RBFGPDemoRuntime | None = None,
        rbf_checkpoint_path: Path | None = None,
        rbf_load_error: str | None = None,
    ) -> None:
        self.root = root
        self.runtime = runtime
        self.rbf_runtime = rbf_runtime
        self.checkpoint_path = checkpoint_path
        self.rbf_checkpoint_path = (
            PROJECT_ROOT / "outputs" / "rbf_gp" / "best_model.pt"
            if rbf_checkpoint_path is None
            else rbf_checkpoint_path
        )
        self.load_error = load_error
        self.rbf_load_error = rbf_load_error
        self.last_result: PredictionResult | None = None
        self._updating_controls = False
        self._default_functions = _default_suite_functions()

        self._configure_window()
        self._configure_styles()
        self._build_menu()
        self._build_layout()
        self._rebuild_functions(refresh=False)
        self._apply_mode(refresh=False)
        if self._active_runtime() is not None:
            self.refresh_prediction()
        else:
            self._show_unavailable_state()

    def _configure_window(self) -> None:
        self.root.title("CNP Function Explorer")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        width = min(1420, screen_width - 80)
        height = min(900, screen_height - 100)
        x = max(20, (screen_width - width) // 2)
        y = max(30, (screen_height - height) // 3)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(1060, 700)
        self.root.option_add("*tearOff", False)
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        try:
            self.root.createcommand("tk::mac::Quit", self.root.destroy)
        except tk.TclError:
            pass

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.configure("Title.TLabel", font=("Helvetica Neue", 24, "bold"))
        style.configure("Subtitle.TLabel", font=("Helvetica Neue", 12))
        style.configure("Section.TLabel", font=("Helvetica Neue", 12, "bold"))
        style.configure("Value.TLabel", font=("Helvetica Neue", 18, "bold"))
        style.configure("MetricName.TLabel", font=("Helvetica Neue", 10, "bold"))
        style.configure("Metric.TFrame", relief="solid", borderwidth=1)
        style.configure("Status.TLabel", font=("Helvetica Neue", 11))
        style.configure(
            "Warning.TLabel",
            foreground="#b45309",
            font=("Helvetica Neue", 11, "bold"),
        )

    def _build_menu(self) -> None:
        menu = tk.Menu(self.root)
        app_menu = tk.Menu(menu)
        app_menu.add_command(label="このアプリについて", command=self._show_about)
        app_menu.add_separator()
        app_menu.add_command(label="終了", command=self.root.destroy, accelerator="⌘Q")
        menu.add_cascade(label="CNP Explorer", menu=app_menu)
        self.root.configure(menu=menu)

    def _show_about(self) -> None:
        messagebox.showinfo(
            "CNP Function Explorer",
            "固定Sin/Cosと、毎episode新しい関数を生成するRBF-GPで、"
            "CNPのContext→Query経路を検証するネイティブアプリです。\n\n"
            "function情報とQueryの正解YはCNP入力に渡しません。",
            parent=self.root,
        )

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=16)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        self.title_var = tk.StringVar(value="CNP Function Explorer")
        self.subtitle_var = tk.StringVar()
        ttk.Label(outer, textvariable=self.title_var, style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            outer,
            textvariable=self.subtitle_var,
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 12))

        workspace = ttk.Frame(outer)
        workspace.grid(row=2, column=0, sticky="nsew")
        workspace.rowconfigure(0, weight=1)
        workspace.columnconfigure(1, weight=1)

        controls = ttk.LabelFrame(workspace, text="入力", padding=14)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 14))
        controls.columnconfigure(0, weight=1)

        row = 0
        default_mode = MODE_RBF_GP if self.rbf_runtime is not None else MODE_CANONICAL
        self.mode_var = tk.StringVar(value=MODE_LABELS[default_mode])
        self.mode_combo = self._add_combo(
            controls,
            row,
            "Experiment mode",
            self.mode_var,
            tuple(MODE_LABELS.values()),
            self._on_mode_changed,
        )
        row += 2

        self.mode_options = ttk.Frame(controls)
        self.mode_options.grid(row=row, column=0, sticky="ew")
        self.mode_options.columnconfigure(0, weight=1)

        self.canonical_options = ttk.Frame(self.mode_options)
        self.canonical_options.grid(row=0, column=0, sticky="ew")
        self.canonical_options.columnconfigure(0, weight=1)
        self.suite_var = tk.StringVar(value=SUITE_LABELS["sin_cos_pair"])
        self.suite_combo = self._add_combo(
            self.canonical_options,
            0,
            "Training suite / model",
            self.suite_var,
            tuple(SUITE_LABELS[key] for key in SUITE_KEYS),
            self._on_suite_changed,
        )
        self.function_var = tk.StringVar()
        self.function_combo = self._add_combo(
            self.canonical_options,
            2,
            "Ground-truth function",
            self.function_var,
            (),
            self._on_function_changed,
        )

        self.rbf_options = ttk.Frame(self.mode_options)
        self.rbf_options.grid(row=0, column=0, sticky="ew")
        self.rbf_options.columnconfigure(0, weight=1)
        self.function_seed_var = tk.IntVar(value=7001)
        self.function_seed_value_var = tk.StringVar(value="7001")
        rbf_row = self._add_integer_control(
            self.rbf_options,
            0,
            "RBF-GP function seed",
            self.function_seed_var,
            self.function_seed_value_var,
            0,
            999_999,
            self._sync_function_seed_label,
        )
        self.resample_function_button = ttk.Button(
            self.rbf_options,
            text="新しいGP関数を生成",
            command=self._resample_function,
        )
        self.resample_function_button.grid(
            row=rbf_row, column=0, sticky="ew", pady=(6, 0)
        )
        row += 1

        self.context_var = tk.IntVar(value=5)
        self.context_value_var = tk.StringVar(value="5 points")
        row = self._add_integer_control(
            controls,
            row,
            "Context count",
            self.context_var,
            self.context_value_var,
            3,
            20,
            self._sync_context_label,
        )

        default_query = 0.0
        self.query_var = tk.DoubleVar(value=default_query)
        self.query_value_var = tk.StringVar(value=f"{default_query:.4f}")
        heading = ttk.Frame(controls)
        heading.grid(row=row, column=0, sticky="ew", pady=(12, 2))
        heading.columnconfigure(0, weight=1)
        ttk.Label(heading, text="Query X", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(heading, textvariable=self.query_value_var).grid(
            row=0, column=1, sticky="e"
        )
        self.query_scale = tk.Scale(
            controls,
            from_=CANONICAL_X_RANGE[0],
            to=CANONICAL_X_RANGE[1],
            orient=tk.HORIZONTAL,
            variable=self.query_var,
            resolution=QUERY_STEP,
            showvalue=False,
            highlightthickness=0,
            borderwidth=0,
            command=lambda _value: self._sync_query_label(),
        )
        self.query_scale.grid(row=row + 1, column=0, sticky="ew")
        self.query_scale.bind(
            "<ButtonRelease-1>", lambda _event: self.refresh_prediction()
        )
        self.query_scale.bind("<KeyRelease>", lambda _event: self.refresh_prediction())
        row += 2

        self.seed_var = tk.IntVar(value=2026)
        self.seed_value_var = tk.StringVar(value="2026")
        row = self._add_integer_control(
            controls,
            row,
            "Context seed",
            self.seed_var,
            self.seed_value_var,
            0,
            999_999,
            self._sync_seed_label,
        )

        self.resample_button = ttk.Button(
            controls,
            text="Contextを再抽選",
            command=self._resample_context,
        )
        self.resample_button.grid(row=row, column=0, sticky="ew", pady=(10, 4))
        row += 1
        self.update_button = ttk.Button(
            controls,
            text="Predictionを更新",
            command=self.refresh_prediction,
            default="active",
        )
        self.update_button.grid(row=row, column=0, sticky="ew", pady=(4, 14))
        row += 1

        ttk.Separator(controls).grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        ttk.Label(controls, text="重要", style="Section.TLabel").grid(
            row=row, column=0, sticky="w"
        )
        row += 1
        self.input_contract_var = tk.StringVar()
        ttk.Label(
            controls,
            textvariable=self.input_contract_var,
            justify="left",
        ).grid(row=row, column=0, sticky="w")

        plot_panel = ttk.Frame(workspace)
        plot_panel.grid(row=0, column=1, sticky="nsew")
        plot_panel.rowconfigure(0, weight=1)
        plot_panel.columnconfigure(0, weight=1)
        self.figure = Figure(figsize=(10.5, 5.8), dpi=100, layout="constrained")
        self.axis = self.figure.subplots()
        self.axis.text(
            0.5,
            0.5,
            "Loading CNP checkpoints…",
            ha="center",
            va="center",
            transform=self.axis.transAxes,
        )
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_panel)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar_frame = ttk.Frame(plot_panel)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(
            self.canvas, toolbar_frame, pack_toolbar=False
        )
        self.toolbar.update()
        self.toolbar.grid(row=0, column=0, sticky="w")

        metrics_frame = ttk.Frame(outer)
        metrics_frame.grid(row=3, column=0, sticky="ew", pady=(14, 8))
        self.metric_vars: dict[str, tk.StringVar] = {}
        for index, (key, label) in enumerate(METRICS):
            metrics_frame.columnconfigure(index, weight=1, uniform="metric")
            card = ttk.Frame(metrics_frame, padding=10, style="Metric.TFrame")
            card.grid(
                row=0,
                column=index,
                sticky="nsew",
                padx=(0 if index == 0 else 4, 0 if index == len(METRICS) - 1 else 4),
            )
            ttk.Label(card, text=label, style="MetricName.TLabel").grid(
                row=0, column=0, sticky="w"
            )
            value_var = tk.StringVar(value="—")
            self.metric_vars[key] = value_var
            ttk.Label(card, textvariable=value_var, style="Value.TLabel").grid(
                row=1, column=0, sticky="w", pady=(5, 0)
            )

        status_frame = ttk.Frame(outer, padding=(2, 8, 2, 0))
        status_frame.grid(row=4, column=0, sticky="ew")
        status_frame.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar()
        ttk.Label(
            status_frame,
            textvariable=self.status_var,
            style="Status.TLabel",
            justify="left",
        ).grid(row=0, column=0, sticky="w")
        self.warning_var = tk.StringVar()
        ttk.Label(
            status_frame,
            textvariable=self.warning_var,
            style="Warning.TLabel",
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.root.bind("<Return>", lambda _event: self.refresh_prediction())

    def _add_combo(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        values: tuple[str, ...],
        callback,
    ) -> ttk.Combobox:
        ttk.Label(parent, text=label, style="Section.TLabel").grid(
            row=row, column=0, sticky="w", pady=(0 if row == 0 else 10, 4)
        )
        combo = ttk.Combobox(
            parent,
            textvariable=variable,
            values=values,
            state="readonly",
            width=35,
        )
        combo.grid(row=row + 1, column=0, sticky="ew")
        combo.bind("<<ComboboxSelected>>", callback)
        return combo

    def _add_integer_control(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.IntVar,
        value_label: tk.StringVar,
        minimum: int,
        maximum: int,
        sync_callback,
    ) -> int:
        heading = ttk.Frame(parent)
        heading.grid(row=row, column=0, sticky="ew", pady=(12, 2))
        heading.columnconfigure(0, weight=1)
        ttk.Label(heading, text=label, style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(heading, textvariable=value_label).grid(row=0, column=1, sticky="e")
        input_row = ttk.Frame(parent)
        input_row.grid(row=row + 1, column=0, sticky="ew")
        input_row.columnconfigure(0, weight=1)
        scale = tk.Scale(
            input_row,
            from_=minimum,
            to=maximum,
            orient=tk.HORIZONTAL,
            variable=variable,
            resolution=1,
            showvalue=False,
            highlightthickness=0,
            borderwidth=0,
            command=lambda _value: sync_callback(),
        )
        scale.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        spin = ttk.Spinbox(
            input_row,
            from_=minimum,
            to=maximum,
            textvariable=variable,
            width=7,
            command=self._refresh_from_controls,
        )
        spin.grid(row=0, column=1, sticky="e")
        scale.bind("<ButtonRelease-1>", lambda _event: self.refresh_prediction())
        scale.bind("<KeyRelease>", lambda _event: self.refresh_prediction())
        spin.bind("<FocusOut>", lambda _event: self._refresh_from_controls())
        return row + 2

    def _selected_suite_key(self) -> str:
        try:
            return SUITE_KEYS_BY_LABEL[self.suite_var.get()]
        except KeyError as error:
            raise ValueError(f"unknown suite selection: {self.suite_var.get()!r}") from error

    def _selected_mode_key(self) -> str:
        try:
            return MODE_KEYS_BY_LABEL[self.mode_var.get()]
        except KeyError as error:
            raise ValueError(f"unknown mode selection: {self.mode_var.get()!r}") from error

    def _active_runtime(self):
        return (
            self.runtime
            if self._selected_mode_key() == MODE_CANONICAL
            else self.rbf_runtime
        )

    def _active_x_range(self) -> tuple[float, float]:
        if self._selected_mode_key() == MODE_CANONICAL:
            return CANONICAL_X_RANGE
        if self.rbf_runtime is not None:
            return self.rbf_runtime.x_range
        return (-2.0, 2.0)

    def _apply_mode(self, *, refresh: bool = True) -> None:
        mode = self._selected_mode_key()
        self._updating_controls = True
        try:
            if mode == MODE_CANONICAL:
                self.rbf_options.grid_remove()
                self.canonical_options.grid()
                self.subtitle_var.set(
                    "既知の sin(x) / cos(x), x∈[-π,π] で、"
                    "Contextから任意Query XのYを復元します。"
                )
                self.input_contract_var.set(
                    "モデル入力:\n"
                    "  • Contextの(X,Y)\n"
                    "  • QueryのX\n\n"
                    "入力しないもの:\n"
                    "  • Sin/Cosラベル\n"
                    "  • Queryの正解Y"
                )
            else:
                self.canonical_options.grid_remove()
                self.rbf_options.grid()
                self.subtitle_var.set(
                    "RBF kernelから生成した未知関数で、CNPと解析的GP posteriorを"
                    "同じContext・任意Query Xで比較します。"
                )
                self.input_contract_var.set(
                    "CNPへの入力:\n"
                    "  • Contextの(X,Y)\n"
                    "  • QueryのX\n\n"
                    "CNPへ入力しないもの:\n"
                    "  • function seed / kernel\n"
                    "  • Queryの正解Y"
                )

            x_min, x_max = self._active_x_range()
            self.query_scale.configure(from_=x_min, to=x_max)
            query_x = float(self.query_var.get())
            if query_x < x_min or query_x > x_max:
                self.query_var.set(min(max(query_x, x_min), x_max))
            self._sync_query_label()
        finally:
            self._updating_controls = False

        available = self._active_runtime() is not None
        self.update_button.state(["!disabled"] if available else ["disabled"])
        self.resample_button.state(["!disabled"] if available else ["disabled"])
        if refresh:
            if available:
                self.refresh_prediction()
            else:
                self._show_unavailable_state()

    def _functions_for_suite(self, suite_key: str) -> tuple[str, ...]:
        if self.runtime is None:
            return self._default_functions[suite_key]
        return self.runtime.suite(suite_key).function_names

    def _rebuild_functions(self, *, refresh: bool = True) -> None:
        suite_key = self._selected_suite_key()
        values = self._functions_for_suite(suite_key)
        self.function_combo.configure(values=values)
        if self.function_var.get() not in values:
            self.function_var.set(values[0])
        if refresh:
            self.refresh_prediction()

    def _on_suite_changed(self, _event=None) -> None:
        if not self._updating_controls:
            self._rebuild_functions()

    def _on_function_changed(self, _event=None) -> None:
        if not self._updating_controls:
            self.refresh_prediction()

    def _on_mode_changed(self, _event=None) -> None:
        if not self._updating_controls:
            self._apply_mode()

    def _sync_context_label(self) -> None:
        try:
            self.context_value_var.set(f"{int(self.context_var.get())} points")
        except (tk.TclError, ValueError):
            pass

    def _sync_query_label(self) -> None:
        try:
            self.query_value_var.set(f"{float(self.query_var.get()):.4f}")
        except (tk.TclError, ValueError):
            pass

    def _sync_seed_label(self) -> None:
        try:
            self.seed_value_var.set(str(int(self.seed_var.get())))
        except (tk.TclError, ValueError):
            pass

    def _sync_function_seed_label(self) -> None:
        try:
            self.function_seed_value_var.set(str(int(self.function_seed_var.get())))
        except (tk.TclError, ValueError):
            pass

    def _refresh_from_controls(self) -> None:
        self._sync_context_label()
        self._sync_query_label()
        self._sync_seed_label()
        self._sync_function_seed_label()
        self.refresh_prediction()

    def _resample_context(self) -> None:
        try:
            self.seed_var.set((int(self.seed_var.get()) + 1) % 1_000_000)
        except (tk.TclError, ValueError):
            self.seed_var.set(2026)
        self._sync_seed_label()
        self.refresh_prediction()

    def _resample_function(self) -> None:
        try:
            self.function_seed_var.set(
                (int(self.function_seed_var.get()) + 1) % 1_000_000
            )
        except (tk.TclError, ValueError):
            self.function_seed_var.set(7001)
        self._sync_function_seed_label()
        self.refresh_prediction()

    @staticmethod
    def _read_integer(
        variable: tk.IntVar,
        minimum: int,
        maximum: int,
        name: str,
    ) -> int:
        try:
            value = int(variable.get())
        except (tk.TclError, ValueError) as error:
            raise ValueError(f"{name} must be a whole number") from error
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}")
        variable.set(value)
        return value

    def _show_unavailable_state(self) -> None:
        mode = self._selected_mode_key()
        for value_var in self.metric_vars.values():
            value_var.set("—")
        self.update_button.state(["disabled"])
        self.resample_button.state(["disabled"])
        self.axis.clear()
        self.axis.text(
            0.5,
            0.56,
            (
                "Canonical checkpoint is not available"
                if mode == MODE_CANONICAL
                else "RBF-GP checkpoint is not available"
            ),
            ha="center",
            va="center",
            fontsize=16,
            transform=self.axis.transAxes,
        )
        self.axis.text(
            0.5,
            0.45,
            (
                "Train the sanity suites and place outputs/sanity/best_models.pt "
                "to enable predictions."
                if mode == MODE_CANONICAL
                else "Run examples/train_rbf_gp.py and place "
                "outputs/rbf_gp/best_model.pt to enable this mode."
            ),
            ha="center",
            va="center",
            transform=self.axis.transAxes,
        )
        self.axis.set_axis_off()
        self.canvas.draw_idle()
        checkpoint = (
            self.checkpoint_path if mode == MODE_CANONICAL else self.rbf_checkpoint_path
        )
        reason = (
            self.load_error if mode == MODE_CANONICAL else self.rbf_load_error
        ) or "checkpointを読み込めませんでした"
        self.status_var.set(f"Checkpoint: {checkpoint}")
        self.warning_var.set(f"⚠ {reason}")

    def refresh_prediction(self) -> None:
        if self._updating_controls:
            return
        mode = self._selected_mode_key()
        active_runtime = self._active_runtime()
        if active_runtime is None:
            self._show_unavailable_state()
            return
        try:
            context_count = self._read_integer(self.context_var, 3, 20, "Context count")
            context_seed = self._read_integer(self.seed_var, 0, 999_999, "Context seed")
            query_x = float(self.query_var.get())
            x_min, x_max = self._active_x_range()
            if not x_min <= query_x <= x_max:
                raise ValueError(
                    f"Query X must be between {x_min:.4f} and {x_max:.4f}"
                )
            if mode == MODE_CANONICAL:
                suite_key = self._selected_suite_key()
                function_name = self.function_var.get()
                function_seed = None
            else:
                suite_key = None
                function_name = None
                function_seed = self._read_integer(
                    self.function_seed_var, 0, 999_999, "RBF-GP function seed"
                )
        except (tk.TclError, ValueError) as error:
            messagebox.showerror("Invalid input", str(error), parent=self.root)
            return

        self._sync_context_label()
        self._sync_query_label()
        self._sync_seed_label()
        self._sync_function_seed_label()
        self.update_button.state(["disabled"])
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            if mode == MODE_CANONICAL:
                assert self.runtime is not None
                assert suite_key is not None and function_name is not None
                result: PredictionResult = self.runtime.predict(
                    suite_key,
                    function_name,
                    context_count=context_count,
                    query_x=query_x,
                    context_seed=context_seed,
                )
            else:
                assert self.rbf_runtime is not None and function_seed is not None
                result = self.rbf_runtime.predict(
                    function_seed,
                    context_count=context_count,
                    query_x=query_x,
                    context_seed=context_seed,
                )
            self.last_result = result
            draw_prediction(self.axis, result)
            self.canvas.draw_idle()
            for key, value in metric_values(result).items():
                self.metric_vars[key].set(f"{value:.4f}")
            self.status_var.set(status_text(result, self.rbf_runtime))
            if isinstance(result, RBFGPPredictionResult):
                if result.context_is_ood:
                    self.warning_var.set(
                        f"⚠ Context {result.context_count}は学習上限"
                        f" {result.training_max_context}を超えるOOD条件です。"
                    )
                elif self.rbf_runtime.gate_passed is False:
                    self.warning_var.set(
                        "⚠ このRBF-GP checkpointは保存時の固定test gateを"
                        "通過していません。"
                    )
                else:
                    self.warning_var.set(
                        "橙色のGP posteriorは真のkernelを知るoracleです。"
                        "両方の95%区間はpointwise区間です。"
                    )
            elif result.suite.gate_passed is False:
                self.warning_var.set(
                    "⚠ このsuiteは保存時の固定sanity gateを通過していません。"
                )
            else:
                self.warning_var.set(
                    "95%区間は各QueryのGaussian出力に対するpointwise区間です。"
                )
        except Exception as error:
            messagebox.showerror("Prediction failed", str(error), parent=self.root)
        finally:
            self.root.configure(cursor="")
            self.update_button.state(["!disabled"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sanity" / "best_models.pt",
    )
    parser.add_argument(
        "--rbf-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "rbf_gp" / "best_model.pt",
    )
    parser.add_argument("--device", choices=("cpu", "auto", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="construct the UI, exercise all available models, and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = tk.Tk()
    root.withdraw()
    runtime: SanityDemoRuntime | None = None
    rbf_runtime: RBFGPDemoRuntime | None = None
    load_error: str | None = None
    rbf_load_error: str | None = None
    invalid_checkpoint = False
    invalid_rbf_checkpoint = False
    try:
        runtime = SanityDemoRuntime.load(args.checkpoint, device=args.device)
    except FileNotFoundError as error:
        load_error = str(error)
    except Exception as error:
        load_error = str(error)
        invalid_checkpoint = True

    try:
        rbf_runtime = RBFGPDemoRuntime.load(
            args.rbf_checkpoint, device=args.device
        )
    except FileNotFoundError as error:
        rbf_load_error = str(error)
    except Exception as error:
        rbf_load_error = str(error)
        invalid_rbf_checkpoint = True

    try:
        application = SanityDesktopApp(
            root,
            runtime,
            checkpoint_path=args.checkpoint,
            load_error=load_error,
            rbf_runtime=rbf_runtime,
            rbf_checkpoint_path=args.rbf_checkpoint,
            rbf_load_error=rbf_load_error,
        )
    except Exception as error:
        if args.smoke_test:
            print(f"cnp_function_app_error={error}")
        else:
            messagebox.showerror("CNP Function Explorer", str(error), parent=root)
        root.destroy()
        return 1

    if args.smoke_test:
        root.update_idletasks()
        root.update()
        available_modes: list[str] = []
        if runtime is not None:
            application.mode_var.set(MODE_LABELS[MODE_CANONICAL])
            application._apply_mode(refresh=False)
            for suite in runtime.suites:
                application.suite_var.set(SUITE_LABELS[suite.key])
                application._rebuild_functions(refresh=False)
                for function_name in suite.function_names:
                    application.function_var.set(function_name)
                    application.context_var.set(5)
                    application.query_var.set(0.0)
                    application.seed_var.set(2026)
                    application.refresh_prediction()
                    root.update_idletasks()
                    root.update()
                    result = application.last_result
                    if (
                        not isinstance(result, SanityPredictionResult)
                        or result.suite.key != suite.key
                        or result.function_name != function_name
                    ):
                        root.destroy()
                        return 1
            available_modes.append(MODE_CANONICAL)

        rbf_result: RBFGPPredictionResult | None = None
        if rbf_runtime is not None:
            application.mode_var.set(MODE_LABELS[MODE_RBF_GP])
            application._apply_mode(refresh=False)
            application.function_seed_var.set(7001)
            application.context_var.set(5)
            application.query_var.set(0.317)
            application.seed_var.set(2026)
            application.refresh_prediction()
            root.update_idletasks()
            root.update()
            result = application.last_result
            if not isinstance(result, RBFGPPredictionResult):
                root.destroy()
                return 1
            rbf_result = result
            available_modes.append(MODE_RBF_GP)

        if not available_modes:
            print(
                "cnp_function_app_ok checkpoints=unavailable "
                f"canonical={load_error} rbf={rbf_load_error}"
            )
            root.destroy()
            return 1 if invalid_checkpoint or invalid_rbf_checkpoint else 0

        rbf_summary = (
            ""
            if rbf_result is None
            else f" rbf_query={rbf_result.query_x:.4f} "
            f"rbf_prediction={rbf_result.cnp_query_y:.4f}"
        )
        print(
            f"cnp_function_app_ok modes={','.join(available_modes)} "
            f"canonical_suites={0 if runtime is None else len(runtime.suites)}"
            f"{rbf_summary}"
        )
        root.destroy()
        return 0

    root.deiconify()
    root.lift()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
