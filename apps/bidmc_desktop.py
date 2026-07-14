"""Native macOS desktop application for querying the BIDMC heart-rate CNP."""

from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

# In this conda environment, loading TkAgg before PyTorch also avoids the two
# libraries racing to initialize incompatible OpenMP runtimes.
import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from cnp.bidmc_demo import BIDMCDemoRuntime, PredictionResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_LABELS = {
    "test": "Test（学習に未使用の患者）",
    "validation": "Validation（モデル選択用の患者）",
    "train": "Train（学習に使用した患者）",
}
SPLIT_KEYS = {label: key for key, label in SPLIT_LABELS.items()}

METRICS = (
    ("cnp_mean", "CNP Prediction"),
    ("cnp_scale", "CNP SD"),
    ("lower", "95% lower"),
    ("upper", "95% upper"),
    ("truth", "Ground truth"),
    ("error", "Absolute error"),
    ("linear", "Linear baseline"),
)


def metric_values(result: PredictionResult) -> dict[str, float]:
    """Return the seven numeric cards shown below the plot."""

    return {
        "cnp_mean": result.cnp_query_bpm,
        "cnp_scale": result.cnp_query_scale_bpm,
        "lower": result.lower_95_bpm,
        "upper": result.upper_95_bpm,
        "truth": result.truth_query_bpm,
        "error": result.absolute_error_bpm,
        "linear": result.linear_query_bpm,
    }


def status_text(result: PredictionResult, checkpoint_step: int) -> str:
    """Describe exactly which points were model input and which were hidden."""

    return (
        f"Split: {SPLIT_LABELS[result.entry.split]}\n"
        f"Input: {len(result.context_indices)} context points / "
        f"hidden {result.gap_start}–{result.gap_stop - 1} s "
        f"({result.gap_width} points)\n"
        f"Query: X = {result.query_index} s  →  "
        f"CNP Prediction Y = {result.cnp_query_bpm:.2f} bpm\n"
        f"Checkpoint: step {checkpoint_step}. "
        "正解HRは評価表示にだけ使い、CNPの入力には渡していません。"
    )


def draw_prediction(axis, result: PredictionResult) -> None:
    """Draw one prediction on an existing Matplotlib axis."""

    seconds = result.seconds.numpy()
    truth = result.truth_bpm.numpy()
    mean = result.cnp_mean_bpm.numpy()
    scale = result.cnp_scale_bpm.numpy()
    linear = result.linear_bpm.numpy()
    context = result.context_indices.numpy()

    axis.clear()
    axis.axvspan(
        result.gap_start - 0.5,
        result.gap_stop - 0.5,
        color="#ef4444",
        alpha=0.09,
        label="Hidden gap",
    )
    axis.plot(
        seconds,
        truth,
        color="#111827",
        linewidth=1.5,
        alpha=0.58,
        label="Ground truth (display only)",
    )
    axis.scatter(
        seconds[context],
        truth[context],
        s=15,
        color="#15803d",
        alpha=0.82,
        label="Context given to CNP",
        zorder=4,
    )
    axis.fill_between(
        seconds,
        mean - 1.96 * scale,
        mean + 1.96 * scale,
        color="#2563eb",
        alpha=0.16,
        label="CNP pointwise 95% interval",
    )
    axis.plot(seconds, mean, color="#1d4ed8", linewidth=2.2, label="CNP mean")
    axis.plot(
        seconds,
        linear,
        color="#ea580c",
        linewidth=1.8,
        linestyle="--",
        label="Linear interpolation",
    )
    axis.axvline(
        result.query_index,
        color="#a21caf",
        linewidth=1.4,
        linestyle=":",
        label=f"Query: {result.query_index} s",
    )
    axis.scatter(
        [result.query_index],
        [result.cnp_query_bpm],
        s=68,
        color="#a21caf",
        edgecolor="white",
        linewidth=1.2,
        zorder=6,
    )
    axis.set_xlim(0, len(seconds) - 1)
    axis.set_xlabel("Seconds within the selected window")
    axis.set_ylabel("Heart rate (bpm)")
    axis.set_title(
        f"{result.entry.window.record_id} / {result.entry.window.patient_id} / "
        f"window {result.entry.window.window_index + 1}"
    )
    axis.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.55)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=4,
        frameon=False,
        fontsize=8,
    )


class BIDMCDesktopApp:
    """Tkinter application that owns controls and an embedded Matplotlib canvas."""

    def __init__(self, root: tk.Tk, runtime: BIDMCDemoRuntime) -> None:
        self.root = root
        self.runtime = runtime
        self.last_result: PredictionResult | None = None
        self._updating_controls = False
        self._window_keys: dict[str, str] = {}

        self._configure_window()
        self._configure_styles()
        self._build_menu()
        self._build_layout()
        self._rebuild_from_split(refresh=False)
        self.refresh_prediction()

    def _configure_window(self) -> None:
        self.root.title("BIDMC CNP Query Explorer")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        width = min(1500, screen_width - 80)
        height = min(940, screen_height - 100)
        x = max(20, (screen_width - width) // 2)
        y = max(30, (screen_height - height) // 3)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(1080, 720)
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
        menu.add_cascade(label="BIDMC CNP", menu=app_menu)
        self.root.configure(menu=menu)

    def _show_about(self) -> None:
        messagebox.showinfo(
            "BIDMC CNP Query Explorer",
            "保存済みCNPでBIDMC心拍数の欠損補間を確認するネイティブアプリです。\n\n"
            "正解HRは評価表示にのみ使用し、モデル入力には渡しません。",
            parent=self.root,
        )

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=16)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        ttk.Label(outer, text="BIDMC CNP Query Explorer", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            outer,
            text=(
                f"{self.runtime.point_count}秒の心拍系列から欠損区間を作り、"
                "任意のQuery時刻 X に対する心拍数 Y を推定します。"
            ),
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
        self.split_var = tk.StringVar(value=SPLIT_LABELS["test"])
        self.split_combo = self._add_combo(
            controls,
            row,
            "Dataset split",
            self.split_var,
            tuple(SPLIT_KEYS),
            self._on_split_changed,
        )
        row += 2

        self.patient_var = tk.StringVar()
        self.patient_combo = self._add_combo(
            controls, row, "Patient", self.patient_var, (), self._on_patient_changed
        )
        row += 2

        self.record_var = tk.StringVar()
        self.record_combo = self._add_combo(
            controls, row, "Record", self.record_var, (), self._on_record_changed
        )
        row += 2

        self.window_var = tk.StringVar()
        self.window_combo = self._add_combo(
            controls,
            row,
            f"{self.runtime.point_count}-second window",
            self.window_var,
            (),
            self._on_window_changed,
        )
        row += 2

        query_min, query_max = self.runtime.query_bounds
        self.query_var = tk.IntVar(value=min(max(60, query_min), query_max))
        self.query_value_var = tk.StringVar(value=f"{self.query_var.get()} s")
        row = self._add_integer_control(
            controls,
            row,
            "Query time X",
            self.query_var,
            self.query_value_var,
            query_min,
            query_max,
        )

        gap_max = min(60, self.runtime.point_count - 2 * self.runtime.margin)
        self.gap_var = tk.IntVar(value=min(30, gap_max))
        self.gap_value_var = tk.StringVar(value=f"{self.gap_var.get()} s")
        row = self._add_integer_control(
            controls,
            row,
            "Hidden gap width",
            self.gap_var,
            self.gap_value_var,
            1,
            gap_max,
        )

        self.update_button = ttk.Button(
            controls,
            text="Predictionを更新",
            command=self.refresh_prediction,
            default="active",
        )
        self.update_button.grid(row=row, column=0, sticky="ew", pady=(12, 14))
        row += 1

        ttk.Separator(controls).grid(row=row, column=0, sticky="ew", pady=(0, 12))
        row += 1
        ttk.Label(controls, text="読み方", style="Section.TLabel").grid(
            row=row, column=0, sticky="w"
        )
        row += 1
        ttk.Label(
            controls,
            text=(
                "緑点: CNPへ渡したContext\n"
                "赤い帯: 人工的に隠した区間\n"
                "紫の縦線: 選んだQuery X\n"
                "青線/青帯: CNP平均と95%区間\n"
                "灰線: 正解（比較表示のみ）"
            ),
            justify="left",
        ).grid(row=row, column=0, sticky="w")

        plot_panel = ttk.Frame(workspace)
        plot_panel.grid(row=0, column=1, sticky="nsew")
        plot_panel.rowconfigure(0, weight=1)
        plot_panel.columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(10.8, 5.9), dpi=100, layout="constrained")
        self.axis = self.figure.subplots()
        self.axis.text(
            0.5,
            0.5,
            "Loading prediction…",
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
            ttk.Label(card, text=f"{label} (bpm)", style="MetricName.TLabel").grid(
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
        self.status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            style="Status.TLabel",
            justify="left",
        )
        self.status_label.grid(row=0, column=0, sticky="w")
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
            command=lambda _value: self._sync_integer_labels(),
        )
        scale.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        spin = ttk.Spinbox(
            input_row,
            from_=minimum,
            to=maximum,
            textvariable=variable,
            width=5,
            command=self._refresh_from_controls,
        )
        spin.grid(row=0, column=1, sticky="e")
        scale.bind("<ButtonRelease-1>", lambda _event: self.refresh_prediction())
        scale.bind("<KeyRelease>", lambda _event: self.refresh_prediction())
        # Return bubbles to the root binding below; a second widget binding
        # would run the same prediction twice.
        spin.bind("<FocusOut>", lambda _event: self._refresh_from_controls())
        return row + 2

    def _sync_integer_labels(self) -> None:
        try:
            self.query_value_var.set(f"{int(self.query_var.get())} s")
            self.gap_value_var.set(f"{int(self.gap_var.get())} s")
        except (AttributeError, tk.TclError):
            pass

    def _refresh_from_controls(self) -> None:
        self._sync_integer_labels()
        self.refresh_prediction()

    @staticmethod
    def _set_combo_values(
        combo: ttk.Combobox,
        variable: tk.StringVar,
        values: tuple[str, ...],
    ) -> None:
        combo.configure(values=values)
        if not values:
            variable.set("")
        elif variable.get() not in values:
            variable.set(values[0])

    def _selected_split(self) -> str:
        try:
            return SPLIT_KEYS[self.split_var.get()]
        except KeyError as error:
            raise ValueError(f"Unknown split selection: {self.split_var.get()!r}") from error

    def _split_entries(self):
        return self.runtime.entries_for_split(self._selected_split())

    def _rebuild_from_split(self, *, refresh: bool = True) -> None:
        self._updating_controls = True
        try:
            patient_ids = tuple(
                sorted({entry.window.patient_id for entry in self._split_entries()})
            )
            self._set_combo_values(self.patient_combo, self.patient_var, patient_ids)
            self._rebuild_records()
        finally:
            self._updating_controls = False
        if refresh:
            self.refresh_prediction()

    def _rebuild_records(self) -> None:
        records = tuple(
            sorted(
                {
                    entry.window.record_id
                    for entry in self._split_entries()
                    if entry.window.patient_id == self.patient_var.get()
                }
            )
        )
        self._set_combo_values(self.record_combo, self.record_var, records)
        self._rebuild_windows()

    def _rebuild_windows(self) -> None:
        entries = tuple(
            sorted(
                (
                    entry
                    for entry in self._split_entries()
                    if entry.window.patient_id == self.patient_var.get()
                    and entry.window.record_id == self.record_var.get()
                ),
                key=lambda entry: entry.window.window_index,
            )
        )
        self._window_keys = {
            f"Window {entry.window.window_index + 1} "
            f"(local 0–{self.runtime.point_count - 1} s)": entry.key
            for entry in entries
        }
        self._set_combo_values(
            self.window_combo,
            self.window_var,
            tuple(self._window_keys),
        )

    def _on_split_changed(self, _event=None) -> None:
        if not self._updating_controls:
            self._rebuild_from_split()

    def _on_patient_changed(self, _event=None) -> None:
        if self._updating_controls:
            return
        self._updating_controls = True
        try:
            self._rebuild_records()
        finally:
            self._updating_controls = False
        self.refresh_prediction()

    def _on_record_changed(self, _event=None) -> None:
        if self._updating_controls:
            return
        self._updating_controls = True
        try:
            self._rebuild_windows()
        finally:
            self._updating_controls = False
        self.refresh_prediction()

    def _on_window_changed(self, _event=None) -> None:
        if not self._updating_controls:
            self.refresh_prediction()

    def _read_integer(
        self,
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

    def refresh_prediction(self) -> None:
        if self._updating_controls or not self._window_keys:
            return
        try:
            window_key = self._window_keys[self.window_var.get()]
            query_min, query_max = self.runtime.query_bounds
            query = self._read_integer(
                self.query_var, query_min, query_max, "Query time"
            )
            gap_max = min(60, self.runtime.point_count - 2 * self.runtime.margin)
            gap = self._read_integer(self.gap_var, 1, gap_max, "Gap width")
        except (KeyError, ValueError) as error:
            messagebox.showerror("Invalid input", str(error), parent=self.root)
            return

        self._sync_integer_labels()
        self.update_button.state(["disabled"])
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            result = self.runtime.predict(window_key, query, gap)
            self.last_result = result
            draw_prediction(self.axis, result)
            self.canvas.draw_idle()
            for key, value in metric_values(result).items():
                self.metric_vars[key].set(f"{value:.2f}")
            self.status_var.set(status_text(result, self.runtime.checkpoint_step))
            if result.gap_width > 30:
                self.warning_var.set(
                    "⚠ このモデルが学習時に見た連続欠損は最大30秒です。"
                    "30秒を超える設定は学習範囲外です。"
                )
            else:
                self.warning_var.set(
                    "95%区間は各時刻のGaussian出力に対するpointwise区間です。"
                )
        except Exception as error:
            messagebox.showerror("Prediction failed", str(error), parent=self.root)
        finally:
            self.root.configure(cursor="")
            self.update_button.state(["!disabled"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "bidmc")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "bidmc" / "best_model.pt",
    )
    parser.add_argument("--device", choices=("cpu", "auto", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="construct the native UI, run two predictions, and exit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = tk.Tk()
    root.withdraw()
    try:
        runtime = BIDMCDemoRuntime.load(
            args.data_dir,
            args.checkpoint,
            device=args.device,
        )
        application = BIDMCDesktopApp(root, runtime)
    except Exception as error:
        if args.smoke_test:
            print(f"native_app_error={error}")
        else:
            messagebox.showerror("BIDMC CNP Query Explorer", str(error), parent=root)
        root.destroy()
        return 1

    if args.smoke_test:
        application.query_var.set(80)
        application.gap_var.set(15)
        application.refresh_prediction()
        root.update_idletasks()
        root.update()
        result = application.last_result
        if result is None:
            root.destroy()
            return 1
        print(
            f"native_app_ok query={result.query_index} gap={result.gap_width} "
            f"prediction={result.cnp_query_bpm:.2f}"
        )
        root.destroy()
        return 0

    root.deiconify()
    root.lift()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
