from __future__ import annotations

import contextlib
import io
import queue
import threading
import traceback
import uuid
from pathlib import Path
from typing import Callable

import tkinter as tk
from tkinter import messagebox, ttk

from . import cli
from .bench.report import TaskMetrics, compute_session_metrics
from .bench.tasks import TASK_REGISTRY

WEB_TRACKERS = ("webgazer", "gazerecorder")


def parse_optional_int(value: str) -> int | None:
    text = value.strip()
    if not text:
        return None
    return int(text)


def parse_optional_text(value: str) -> str | None:
    text = value.strip()
    return text or None


def selected_names(items: dict[str, tk.BooleanVar]) -> list[str]:
    return [name for name, var in items.items() if var.get()]


def fmt_metric(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def read_report_source_and_diag(report_path: Path) -> tuple[str | None, dict | None]:
    if not report_path.exists():
        return None, None
    source = None
    try:
        first_line = report_path.read_text(encoding="utf-8").splitlines()[0].strip()
        if first_line.startswith("CSV:"):
            source = first_line[4:].strip()
    except Exception:
        source = None
    diag = cli._load_report_diag(report_path)
    return source, diag


class QueueWriter(io.TextIOBase):
    def __init__(self, emit: Callable[[str], None]):
        super().__init__()
        self._emit = emit
        self._buffer = ""

    def write(self, text: str | bytes) -> int:
        if not text:
            return 0
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(line.rstrip())
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._emit(self._buffer.rstrip())
            self._buffer = ""


class EyetrackerGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("EyeTracker Control Panel")
        self.geometry("980x660")
        self.minsize(840, 540)

        self._events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._running = False
        self._runner_thread: threading.Thread | None = None
        self._bridge_stop: Callable[[], None] | None = None

        self._build_variables()
        self._build_ui()
        self._set_running(False)

        self.after(120, self._drain_events)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_variables(self) -> None:
        self.tracker_vars = {
            name: tk.BooleanVar(value=(name == "mpiris"))
            for name in cli.DEFAULT_TRACKERS
        }

        self.width_var = tk.StringVar(value=str(cli._SCREEN_W))
        self.height_var = tk.StringVar(value=str(cli._SCREEN_H))
        self.fullscreen_var = tk.BooleanVar(value=False)
        self.session_id_var = tk.StringVar(value="")
        self.start_bridge_var = tk.BooleanVar(value=False)
        self.bridge_ws_port_var = tk.StringVar(value="8000")
        self.bridge_http_port_var = tk.StringVar(value="8001")
        self.bridge_open_browser_var = tk.BooleanVar(value=False)
        self.wait_ready_var = tk.BooleanVar(value=False)

        self.report_runs_var = tk.StringVar(value="runs")
        self.report_session_var = tk.StringVar(value="")
        self.report_tracker_var = tk.StringVar(value="")
        self.report_summary_var = tk.StringVar(value="No data")
        self.model_summary_var = tk.StringVar(
            value="Calibration uses models/<tracker>.json. Tasks and replay prefer models/<tracker>.transfer.json when available."
        )
        self.pair_calib_session_var = tk.StringVar(value="")
        self.pair_task_session_var = tk.StringVar(value="")
        self.pair_per_stim_median_var = tk.BooleanVar(value=True)
        self.camera_source_var = tk.StringVar(value="webcam")
        self.camera_index_var = tk.StringVar(value="1")

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=6)
        top.grid(row=0, column=0, sticky="nsew")
        top.columnconfigure(0, weight=0)
        top.columnconfigure(1, weight=1)
        top.columnconfigure(2, weight=0)

        self._build_trackers_frame(top).grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self._build_settings_frame(top).grid(row=0, column=1, sticky="nsew", padx=4)
        self._build_actions_frame(top).grid(row=0, column=2, sticky="nsew", padx=(4, 0))

        notebook = ttk.Notebook(self)
        notebook.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 4))

        report_tab = ttk.Frame(notebook, padding=8)
        notebook.add(report_tab, text="Report")

        self._build_report_tab(report_tab)

        log_frame = ttk.LabelFrame(self, text="Log", padding=4)
        log_frame.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 6))
        log_frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, wrap="word", height=7)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")

        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set)

        log_actions = ttk.Frame(log_frame)
        log_actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(log_actions, text="Clear Log", command=self._clear_log).pack(side="left")
        self.stop_button = ttk.Button(log_actions, text="Abort Run", command=self._abort_run)
        self.stop_button.pack(side="left", padx=(6, 0))

    def _build_trackers_frame(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Trackers", padding=8)
        for idx, name in enumerate(cli.DEFAULT_TRACKERS):
            ttk.Checkbutton(frame, text=name, variable=self.tracker_vars[name]).grid(
                row=idx, column=0, sticky="w", pady=1,
            )
        n = len(cli.DEFAULT_TRACKERS)
        ttk.Button(frame, text="Select All", command=lambda: self._set_flags(self.tracker_vars, True)).grid(
            row=n, column=0, sticky="ew", pady=(8, 2),
        )
        ttk.Button(frame, text="Clear All", command=lambda: self._set_flags(self.tracker_vars, False)).grid(
            row=n + 1, column=0, sticky="ew",
        )
        return frame

    def _build_settings_frame(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Settings", padding=8)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)

        # Row 0: Camera source + index
        ttk.Label(frame, text="Camera").grid(row=0, column=0, sticky="w", pady=2)
        cam_box = ttk.Combobox(
            frame, textvariable=self.camera_source_var,
            values=["webcam", "daheng"], state="readonly", width=10,
        )
        cam_box.grid(row=0, column=1, sticky="ew", pady=2, padx=(2, 4))
        ttk.Label(frame, text="Index").grid(row=0, column=2, sticky="w", pady=2)
        ttk.Spinbox(
            frame, textvariable=self.camera_index_var,
            from_=0, to=9, width=3,
        ).grid(row=0, column=3, sticky="w", pady=2, padx=(2, 0))

        # Row 1: Session ID (spanning)
        ttk.Label(frame, text="Session ID").grid(row=1, column=0, sticky="w", pady=2)
        session_row = ttk.Frame(frame)
        session_row.grid(row=1, column=1, columnspan=3, sticky="ew", pady=2)
        session_row.columnconfigure(0, weight=1)
        ttk.Entry(session_row, textvariable=self.session_id_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(session_row, text="Generate", command=self._generate_session_id).grid(row=0, column=1, padx=(4, 0))

        return frame

    def _build_actions_frame(self, parent: ttk.Frame) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text="Actions", padding=8)
        frame.columnconfigure(0, weight=1)

        self.calibrate_button = ttk.Button(frame, text="Run Calibration", command=self._start_calibrate)
        self.calibrate_button.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        self.quickstart_button = ttk.Button(frame, text="Quickstart: Calib + Tasks", command=self._start_quickstart)
        self.quickstart_button.grid(row=1, column=0, sticky="ew", pady=4)

        self.tasks_button = ttk.Button(frame, text="Run Tasks Only", command=self._start_tasks)
        self.tasks_button.grid(row=2, column=0, sticky="ew", pady=4)

        return frame

    def _build_report_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        form = ttk.LabelFrame(parent, text="Filters", padding=10)
        form.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        form.columnconfigure(1, weight=1)
        form.columnconfigure(3, weight=1)
        form.columnconfigure(5, weight=1)

        ttk.Label(form, text="Runs Folder").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(form, textvariable=self.report_runs_var).grid(row=0, column=1, sticky="ew", pady=2, padx=(6, 12))
        ttk.Label(form, text="Session ID").grid(row=0, column=2, sticky="w", pady=2)
        ttk.Entry(form, textvariable=self.report_session_var).grid(row=0, column=3, sticky="ew", pady=2, padx=(6, 12))
        ttk.Label(form, text="Tracker").grid(row=0, column=4, sticky="w", pady=2)
        ttk.Entry(form, textvariable=self.report_tracker_var).grid(row=0, column=5, sticky="ew", pady=2, padx=(6, 12))
        self.report_button = ttk.Button(form, text="Build Report", command=self._start_report)
        self.report_button.grid(row=0, column=6, sticky="ew")

        ttk.Label(parent, textvariable=self.report_summary_var).grid(row=1, column=0, sticky="w", pady=(0, 8))

        columns = ("session", "tracker", "task", "samples", "valid", "drop", "mae", "precision")
        self.report_tree = ttk.Treeview(parent, columns=columns, show="headings", height=12)
        self.report_tree.grid(row=2, column=0, sticky="nsew")
        parent.rowconfigure(2, weight=1)

        headings = {
            "session": "Session",
            "tracker": "Tracker",
            "task": "Task",
            "samples": "Samples",
            "valid": "Valid",
            "drop": "Drop %",
            "mae": "MAE px",
            "precision": "Precision px",
        }
        widths = {
            "session": 240,
            "tracker": 110,
            "task": 150,
            "samples": 80,
            "valid": 80,
            "drop": 80,
            "mae": 90,
            "precision": 100,
        }
        for name in columns:
            self.report_tree.heading(name, text=headings[name])
            self.report_tree.column(name, width=widths[name], anchor="center")

        scroll = ttk.Scrollbar(parent, orient="vertical", command=self.report_tree.yview)
        scroll.grid(row=2, column=1, sticky="ns")
        self.report_tree.configure(yscrollcommand=scroll.set)

    def _labeled_entry(self, parent: ttk.Frame, label: str, variable: tk.StringVar, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=2)

    def _set_flags(self, vars_map: dict[str, tk.BooleanVar], value: bool) -> None:
        for var in vars_map.values():
            var.set(value)

    def _generate_session_id(self) -> None:
        self.session_id_var.set(str(uuid.uuid4()))

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _set_running(self, running: bool) -> None:
        self._running = running
        state = "disabled" if running else "normal"
        self.calibrate_button.configure(state=state)
        self.quickstart_button.configure(state=state)
        self.tasks_button.configure(state=state)
        self.report_button.configure(state=state)
        self.stop_button.configure(state=("normal" if running else "disabled"))

    def _selected_trackers(self) -> list[str]:
        trackers = selected_names(self.tracker_vars)
        if not trackers:
            raise ValueError("Select at least one tracker.")
        return trackers

    def _selected_tasks(self) -> list[str]:
        return list(TASK_REGISTRY)

    def _common_kwargs(self) -> dict:
        return {
            "trackers": self._selected_trackers(),
            "fullscreen": True,
            "width_px": cli._SCREEN_W,
            "height_px": cli._SCREEN_H,
            "session_id": parse_optional_text(self.session_id_var.get()),
            "record_video": True,
            "camera_source": self.camera_source_var.get(),
            "camera_index": int(self.camera_index_var.get()),
            "auto_start_ms": None,
            "camera_backend": "auto",
            "all_trackers": False,
        }

    def _calibration_kwargs(self) -> dict:
        trackers = self._selected_trackers()
        has_web = any(name in WEB_TRACKERS for name in trackers)
        return {
            **{
                **self._common_kwargs(),
                "trackers": trackers,
            },
            "dwell_ms": 1500,
            "start_bridge": has_web,
            "bridge_ws_port": int(self.bridge_ws_port_var.get().strip()),
            "bridge_http_port": int(self.bridge_http_port_var.get().strip()),
            "bridge_open_browser": has_web,
            "wait_ready": False,
        }

    def _run_with_logging(
        self,
        label: str,
        func: Callable,
        kwargs: dict,
        *,
        on_success: Callable[[object], None] | None = None,
    ) -> None:
        if self._running:
            messagebox.showwarning("Run Busy", "Wait for the current run to finish first.")
            return

        def worker() -> None:
            writer = QueueWriter(lambda msg: self._events.put(("log", msg)))
            self._events.put(("log", f"[gui] Starting: {label}"))
            try:
                with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                    result = func(**kwargs)
                writer.flush()
                self._events.put(("done", (label, result, on_success, None)))
            except BaseException as exc:
                writer.flush()
                if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                    self._events.put(("log", f"[gui] {label} cancelled"))
                    self._events.put(("done", (label, None, None, None)))
                else:
                    self._events.put(("done", (label, None, on_success, exc)))
                    self._events.put(("log", traceback.format_exc().rstrip()))

        self._set_running(True)
        self._runner_thread = threading.Thread(target=worker, daemon=True)
        self._runner_thread.start()

    def _start_calibrate(self) -> None:
        try:
            kwargs = self._calibration_kwargs()
        except Exception as exc:
            messagebox.showerror("Invalid Parameters", str(exc))
            return

        if self.wait_ready_var.get():
            ok = messagebox.askokcancel(
                "Tracker Readiness",
                "Browser tracker pages will open automatically. Press OK when the trackers are ready to start.",
            )
            if not ok:
                return

        self._run_with_logging("calibrate", cli.calibrate, kwargs)

    def _start_quickstart(self) -> None:
        try:
            kwargs = {
                **self._calibration_kwargs(),
                "tasks": self._selected_tasks(),
            }
        except Exception as exc:
            messagebox.showerror("Invalid Parameters", str(exc))
            return

        if self.wait_ready_var.get():
            ok = messagebox.askokcancel(
                "Tracker Readiness",
                "Browser tracker pages will open automatically. Press OK when the trackers are ready to start.",
            )
            if not ok:
                return

        self._run_with_logging("quickstart", cli.quickstart, kwargs)

    def _start_tasks(self) -> None:
        try:
            trackers = self._selected_trackers()
            has_web = any(name in WEB_TRACKERS for name in trackers)
            kwargs = {
                **self._common_kwargs(),
                "trackers": trackers,
                "tasks": self._selected_tasks(),
                "start_bridge": has_web,
                "bridge_ws_port": int(self.bridge_ws_port_var.get().strip()),
                "bridge_http_port": int(self.bridge_http_port_var.get().strip()),
                "bridge_open_browser": has_web,
                "wait_ready": False,
            }
        except Exception as exc:
            messagebox.showerror("Invalid Parameters", str(exc))
            return
        if self.wait_ready_var.get():
            ok = messagebox.askokcancel(
                "Tracker Readiness",
                "Browser tracker pages will open automatically. Press OK when the trackers are ready to start.",
            )
            if not ok:
                return
        self._run_with_logging("run-tasks", cli.run_tasks, kwargs)

    def _start_report(self) -> None:
        if self._running:
            messagebox.showwarning("Run Busy", "Wait for the current run to finish first.")
            return

        def build_report(runs: Path, session: str | None, tracker: str | None) -> list[TaskMetrics]:
            sessions = cli._list_sessions(runs, session)
            rows: list[TaskMetrics] = []
            for sess in sessions:
                rows.extend(compute_session_metrics(sess))
            if tracker:
                rows = [row for row in rows if row.tracker_id == tracker]
            return rows

        try:
            runs = Path(self.report_runs_var.get().strip() or "runs")
            session = parse_optional_text(self.report_session_var.get())
            tracker = parse_optional_text(self.report_tracker_var.get())
        except Exception as exc:
            messagebox.showerror("Invalid Parameters", str(exc))
            return

        self._run_with_logging(
            "report",
            build_report,
            {"runs": runs, "session": session, "tracker": tracker},
            on_success=self._populate_report,
        )

    def _populate_report(self, result: object) -> None:
        rows = list(result or [])
        for item in self.report_tree.get_children():
            self.report_tree.delete(item)

        rows.sort(key=lambda row: (row.session_id, row.tracker_id, row.task_name))
        for row in rows:
            self.report_tree.insert(
                "",
                "end",
                values=(
                    row.session_id,
                    row.tracker_id,
                    row.task_name,
                    row.samples,
                    row.valid_samples,
                    fmt_metric(row.drop_rate * 100.0),
                    fmt_metric(row.mae_px),
                    fmt_metric(row.precision_px),
                ),
            )

        if rows:
            sessions = len({row.session_id for row in rows})
            self.report_summary_var.set(f"Rows: {len(rows)} | Sessions: {sessions}")
            self._append_log(f"[gui] Report built: {len(rows)} rows")
        else:
            self.report_summary_var.set("No data matched the current filter")
            self._append_log("[gui] Report is empty")

    def _refresh_models_view(self) -> None:
        pass

    def _stop_bridge(self) -> None:
        if self._bridge_stop is None:
            return
        try:
            self._bridge_stop()
        finally:
            self._bridge_stop = None
            self._append_log("[gui] Bridge stopped")

    def _abort_run(self) -> None:
        messagebox.showinfo(
            "Abort Run",
            "Safe forced termination is not implemented for pygame or camera workflows. "
            "Close the stimulus window or wait for the current process to finish.",
        )

    def _drain_events(self) -> None:
        try:
            while True:
                try:
                    kind, payload = self._events.get_nowait()
                except queue.Empty:
                    break

                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "done":
                    label, result, on_success, error = payload  # type: ignore[misc]
                    try:
                        if error is None:
                            self._append_log(f"[gui] Finished: {label}")
                            if on_success is not None:
                                on_success(result)
                            elif label in {"calibrate", "quickstart"}:
                                self._refresh_models_view()
                        else:
                            self._append_log(f"[gui] Error in {label}: {error}")
                            messagebox.showerror("Run Error", f"{label}: {error}")
                    finally:
                        self._set_running(False)
        except Exception:
            self._set_running(False)
        finally:
            self.after(120, self._drain_events)

    def _on_close(self) -> None:
        if self._running:
            ok = messagebox.askyesno(
                "Close Window",
                "A workflow is still running. Closing the window may leave camera or browser resources open. Exit anyway?",
            )
            if not ok:
                return
        self._stop_bridge()
        self.destroy()


def main() -> None:
    app = EyetrackerGui()
    app.mainloop()


if __name__ == "__main__":
    main()
