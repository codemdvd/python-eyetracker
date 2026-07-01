# src/eyetrk/orchestrator.py
from __future__ import annotations

import time
import threading
from typing import Dict, List
from collections import defaultdict

from .core.types import Sample
from .io.logger import RunLogger
from .io.schema import SessionMeta
from .calib.protocols import generate_9pt_grid, generate_25pt_grid
from .calib.acceptance import acceptance
from .core.tracker import Tracker
from .bench.tasks import Task
from .bench.postprocess import BenchmarkAnchorRecalibrator, BenchmarkBiasCorrector


class Orchestrator:
    """Central coordinator: drives the stimulus engine, routes events to tracker adapters, buffers samples, and triggers model fitting."""

    def __init__(
        self,
        adapters: Dict[str, Tracker],
        logger: RunLogger,
        session_id: str,
        session_meta: SessionMeta | None = None,
    ):
        self.adapters = adapters
        self.logger = logger
        self.session_id = session_id
        self.session_meta = session_meta
        self._active_stim_id: str | None = None
        self._active_task_name: str | None = None
        self._active_target_px: tuple[float, float] | None = None
        self._buffers: Dict[str, Dict[str, List[Sample]]] = defaultdict(lambda: defaultdict(list))
        self._runtime_dims: tuple[int, int] | None = None
        self._stim_windows: Dict[str, tuple[int, int]] = {}
        self._stim_targets: Dict[str, tuple[float, float]] = {}
        self._stim_t_on: Dict[str, int] = {}
        self._state_lock = threading.RLock()
        self._calib_settle_ms = 200  # drop early saccade
        self._calib_tail_ms = 80    # trim noisy tail
        self._task_postprocessors: Dict[str, BenchmarkBiasCorrector] = {}
        self._task_recalibrators: Dict[str, BenchmarkAnchorRecalibrator] = {}
        self._web_ready_counts: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------ #

    def _on_sample(self, sample: Sample):
        """Callback for every incoming gaze sample: stamps task/stim context, applies postprocessors, buffers, and logs."""
        with self._state_lock:
            if not sample.session_id:
                sample.session_id = self.session_id

            ts = getattr(sample, "timestamp_ms", None)
            if ts and sample.tracker_id in ("webgazer", "gazerecorder"):
                # Retroactive stim assignment: GR sends predictions via cloud with 50–500ms
                # latency, so by the time a prediction arrives the browser has moved to the
                # next stim. Use wall-clock timestamp to find the correct completed stim window.
                retroactive = False
                for sid, (t_on, t_off) in self._stim_windows.items():
                    if t_on <= ts <= t_off:
                        sample.stim_id = sid
                        if sid in self._stim_targets:
                            sample.target_x_px, sample.target_y_px = self._stim_targets[sid]
                        retroactive = True
                        break
                if not retroactive:
                    if not sample.stim_id:
                        sample.stim_id = self._active_stim_id
                    if self._active_target_px:
                        sample.target_x_px, sample.target_y_px = self._active_target_px
            else:
                if self._active_stim_id and not sample.stim_id:
                    sample.stim_id = self._active_stim_id
                if self._active_target_px:
                    sample.target_x_px, sample.target_y_px = self._active_target_px
            sample.event = sample.event or "stream"

            if self._active_task_name and not sample.task_name:
                sample.task_name = self._active_task_name

            if sample.task_name in {"fixation-grid", "step-saccades"}:
                post = self._task_postprocessors.setdefault(sample.tracker_id, BenchmarkBiasCorrector())
                sample = post.apply(sample)
            if sample.task_name in {"fixation-grid", "step-saccades", "smooth-pursuit"}:
                recal = self._task_recalibrators.setdefault(sample.tracker_id, BenchmarkAnchorRecalibrator())
                sample = recal.apply(sample)

            if sample.stim_id and sample.tracker_id:
                self._buffers[sample.tracker_id][sample.stim_id].append(sample)
            if sample.tracker_id in {"webgazer", "gazerecorder"}:
                self._web_ready_counts[sample.tracker_id] += 1

        self.logger.write_sample(sample)

    def start_streams(self):
        """Start all adapter data streams, registering _on_sample as the shared callback."""
        for a in self.adapters.values():
            a.start_stream(self._on_sample, self.session_id)

    def stop_streams(self):
        """Stop all adapter streams, silencing errors so a failed stop never aborts teardown."""
        for a in self.adapters.values():
            try:
                a.stop()
            except Exception:
                pass

    def _broadcast(self, event: str, payload: dict | None = None):
        """Send an event to every registered adapter's on_event handler."""
        for a in self.adapters.values():
            a.on_event(event, payload or {})

    def _record_event(self, event: str, payload: dict | None = None) -> None:
        """Write one event to timeline.jsonl, injecting t_ms and current task_name if absent."""
        data = dict(payload or {})
        data.setdefault("t_ms", int(time.time() * 1000))
        if "task_name" not in data:
            with self._state_lock:
                task_name = self._active_task_name
            if task_name is not None:
                data["task_name"] = task_name
        try:
            self.logger.write_event(event, data)
        except Exception:
            pass

    def _reset_web_ready_counts(self) -> None:
        """Zero the per-tracker sample counters used to detect when browser trackers have started streaming."""
        with self._state_lock:
            for name in ("webgazer", "gazerecorder"):
                self._web_ready_counts[name] = 0

    # ------------------------------------------------------------------ #

    def calibrate_all(
        self,
        dwell_ms=1000,
        gap_ms=300,
        fullscreen=False,
        camera_index: int | None = 0,
        mirror_preview: bool = True,
        cam_unmirror: bool = True,
        framing_scale: float = 0.82,
        manage_streams: bool = False,
        auto_start_ms: int | None = None,
        show_ready: bool = True,
        camera_source: str = "webcam",
    ):
        """Run 9-point calibration sequence for all trackers."""
        from .stim.engine import StimEngine, StimConfig

        meta_w, meta_h = self._screen_dims()
        eng = StimEngine(StimConfig(width=meta_w, height=meta_h, fullscreen=fullscreen))
        w, h = eng.size
        with self._state_lock:
            self._runtime_dims = (w, h)
        seq = generate_9pt_grid(dwell_ms, gap_ms).points
        repeated = []
        with self._state_lock:
            self._stim_windows = {}
            self._stim_targets = {}
            self._stim_t_on = {}
        started_streams = False

        try:
            # Show the "press Space" ready screen only when there is a native camera to preview.
            # Web-only phases skip it; _wait_for_web_ready handles the automatic wait below.
            if show_ready and (camera_index is not None or camera_source == "daheng"):
                ready_msg = (
                    "Daheng camera — fit your face in the frame, then press Space to start calibration"
                    if camera_source == "daheng"
                    else "Fit your face in the frame, then press Space to start calibration"
                )
                eng.wait_for_ready(
                    ready_msg,
                    cam_index=camera_index,
                    mirror_preview=mirror_preview,
                    unmirror_cam=cam_unmirror,
                    frame_scale=framing_scale,
                    auto_start_ms=auto_start_ms,
                    show_alignment_overlay=True,
                    camera_source=camera_source,
                )
            if manage_streams:
                self.start_streams()
                started_streams = True
            start_payload = {"screen_w": w, "screen_h": h, "t_ms": int(time.time() * 1000)}
            self._broadcast("calibration_start", start_payload)
            self._record_event("calibration_start", start_payload)
            web_like = any(name in ("webgazer", "gazerecorder") for name in self.adapters)
            if web_like:
                if "gazerecorder" in self.adapters:
                    # Force the browser to reload the bridge page so stale cached HTML
                    # is never used. The local HTTP server serves with Cache-Control:no-store,
                    # so the reload always fetches the latest gazerecorder_bridge.html.
                    try:
                        from eyetrk.web_bridge import server as _bridge_srv
                        _bridge_srv.push_event("gazerecorder", {"type": "reload"})
                    except Exception:
                        pass
                    time.sleep(3)  # allow browser reload + WS reconnect before sending start
                self._reset_web_ready_counts()
                self._broadcast(
                    "bridge_start",
                    {
                        "t": int(time.time() * 1000),
                        "screen_w": w,
                        "screen_h": h,
                        "phase": "calibration",
                    },
                )
                # GazeRecorder SDK shows a calibration wizard when it starts.
                # Always wait for OnCalibrationComplete (or first-gaze heuristic) before
                # showing our stims. Timeout=90s for internal-calib mode (user must complete
                # the wizard manually); 30s for external-calib (soft-reset path is instant;
                # fresh-tab needs time for the automatic wizard). WebGazer needs 5s min.
                gr_present = "gazerecorder" in self.adapters
                gr_internal = gr_present and any(
                    name == "gazerecorder" and getattr(self.adapters.get(name), "uses_internal_calibration", False)
                    for name in self.adapters
                )
                wg_present = "webgazer" in self.adapters
                # GR bridge shows a click-to-fullscreen gate before starting the SDK,
                # then calls ShowCalibration() if the server returns a cached model.
                # Total wait: user click (up to 30s) + wizard (~60s) + 110s safety net.
                # Give Python 180s so the 110s bridge timeout always fires first.
                wait_timeout = 180_000 if gr_present else 12_000
                min_wait = 2500 if gr_present else (5000 if wg_present else 2500)
                self._wait_for_web_ready(eng, timeout_ms=wait_timeout, min_wait_ms=min_wait)
            else:
                eng.show_message("", duration_ms=3000, bg=(255, 255, 255))  # brief blank screen for camera warmup
            _cam_set = {"mpiris", "optimeyes"}
            _web_only = not any(n in _cam_set for n in self.adapters)
            _total_pts = len(seq)
            with self._state_lock:
                self._active_task_name = "calibration"
            for _pt_idx, pt in enumerate(seq, 1):
                for attempt in (1, 2, 3):
                    stim_id = pt.id if attempt == 1 else f"{pt.id}_retry"
                    with self._state_lock:
                        self._active_stim_id = stim_id
                        for tr in self.adapters:
                            self._buffers[tr][stim_id] = []

                    if _web_only:
                        eng.show_message(
                            f"Browser calibration  {_pt_idx}/{_total_pts}",
                            subtitle="Look at the red dot in the browser window",
                            bg=(255, 255, 255),
                            fg=(40, 40, 40),
                            sub_fg=(100, 100, 100),
                        )
                    else:
                        eng.show_point(pt.x_norm, pt.y_norm)
                    self._set_active_target_norm(pt.x_norm, pt.y_norm, dims=(w, h))
                    t_on = int(time.time() * 1000)
                    tx_px = pt.x_norm * w
                    ty_px = pt.y_norm * h
                    with self._state_lock:
                        self._stim_targets[stim_id] = (tx_px, ty_px)
                        self._stim_t_on[stim_id] = t_on
                    self._broadcast(
                        "stim",
                        {
                            "stim_id": stim_id,
                            "x_norm": pt.x_norm,
                            "y_norm": pt.y_norm,
                            "target_x_px": tx_px,
                            "target_y_px": ty_px,
                            "t": t_on,
                        },
                    )
                    self._record_event(
                        "stim",
                        {
                            "stim_id": stim_id,
                            "task_name": "calibration",
                            "x_norm": pt.x_norm,
                            "y_norm": pt.y_norm,
                            "target_x_px": tx_px,
                            "target_y_px": ty_px,
                            "t_ms": t_on,
                        },
                    )
                    self._broadcast(
                        "calib_point_start",
                        {"id": stim_id, "x_norm": pt.x_norm, "y_norm": pt.y_norm, "t": t_on},
                    )

                    t_end = t_on + pt.dwell_ms
                    while int(time.time() * 1000) < t_end:
                        eng.tick()

                    if not _web_only:
                        eng.hide()
                    self._clear_active_target()
                    t_off = int(time.time() * 1000)
                    with self._state_lock:
                        self._stim_windows[stim_id] = (t_on, t_off)
                        self._stim_t_on.pop(stim_id, None)
                    self._broadcast("calib_point_end", {"id": stim_id, "t": t_off})
                    self._record_event("stim_off", {"stim_id": stim_id, "task_name": "calibration", "t_ms": t_off})

                    base_id = stim_id.replace("_retry", "")
                    fails = []
                    for tr_name in self.adapters:
                        adapter = self.adapters.get(tr_name)
                        target_xy_px = None
                        # Only apply offset check for gazerecorder (persistent cloud calibration).
                        # webgazer starts uncalibrated each session, so offset checks cause
                        # excessive retries that make calibration look broken to the participant.
                        if tr_name == "gazerecorder" and adapter is not None and getattr(adapter, "uses_internal_calibration", False):
                            target_xy_px = (tx_px, ty_px)
                        samples = self._calib_slice(stim_id, self._get_stim_samples(tr_name, stim_id))
                        ok = acceptance(
                            samples,
                            screen_w_px=w,
                            screen_h_px=h,
                            target_xy_px=target_xy_px,
                            max_offset_px=60.0,
                            max_disp_px=40.0,
                            )
                        if not ok:
                            fails.append(tr_name)

                    if fails and attempt < 3:
                        repeated.append(pt.id)
                        gap_until = int(time.time() * 1000) + pt.gap_ms
                        while int(time.time() * 1000) < gap_until:
                            eng.tick()
                        continue

                    break

                gap_until = int(time.time() * 1000) + pt.gap_ms
                while int(time.time() * 1000) < gap_until:
                    eng.tick()

            done_payload = {"t_ms": int(time.time() * 1000)}
            self._broadcast("calibration_done", done_payload)
            self._record_event("calibration_done", done_payload)
            self._fit_and_apply_models()

        finally:
            if started_streams:
                self.stop_streams()
            eng.close()
            with self._state_lock:
                self._active_stim_id = None
                self._active_task_name = None
                self._active_target_px = None
                self._runtime_dims = None
                self._stim_windows = {}

    # ------------------------------------------------------------------ #

    def run_tasks(
        self,
        tasks: List[Task],
        fullscreen: bool = False,
        camera_index: int | None = 0,
        mirror_preview: bool = True,
        cam_unmirror: bool = True,
        framing_scale: float = 0.82,
        auto_start_ms: int | None = None,
        show_ready: bool = True,
        show_head_prompt: bool = True,
        camera_source: str = "webcam",
        manage_streams: bool = False,
    ):
        """Play benchmark tasks timeline on the stimulus engine."""
        from .stim.engine import StimEngine, StimConfig

        meta_w, meta_h = self._screen_dims()
        eng = StimEngine(StimConfig(width=meta_w, height=meta_h, fullscreen=fullscreen))
        w, h = eng.size
        with self._state_lock:
            self._runtime_dims = (w, h)

        started_streams = False
        try:
            if show_ready and (camera_index is not None or camera_source == "daheng"):
                ready_msg = (
                    "Daheng camera — fit your face in the frame, then press Space to start tasks"
                    if camera_source == "daheng"
                    else "Fit your face in the frame, then press Space to start tasks"
                )
                eng.wait_for_ready(
                    ready_msg,
                    cam_index=camera_index,
                    mirror_preview=mirror_preview,
                    unmirror_cam=cam_unmirror,
                    frame_scale=framing_scale,
                    auto_start_ms=auto_start_ms,
                    show_alignment_overlay=True,
                    camera_source=camera_source,
                )
            if manage_streams:
                self.start_streams()
                started_streams = True
            tasks_payload = {"screen_w": w, "screen_h": h, "t_ms": int(time.time() * 1000)}
            self._broadcast("tasks_start", tasks_payload)
            self._record_event("tasks_start", tasks_payload)
            web_like = any(name in ("webgazer", "gazerecorder") for name in self.adapters)
            if web_like:
                self._reset_web_ready_counts()
                self._broadcast(
                    "bridge_start",
                    {
                        "t": int(time.time() * 1000),
                        "screen_w": w,
                        "screen_h": h,
                        "phase": "tasks",
                    },
                )
                wg_tasks = "webgazer" in self.adapters
                self._wait_for_web_ready(eng, timeout_ms=12000, min_wait_ms=5000 if wg_tasks else 2500)
            else:
                eng.show_message("", duration_ms=3000, bg=(255, 255, 255))  # brief blank screen for camera warmup

            _tasks_web_only = not any(n in {"mpiris", "optimeyes"} for n in self.adapters)
            if _tasks_web_only:
                eng.show_message(
                    "Browser tasks in progress",
                    subtitle="Follow the blue dot in the browser window",
                    bg=(255, 255, 255),
                    fg=(40, 40, 40),
                    sub_fg=(100, 100, 100),
                )
            for idx, task in enumerate(tasks):  # noqa: B007
                if idx == 1 and show_head_prompt:
                    if camera_source == "daheng":
                        prompt_msg = "Rotate head left/right/up/down, then press Space to continue"
                    elif camera_index is None:
                        prompt_msg = (
                            "If needed, adjust your position in the browser tracker page,\n"
                            "then press Space to continue"
                        )
                    else:
                        prompt_msg = "Rotate head left/right/up/down, then press Space to continue"
                    eng.wait_for_ready(
                        prompt_msg,
                        cam_index=camera_index,
                        mirror_preview=mirror_preview,
                        unmirror_cam=cam_unmirror,
                        frame_scale=framing_scale,
                        auto_start_ms=auto_start_ms,
                        show_alignment_overlay=camera_index is not None or camera_source == "daheng",
                        camera_source=camera_source,
                    )
                with self._state_lock:
                    self._active_task_name = task.name
                task_start_payload = {"task": task.name, "t_ms": int(time.time() * 1000)}
                self._broadcast("task_start", task_start_payload)
                self._record_event("task_start", task_start_payload)
                for event in task.timeline:
                    self._handle_task_event(event, eng, dims=(w, h), web_only=_tasks_web_only)
                task_end_payload = {"task": task.name, "t_ms": int(time.time() * 1000)}
                self._broadcast("task_end", task_end_payload)
                self._record_event("task_end", task_end_payload)
                with self._state_lock:
                    self._active_task_name = None
                    self._active_stim_id = None
                    self._active_target_px = None
                self._wait_with_engine(eng, 400)
        finally:
            done_payload = {"t_ms": int(time.time() * 1000)}
            self._broadcast("tasks_done", done_payload)
            self._record_event("tasks_done", done_payload)
            if manage_streams and started_streams:
                self.stop_streams()
            eng.close()
            with self._state_lock:
                self._active_stim_id = None
                self._active_task_name = None
                self._active_target_px = None
                self._runtime_dims = None

    # ------------------------------------------------------------------ #

    def _handle_task_event(self, event, eng, dims: tuple[int, int] | None = None, web_only: bool = False):
        """Dispatch a single Task timeline event to the appropriate stim or wait helper."""
        etype = event.type
        payload = event.payload or {}
        if etype == "wait_ms":
            self._wait_with_engine(eng, int(payload.get("ms", 0)))
        elif etype == "stim_on":
            self._task_stim_on(payload, eng, dims, web_only=web_only)
        elif etype == "stim_move":
            self._task_stim_move(payload, eng, dims, web_only=web_only)
        elif etype == "stim_off":
            self._task_stim_off(payload, eng, web_only=web_only)

    def _wait_with_engine(self, eng, wait_ms: int):
        """Busy-wait for wait_ms milliseconds while pumping the stimulus engine's event loop."""
        wait_ms = max(0, int(wait_ms))
        end_ts = int(time.time() * 1000) + wait_ms
        while int(time.time() * 1000) < end_ts:
            eng.tick()

    def _task_stim_on(self, payload: dict, eng, dims: tuple[int, int] | None = None, web_only: bool = False):
        """Show a stimulus at normalised coordinates, record its t_on window, and broadcast stim/stim_on events."""
        stim_id = payload.get("id")
        x = payload.get("x_norm")
        y = payload.get("y_norm")
        if stim_id:
            with self._state_lock:
                self._active_stim_id = stim_id
                for tr in self.adapters:
                    self._buffers[tr][stim_id] = []
        if x is None or y is None:
            return
        if not web_only:
            eng.show_point(float(x), float(y))
        self._set_active_target_norm(float(x), float(y), dims=dims)
        w, h = dims if dims else self._screen_dims()
        tx_px = float(x) * w
        ty_px = float(y) * h
        t_on_ms = int(time.time() * 1000)
        with self._state_lock:
            self._stim_targets[stim_id] = (tx_px, ty_px)
            self._stim_t_on[stim_id] = t_on_ms
        self._broadcast(
            "stim",
            {
                "stim_id": stim_id,
                "x_norm": float(x),
                "y_norm": float(y),
                "target_x_px": tx_px,
                "target_y_px": ty_px,
                "t": t_on_ms,
            },
        )
        self._record_event(
            "stim",
            {
                "stim_id": stim_id,
                "task_name": self._active_task_name,
                "x_norm": float(x),
                "y_norm": float(y),
                "target_x_px": tx_px,
                "target_y_px": ty_px,
                "t_ms": t_on_ms,
            },
        )
        self._broadcast(
            "stim_on",
            {
                "id": stim_id,
                "x_norm": float(x),
                "y_norm": float(y),
                "t": int(time.time() * 1000),
            },
        )

    def _task_stim_move(self, payload: dict, eng, dims: tuple[int, int] | None = None, web_only: bool = False):
        """Move an already-visible stimulus to a new normalised position and broadcast stim/stim_move events."""
        x = payload.get("x_norm")
        y = payload.get("y_norm")
        if x is None or y is None:
            return
        with self._state_lock:
            current_stim_id = self._active_stim_id
        stim_id = payload.get("id") or current_stim_id
        if stim_id and stim_id != current_stim_id:
            with self._state_lock:
                self._active_stim_id = stim_id
                for tr in self.adapters:
                    self._buffers[tr][stim_id] = []
        if not web_only:
            eng.show_point(float(x), float(y))
        self._set_active_target_norm(float(x), float(y), dims=dims)
        w, h = dims if dims else self._screen_dims()
        tx_px = float(x) * w
        ty_px = float(y) * h
        with self._state_lock:
            self._stim_targets[stim_id] = (tx_px, ty_px)
        t_move_ms = int(time.time() * 1000)
        self._broadcast(
            "stim",
            {
                "stim_id": stim_id,
                "x_norm": float(x),
                "y_norm": float(y),
                "target_x_px": tx_px,
                "target_y_px": ty_px,
                "t": t_move_ms,
            },
        )
        self._record_event(
            "stim",
            {
                "stim_id": stim_id,
                "task_name": self._active_task_name,
                "x_norm": float(x),
                "y_norm": float(y),
                "target_x_px": tx_px,
                "target_y_px": ty_px,
                "t_ms": int(time.time() * 1000),
            },
        )
        self._broadcast(
            "stim_move",
            {
                "id": stim_id,
                "x_norm": float(x),
                "y_norm": float(y),
                "t": int(time.time() * 1000),
            },
        )

    def _task_stim_off(self, payload: dict, eng, web_only: bool = False):
        """Hide the active stimulus, close its t_on/t_off window, and broadcast stim_off events."""
        with self._state_lock:
            current_stim_id = self._active_stim_id
        stim_id = payload.get("id") or current_stim_id
        if not web_only:
            eng.hide()
        t_ms = int(time.time() * 1000)
        self._broadcast("stim_off", {"id": stim_id, "t": t_ms})
        self._record_event("stim_off", {"stim_id": stim_id, "task_name": self._active_task_name, "t_ms": t_ms})
        with self._state_lock:
            t_on_ms = self._stim_t_on.pop(stim_id, t_ms)
            if stim_id:
                self._stim_windows[stim_id] = (t_on_ms, t_ms)
            self._active_stim_id = None
            self._active_target_px = None

    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #

    def _fit_and_apply_models(self):
        """Train calibration model for each tracker based on buffered samples."""
        import pandas as pd
        from .calib.fitting import fit_dataframe

        w, h = self._screen_dims()
        with self._state_lock:
            buffers_snapshot = {
                tr_name: {
                    stim_id: list(samples)
                    for stim_id, samples in stim_map.items()
                }
                for tr_name, stim_map in self._buffers.items()
            }

        for tr_name, stim_map in buffers_snapshot.items():
            if tr_name == "webgazer":
                continue
            adapter = self.adapters.get(tr_name)
            if adapter is not None and getattr(adapter, "uses_internal_calibration", False):
                continue

            # Collect all buffered calibration samples into a DataFrame.
            rows = []
            for stim_id, samples in stim_map.items():
                if not str(stim_id).startswith("calib_"):
                    continue
                sliced = self._calib_slice(stim_id, samples)
                for s in sliced:
                    rows.append(s.model_dump())

            if len(rows) < 9:
                continue

            try:
                df = pd.DataFrame(rows)
                if "validity" not in df.columns:
                    df["validity"] = 0
                fit_out = fit_dataframe(df, width=w, height=h, per_stim_median=True)
                model = fit_out.model
            except Exception as exc:
                print(f"[orchestrator] Inline model fit failed for {tr_name}: {exc}")
                continue

            if tr_name in self.adapters:
                try:
                    self.adapters[tr_name].set_external_model(model)
                except Exception:
                    pass


    # ------------------------------------------------------------------ #

    def _calib_slice(self, stim_id: str, samples: List[Sample]) -> List[Sample]:
        """Drop the initial saccade and tail noise for calibration stimuli."""
        if not samples:
            return []
        with self._state_lock:
            window = self._stim_windows.get(stim_id)
        if not window:
            return samples
        t_on, t_off = window
        start = t_on + max(0, self._calib_settle_ms)
        end = max(start, t_off - self._calib_tail_ms)
        filtered = [
            s for s in samples
            if getattr(s, "timestamp_ms", 0) >= start and getattr(s, "timestamp_ms", 0) <= end
        ]
        return filtered or samples

    def _screen_dims(self) -> tuple[int, int]:
        """Return current screen pixel dimensions: runtime > session_meta > fallback 1280×720."""
        with self._state_lock:
            runtime_dims = self._runtime_dims
        if runtime_dims:
            return runtime_dims
        if self.session_meta:
            return self.session_meta.width_px, self.session_meta.height_px
        return 1280, 720

    def _set_active_target_norm(self, x_norm: float | None, y_norm: float | None, dims: tuple[int, int] | None = None):
        """Convert normalised coordinates to pixels and store as the current target for incoming samples."""
        if x_norm is None or y_norm is None:
            with self._state_lock:
                self._active_target_px = None
            return
        w, h = dims if dims else self._screen_dims()
        with self._state_lock:
            self._active_target_px = (x_norm * w, y_norm * h)

    def _clear_active_target(self):
        """Clear the active target so samples arriving between stims get no target annotation."""
        with self._state_lock:
            self._active_target_px = None

    def _get_stim_samples(self, tracker_id: str, stim_id: str) -> List[Sample]:
        """Return a snapshot of all samples buffered for a given tracker × stim_id pair."""
        with self._state_lock:
            return list(self._buffers[tracker_id][stim_id])

    def _wait_for_web_ready(self, eng, *, timeout_ms: int = 6000, min_wait_ms: int = 1500) -> None:
        """Wait for browser trackers to emit current-session samples (or timeout)."""
        web_names = [n for n in self.adapters if n in ("webgazer", "gazerecorder")]
        if not web_names:
            return

        gr_uses_wizard = any(
            name == "gazerecorder" and getattr(self.adapters.get(name), "uses_internal_calibration", False)
            for name in web_names
        )
        gr_present = "gazerecorder" in self.adapters
        if gr_present:
            msg = "Complete GazeRecorder calibration in the browser, then look straight ahead"
        else:
            msg = "Initializing browser trackers..."
        eng.show_message(msg, duration_ms=0, bg=(255, 255, 255), fg=(0, 0, 0), sub_fg=(0, 0, 0))
        start_ms = int(time.time() * 1000)
        gr_adapter = self.adapters.get("gazerecorder")
        while True:
            eng.tick(sleep_ms=30)
            elapsed = int(time.time() * 1000) - start_ms
            if elapsed >= min_wait_ms:
                if gr_present and gr_adapter is not None:
                    # Wait for GR SDK to signal calibration ready, AND for any other
                    # web trackers (e.g. webgazer) to have started streaming samples.
                    gr_ready = getattr(gr_adapter, "_sdk_calibrated", False)
                    other_web = [n for n in web_names if n != "gazerecorder"]
                    with self._state_lock:
                        others_ready = all(
                            self._web_ready_counts.get(n, 0) >= 3 for n in other_web
                        )
                    ready = gr_ready and (not other_web or others_ready)
                else:
                    with self._state_lock:
                        ready = all(self._web_ready_counts.get(name, 0) >= 3 for name in web_names)
                if ready:
                    break
            if elapsed >= timeout_ms:
                break


