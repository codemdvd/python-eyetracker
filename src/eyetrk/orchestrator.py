# src/eyetrk/orchestrator.py
from __future__ import annotations

import time
import os
import json
import numpy as np
from typing import Dict, List, TYPE_CHECKING
from collections import defaultdict

from .core.types import Sample
from .io.logger import RunLogger
from .io.schema import SessionMeta
from .calib.protocols import generate_9pt_grid
from .calib.acceptance import acceptance
from .core.calibrator import Poly2Fitter
from .bench.tasks import Task
if TYPE_CHECKING:
    from .cli import TrackerAdapter


class Orchestrator:
    def __init__(
        self,
        adapters: Dict[str, "TrackerAdapter"],
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
        self._calib_settle_ms = 220
        self._calib_tail_ms = 80

    # ------------------------------------------------------------------ #

    def _on_sample(self, sample: Sample):
        if not sample.session_id:
            sample.session_id = self.session_id

        if self._active_stim_id and not sample.stim_id:
            sample.stim_id = self._active_stim_id
            sample.event = sample.event or "stream"

        if self._active_task_name and not sample.task_name:
            sample.task_name = self._active_task_name

        if self._active_target_px:
            tx, ty = self._active_target_px
            sample.target_x_px = tx
            sample.target_y_px = ty

        self.logger.write_sample(sample)

        if sample.stim_id and sample.tracker_id:
            self._buffers[sample.tracker_id][sample.stim_id].append(sample)

    def start_streams(self):
        for a in self.adapters.values():
            a.start_stream(self._on_sample, self.session_id)

    def stop_streams(self):
        for a in self.adapters.values():
            try:
                a.stop()
            except Exception:
                pass

    def _broadcast(self, event: str, payload: dict | None = None):
        for a in self.adapters.values():
            a.on_event(event, payload or {})

    # ------------------------------------------------------------------ #

    def calibrate_all(self, dwell_ms=1000, gap_ms=300, fullscreen=False):
        """Run 9-point calibration sequence for all trackers."""
        from .stim.engine import StimEngine, StimConfig

        meta_w, meta_h = self._screen_dims()
        eng = StimEngine(StimConfig(width=meta_w, height=meta_h, fullscreen=fullscreen))
        w, h = eng.size
        self._runtime_dims = (w, h)
        seq = generate_9pt_grid(dwell_ms, gap_ms).points
        repeated = []
        self._stim_windows = {}

        try:
            self._active_task_name = "calibration"
            for pt in seq:
                for attempt in (1, 2):
                    self._active_stim_id = pt.id if attempt == 1 else f"{pt.id}_retry"
                    for tr in self.adapters:
                        self._buffers[tr][self._active_stim_id] = []

                    eng.show_point(pt.x_norm, pt.y_norm)
                    self._set_active_target_norm(pt.x_norm, pt.y_norm, dims=(w, h))
                    t_on = int(time.time() * 1000)
                    self._broadcast(
                        "calib_point_start",
                        {"id": self._active_stim_id, "x_norm": pt.x_norm, "y_norm": pt.y_norm, "t": t_on},
                    )

                    t_end = t_on + pt.dwell_ms
                    while int(time.time() * 1000) < t_end:
                        eng.tick()

                    eng.hide()
                    self._clear_active_target()
                    t_off = int(time.time() * 1000)
                    self._stim_windows[self._active_stim_id] = (t_on, t_off)
                    self._broadcast("calib_point_end", {"id": self._active_stim_id, "t": t_off})

                    base_id = self._active_stim_id.replace("_retry", "")
                    tx_px = pt.x_norm * w
                    ty_px = pt.y_norm * h
                    fails = []
                    for tr_name in self.adapters:
                        samples = self._calib_slice(self._active_stim_id, self._buffers[tr_name][self._active_stim_id])
                        ok = acceptance(
                            samples,
                            screen_w_px=w,
                            screen_h_px=h,
                            target_xy_px=(tx_px, ty_px),
                            max_offset_px=60.0,
                            max_disp_px=35.0,
                            )
                        if not ok:
                            fails.append(tr_name)

                    if fails and attempt == 1:
                        repeated.append(pt.id)
                        gap_until = int(time.time() * 1000) + pt.gap_ms
                        while int(time.time() * 1000) < gap_until:
                            eng.tick()
                        continue

                    break

                gap_until = int(time.time() * 1000) + pt.gap_ms
                while int(time.time() * 1000) < gap_until:
                    eng.tick()

            self._broadcast("calibration_done", {})
            self._fit_and_apply_models()

        finally:
            eng.close()
            self._active_stim_id = None
            self._active_task_name = None
            self._clear_active_target()
            self._runtime_dims = None
            self._stim_windows = {}

    # ------------------------------------------------------------------ #

    def run_tasks(self, tasks: List[Task], fullscreen: bool = False):
        """Play benchmark tasks timeline on the stimulus engine."""
        from .stim.engine import StimEngine, StimConfig

        meta_w, meta_h = self._screen_dims()
        eng = StimEngine(StimConfig(width=meta_w, height=meta_h, fullscreen=fullscreen))
        w, h = eng.size
        self._runtime_dims = (w, h)

        try:
            for task in tasks:
                self._active_task_name = task.name
                self._broadcast("task_start", {"task": task.name})
                for event in task.timeline:
                    self._handle_task_event(event, eng, dims=(w, h))
                self._broadcast("task_end", {"task": task.name})
                self._active_task_name = None
                self._active_stim_id = None
                self._clear_active_target()
                self._wait_with_engine(eng, 400)
        finally:
            eng.close()
            self._active_stim_id = None
            self._active_task_name = None
            self._clear_active_target()
            self._runtime_dims = None

    # ------------------------------------------------------------------ #

    def _handle_task_event(self, event, eng, dims: tuple[int, int] | None = None):
        etype = event.type
        payload = event.payload or {}
        if etype == "wait_ms":
            self._wait_with_engine(eng, int(payload.get("ms", 0)))
        elif etype == "stim_on":
            self._task_stim_on(payload, eng, dims)
        elif etype == "stim_move":
            self._task_stim_move(payload, eng, dims)
        elif etype == "stim_off":
            self._task_stim_off(payload, eng)

    def _wait_with_engine(self, eng, wait_ms: int):
        wait_ms = max(0, int(wait_ms))
        end_ts = int(time.time() * 1000) + wait_ms
        while int(time.time() * 1000) < end_ts:
            eng.tick()

    def _task_stim_on(self, payload: dict, eng, dims: tuple[int, int] | None = None):
        stim_id = payload.get("id")
        x = payload.get("x_norm")
        y = payload.get("y_norm")
        if stim_id:
            self._active_stim_id = stim_id
            for tr in self.adapters:
                self._buffers[tr][stim_id] = []
        if x is None or y is None:
            return
        eng.show_point(float(x), float(y))
        self._set_active_target_norm(float(x), float(y), dims=dims)
        self._broadcast(
            "stim_on",
            {
                "id": stim_id,
                "x_norm": float(x),
                "y_norm": float(y),
                "t": int(time.time() * 1000),
            },
        )

    def _task_stim_move(self, payload: dict, eng, dims: tuple[int, int] | None = None):
        x = payload.get("x_norm")
        y = payload.get("y_norm")
        if x is None or y is None:
            return
        stim_id = payload.get("id") or self._active_stim_id
        if stim_id and stim_id != self._active_stim_id:
            self._active_stim_id = stim_id
            for tr in self.adapters:
                self._buffers[tr][stim_id] = []
        eng.show_point(float(x), float(y))
        self._set_active_target_norm(float(x), float(y), dims=dims)
        self._broadcast(
            "stim_move",
            {
                "id": stim_id,
                "x_norm": float(x),
                "y_norm": float(y),
                "t": int(time.time() * 1000),
            },
        )

    def _task_stim_off(self, payload: dict, eng):
        stim_id = payload.get("id") or self._active_stim_id
        eng.hide()
        self._broadcast("stim_off", {"id": stim_id, "t": int(time.time() * 1000)})
        self._active_stim_id = None
        self._clear_active_target()

    # ------------------------------------------------------------------ #

    def _fit_and_apply_models(self):
        """Train calibration model for each tracker based on buffered samples.

        ВАЖНО: для webgazer мы НИЧЕГО не делаем — используем его собственную
        внутреннюю калибровку и не строим/не применяем внешнюю модель.
        """
        w, h = self._screen_dims()

        from .calib.protocols import generate_9pt_grid
        grid = {
            f"calib_{i + 1:02}": (x, y)
            for i, (x, y) in enumerate(
                [(p.x_norm, p.y_norm) for p in generate_9pt_grid().points]
            )
        }

        for tr_name, stim_map in self._buffers.items():
            # Для webgazer пропускаем внешний Poly2-калибратор
            if tr_name == "webgazer":
                continue

            X_list, Y_list = [], []

            for stim_id, samples in stim_map.items():
                base_id = stim_id.replace("_retry", "")
                if base_id not in grid:
                    continue

                samples = self._calib_slice(stim_id, samples)
                valid = []
                for s in samples:
                    if getattr(s, "validity", 0) != 0:
                        continue
                    if getattr(s, "x_norm", None) is None or getattr(s, "y_norm", None) is None:
                        continue
                    conf = getattr(s, "confidence", None)
                    if conf is not None and conf < 0.75:
                        continue
                    valid.append(s)
                if len(valid) < 3:
                    continue

                xn = float(np.median([s.x_norm for s in valid]))
                yn = float(np.median([s.y_norm for s in valid]))
                gx, gy = grid[base_id]
                tx, ty = gx * w, gy * h

                if np.isnan(xn) or np.isnan(yn):
                    continue

                X_list.append([xn, yn])
                Y_list.append([tx, ty])

            if len(X_list) < 4:
                continue

            X = np.array(X_list, dtype=float)
            Y = np.array(Y_list, dtype=float)

            model = Poly2Fitter().fit(X, Y)

            # Сохраняем модель на диск
            try:
                out_dir = os.path.join("data", "models")
                os.makedirs(out_dir, exist_ok=True)
                path = os.path.join(out_dir, f"{self.session_id}_{tr_name}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(model.model_dump(), f, ensure_ascii=False, indent=2)
            except Exception:
                pass

            # И сразу же применяем к адаптеру (кроме webgazer, который мы выше пропустили)
            if tr_name in self.adapters:
                try:
                    self.adapters[tr_name].set_external_model(model)
                except Exception:
                    # Не роняем оркестратор, если адаптер не умеет модель
                    pass


    # ------------------------------------------------------------------ #

    def _calib_slice(self, stim_id: str, samples: List[Sample]) -> List[Sample]:
        """Drop the initial saccade and tail noise for calibration stimuli."""
        if not samples:
            return []
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
        if getattr(self, "_runtime_dims", None):
            return self._runtime_dims  # type: ignore[attr-defined]
        if self.session_meta:
            return self.session_meta.width_px, self.session_meta.height_px
        return 1280, 720

    def _set_active_target_norm(self, x_norm: float | None, y_norm: float | None, dims: tuple[int, int] | None = None):
        if x_norm is None or y_norm is None:
            self._active_target_px = None
            return
        w, h = dims if dims else self._screen_dims()
        self._active_target_px = (x_norm * w, y_norm * h)

    def _clear_active_target(self):
        self._active_target_px = None
