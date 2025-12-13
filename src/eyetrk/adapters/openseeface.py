from __future__ import annotations
from pathlib import Path

import json
import subprocess
import threading
import time
from typing import Optional, Sequence, Callable

from eyetrk.core.types import CalibModel, Sample
from eyetrk.core.tracker import Tracker, TrackerInfo


class OpenSeeFaceAdapter(Tracker):
    """
    Thin wrapper around the OpenSeeFace tracker.

    Expects OpenSeeFace to emit newline-delimited JSON with gaze keys:
      {"gaze": [x_norm, y_norm], "confidence": <float>}

    Configure `cmd` to point to your OpenSeeFace launcher. Example:
      ["python", "OpenSeeFace/facetracker.py", "--gaze-tracking", "1", "--silent", "1"]
    """

    def __init__(self):
        self._cb: Optional[Callable[[Sample], None]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._session_id: str | None = None
        self._cmd: Sequence[str] = []
        self._flip_x = False
        self._flip_y = False
        self._gain_x = 1.0
        self._gain_y = 1.0
        self._out_w = 1280
        self._out_h = 720
        self._model: CalibModel | None = None

    # ------------------------------------------------------------------ #

    def initialize(self, config: dict) -> TrackerInfo:
        # Команда запуска OpenSeeFace
        self._cmd = config.get("cmd", [])
        if not self._cmd:
            # По умолчанию: репо лежит в OpenSeeFace/, facetracker рядом
            self._cmd = [
                "python",
                "OpenSeeFace/facetracker.py",
                "--gaze-tracking", "1",
                "--silent", "1",
            ]

        self._flip_x = bool(config.get("flip_x", False))
        self._flip_y = bool(config.get("flip_y", False))
        self._gain_x = float(config.get("gain_x", 1.0))
        self._gain_y = float(config.get("gain_y", 1.0))
        self._out_w = int(config.get("out_width", self._out_w))
        self._out_h = int(config.get("out_height", self._out_h))

        version = config.get("version", "external")
        return TrackerInfo(name="openseeface", version=version)

    def set_external_model(self, model: CalibModel) -> None:
        self._model = model

    def start_stream(self, callback: Callable[[Sample], None], session_id: str | None = None) -> None:
        self._cb = callback
        if session_id is not None:
            self._session_id = session_id
        self._stop.clear()

        print("[openseeface] start_stream() called")
        print(f"[openseeface] cmd = {self._cmd}", flush=True)

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self._proc = None

    def on_event(self, event: str, payload: dict | None = None) -> None:
        # Пока ничего не делаем на события
        return

    # ------------------------------------------------------------------ #

    def _loop(self):
        import sys
        from pathlib import Path
        import os

        # Папка, где лежит facetracker.py и dshowcapture.py
        osf_dir = Path(__file__).resolve().parent / "OpenSeeFace"

        print(
            f"[openseeface] _loop starting, cmd={self._cmd}, cwd={osf_dir}",
            flush=True,
        )

        if not osf_dir.exists():
            print(f"[openseeface] ERROR: directory {osf_dir} does not exist", flush=True)
            return

        # === КЛЮЧЕВОЕ: использовать текущий интерпретатор, а не голый 'python' ===
        cmd = list(self._cmd) or []
        if cmd and cmd[0] in ("python", "python3"):
            cmd[0] = sys.executable

        print(f"[openseeface] final cmd = {cmd}", flush=True)

        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=osf_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # шлём всё в stdout, чтобы видеть ошибки
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            print(f"[openseeface] Failed to start command {cmd}: {exc}", flush=True)
            return

        frame_id = 0
        for line in self._proc.stdout or []:
            if self._stop.is_set():
                break
            line = line.strip()
            if not line:
                continue

            # отладочный вывод, чтобы видеть, что пишет facetracker
            print(f"[openseeface] raw line: {line}", flush=True)

            try:
                data = json.loads(line)
            except Exception:
                # строка не JSON — просто игнорируем
                continue

            xn, yn, conf = self._parse_gaze(data)
            if xn is None or yn is None:
                continue
            xp, yp = self._apply_model(xn, yn)

            sample = Sample(
                session_id=self._session_id or "",
                tracker_id="openseeface",
                timestamp_ms=int(time.time() * 1000),
                frame_id=frame_id,
                x_norm=float(xn),
                y_norm=float(yn),
                x_px=float(xp) if xp is not None else None,
                y_px=float(yp) if yp is not None else None,
                confidence=float(conf) if conf is not None else None,
                validity=0 if (conf is None or conf >= 0.5) else 1,
                stim_id=None,
                event="stream",
            )
            if self._cb:
                self._cb(sample)
            frame_id += 1

        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def _parse_gaze(self, data: dict):
        gaze = data.get("gaze")
        if isinstance(gaze, (list, tuple)) and len(gaze) >= 2:
            xn, yn = float(gaze[0]), float(gaze[1])
        else:
            xn = data.get("gaze_x")
            yn = data.get("gaze_y")
            if xn is None or yn is None:
                return None, None, None

        # Флипы
        if self._flip_x:
            xn = 1.0 - xn
        if self._flip_y:
            yn = 1.0 - yn

        # Гейны вокруг центра экрана
        if self._gain_x and self._gain_x != 1.0:
            xn = 0.5 + (xn - 0.5) * self._gain_x
        if self._gain_y and self._gain_y != 1.0:
            yn = 0.5 + (yn - 0.5) * self._gain_y

        # Клип к [0, 1]
        xn = max(0.0, min(1.0, xn))
        yn = max(0.0, min(1.0, yn))

        conf = data.get("confidence", data.get("score"))
        if conf is None:
            conf = 1.0

        return xn, yn, conf

    def _apply_model(self, xn: float, yn: float) -> tuple[Optional[float], Optional[float]]:
        # Без модели – просто масштабируем в пиксели окна стимула
        if self._model is None:
            return xn * self._out_w, yn * self._out_h

        if self._model.model_type == "poly2":
            try:
                import numpy as np
            except ImportError:
                return None, None

            try:
                coef_x = np.array(self._model.params["coef_x"], dtype=float)
                coef_y = np.array(self._model.params["coef_y"], dtype=float)
                ix = float(self._model.params["intercept_x"])
                iy = float(self._model.params["intercept_y"])
            except Exception:
                return None, None

            feats = np.array(
                [1.0, xn, yn, xn * xn, xn * yn, yn * yn],
                dtype=float,
            )
            xp = float(np.dot(feats, coef_x) + ix)
            yp = float(np.dot(feats, coef_y) + iy)
            return xp, yp

        return None, None
