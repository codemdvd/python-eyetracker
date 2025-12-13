# src/eyetrk/adapters/mpiris.py
from __future__ import annotations

import time
import threading
from typing import Optional, cast, Any

import cv2
import numpy as np

_MP_OK = False
mp: Any | None = None  # <── всегда объявлена

try:
    import mediapipe as _mp  # type: ignore[import]
    mp = _mp
    _MP_OK = True
except Exception:
    mp = None
    _MP_OK = False

from .iris_common import aggregate_eyes, eye_from_landmarks, head_center
from ..core.types import Sample, CalibModel


class MpirisAdapter:
    """
    Webcam-based tracker using MediaPipe FaceMesh (iris landmarks).
    Emits normalized gaze proxy (x_norm, y_norm). If a calibration model
    is set via set_external_model, also emits x_px, y_px.
    """

    def __init__(self) -> None:
        self._tracker_id = "mpiris"
        self._session_id: Optional[str] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # capture / runtime
        self._cam_index = 0
        self._target_fps = 30.0
        self._sleep = 0.0
        self._width = 1280
        self._height = 720
        self._flip_x = False
        self._flip_y = False
        self._gain_x = 1.0
        self._gain_y = 1.0
        self._out_w = 1280
        self._out_h = 720
        self._preview = False

        # mediapipe
        self._mp_face_mesh = None
        self._face_mesh = None

        # external calibration model
        self._model: Optional[CalibModel] = None

        # simple smoothing
        self._ema_alpha = 0.2
        self._ema_xy: Optional[tuple[float, float]] = None
        self._max_jump = 0.18
        self._last_raw_xy: Optional[tuple[float, float]] = None
        self._min_eye_w = 0.02
        self._min_eye_h = 0.008
        self._blink_ratio = 0.18
        self._head_origin: Optional[tuple[float, float]] = None
        self._head_alpha = 0.05
        self._head_gain_x = 1.0
        self._head_gain_y = 0.7
        self._use_head_comp = False
        self._min_conf = 0.5
        self._no_face_warned = False
        self._auto_gain = True
        self._auto_gain_alpha = 0.02
        self._auto_gain_margin = 0.05
        self._range_x: Optional[tuple[float, float]] = None
        self._range_y: Optional[tuple[float, float]] = None

    # ---- public API -----------------------------------------------------

    def initialize(self, cfg: dict) -> None:
        """cfg: {fps?: float, camera_index?: int, width?: int, height?: int}"""
        self._target_fps = float(cfg.get("fps", self._target_fps))
        self._cam_index = int(cfg.get("camera_index", self._cam_index))
        self._width = int(cfg.get("width", self._width))
        self._height = int(cfg.get("height", self._height))
        self._flip_x = bool(cfg.get("flip_x", self._flip_x))
        self._flip_y = bool(cfg.get("flip_y", self._flip_y))
        self._gain_x = float(cfg.get("gain_x", self._gain_x))
        self._gain_y = float(cfg.get("gain_y", self._gain_y))
        self._ema_alpha = float(cfg.get("ema_alpha", self._ema_alpha))
        self._head_alpha = float(cfg.get("head_alpha", self._head_alpha))
        self._head_gain_x = float(cfg.get("head_gain_x", self._head_gain_x))
        self._head_gain_y = float(cfg.get("head_gain_y", self._head_gain_y))
        self._use_head_comp = bool(cfg.get("head_comp", self._use_head_comp))
        self._min_conf = float(cfg.get("min_conf", self._min_conf))
        self._max_jump = float(cfg.get("max_jump_norm", self._max_jump))
        self._min_eye_w = float(cfg.get("min_eye_w", self._min_eye_w))
        self._min_eye_h = float(cfg.get("min_eye_h", self._min_eye_h))
        self._blink_ratio = float(cfg.get("blink_ratio", self._blink_ratio))
        self._auto_gain = bool(cfg.get("auto_gain", self._auto_gain))
        self._auto_gain_alpha = float(cfg.get("auto_gain_alpha", self._auto_gain_alpha))
        self._auto_gain_margin = float(cfg.get("auto_gain_margin", self._auto_gain_margin))
        self._out_w = int(cfg.get("out_width", self._out_w))
        self._out_h = int(cfg.get("out_height", self._out_h))
        self._preview = bool(cfg.get("preview", self._preview))
        self._sleep = max(0.0, 1.0 / self._target_fps - 0.001)

        if _MP_OK and mp is not None:
            face_mesh_module = cast(Any, mp).solutions.face_mesh
            self._mp_face_mesh = face_mesh_module
            self._face_mesh = face_mesh_module.FaceMesh(
                static_image_mode=False,
                refine_landmarks=True,
                max_num_faces=1,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3,
            )

    def set_external_model(self, model: CalibModel) -> None:
        self._model = model

    def start_stream(self, callback, session_id: str | None = None) -> None:
        if session_id is not None:
            self._session_id = session_id
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, args=(callback,), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def on_event(self, event: str, payload: dict | None = None) -> None:
        # no-op, but kept for compatibility
        return

    # ---- internal -------------------------------------------------------

    def _run_loop(self, callback) -> None:
        cap = cv2.VideoCapture(self._cam_index, cv2.CAP_DSHOW)
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            cap.set(cv2.CAP_PROP_FPS, self._target_fps)
            cap.set(cv2.CAP_PROP_FOURCC, cast(Any, cv2).VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        if not cap.isOpened():
            # fall back to default index 0 if custom index failed
            if self._cam_index != 0:
                cap.release()
                cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"[mpiris] Failed to open camera index {self._cam_index}")
            return

        ret_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        ret_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[mpiris] Camera opened (index={self._cam_index}, {ret_w:.0f}x{ret_h:.0f})")

        frame_id = 0
        misses = 0
        reopen_attempts = 0
        while self._running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]

            # MediaPipe expects RGB
            xn, yn, conf = self._infer_norm(frame, w, h)
            if xn is None or yn is None:
                misses += 1
                if misses > self._target_fps:
                    self._ema_xy = None
                    self._head_origin = None
                    self._last_raw_xy = None
            else:
                misses = 0

            if misses >= int(self._target_fps * 2):
                cap.release()
                reopen_attempts += 1
                backend = cv2.CAP_DSHOW if reopen_attempts % 2 == 1 else cv2.CAP_ANY
                cap = cv2.VideoCapture(self._cam_index, backend)
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                    cap.set(cv2.CAP_PROP_FPS, self._target_fps)
                except Exception:
                    pass
                if not cap.isOpened() and self._cam_index != 0:
                    cap.release()
                    cap = cv2.VideoCapture(0, backend)
                if not cap.isOpened():
                    print("[mpiris] Camera stream lost; giving up.")
                    break
                misses = 0
                continue

            # smoothing (EMA)
            if xn is not None and yn is not None:
                if self._ema_xy is None:
                    self._ema_xy = (xn, yn)
                else:
                    ex, ey = self._ema_xy
                    self._ema_xy = (
                        ex * (1 - self._ema_alpha) + xn * self._ema_alpha,
                        ey * (1 - self._ema_alpha) + yn * self._ema_alpha,
                    )
                xn_out, yn_out = self._ema_xy
            else:
                xn_out, yn_out = None, None

            x_px, y_px = self._apply_model(xn_out, yn_out) if (xn_out is not None and yn_out is not None) else (None, None)

            if self._preview:
                disp = frame.copy()
                h, w = disp.shape[:2]
                if xn_out is not None and yn_out is not None:
                    cv2.circle(disp, (int(xn_out * w), int(yn_out * h)), 6, (0, 255, 0), -1)
                cv2.imshow("mpiris preview", disp)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop
                    self._running = False
                    break

            sample = Sample(
                session_id=self._session_id or "",
                tracker_id=self._tracker_id,
                timestamp_ms=int(time.time() * 1000),
                frame_id=frame_id,
                x_norm=float(xn_out) if xn_out is not None else None,
                y_norm=float(yn_out) if yn_out is not None else None,
                x_px=float(x_px) if x_px is not None else None,
                y_px=float(y_px) if y_px is not None else None,
                confidence=float(conf) if conf is not None else None,
                validity=0
                if (xn_out is not None and yn_out is not None and (conf is None or conf >= self._min_conf))
                else 1,
                stim_id=None,
                event="stream",
            )
            callback(sample)

            frame_id += 1
            if self._sleep > 0:
                time.sleep(self._sleep)

        cap.release()
        if self._preview:
            try:
                cv2.destroyWindow("mpiris preview")
            except Exception:
                pass

    def _infer_norm(self, bgr_frame: np.ndarray, w: int, h: int) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Return (x_norm, y_norm, confidence) in [0..1] if possible."""
        if not _MP_OK or self._face_mesh is None:
            return None, None, None

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        res = self._face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            if not self._no_face_warned:
                print("[mpiris] No face detected. Check lighting and framing.")
                self._no_face_warned = True
            self._head_origin = None
            self._last_raw_xy = None
            return None, None, 0.0
        self._no_face_warned = False

        face = res.multi_face_landmarks[0]
        lm = face.landmark

        rel_left = eye_from_landmarks(
            lm,
            iris_idx=[468, 469, 470, 471],
            corners=[33, 133],
            lids=[159, 145],
            blink_ratio=self._blink_ratio,
        )
        rel_right = eye_from_landmarks(
            lm,
            iris_idx=[473, 474, 475, 476],
            corners=[362, 263],
            lids=[386, 374],
            blink_ratio=self._blink_ratio,
        )

        cx, cy, conf, _used = aggregate_eyes(
            (rel_left, rel_right),
            min_eye_w=self._min_eye_w,
            min_eye_h=self._min_eye_h,
        )

        if cx is None or cy is None:
            fallback = [p.iris_xy for p in (rel_left, rel_right) if p is not None]
            fallback = [p for p in fallback if p[0] is not None and p[1] is not None]
            if not fallback:
                self._last_raw_xy = None
                return None, None, 0.0
            cx = float(np.mean([p[0] for p in fallback]))
            cy = float(np.mean([p[1] for p in fallback]))
            conf = 0.35

        if self._max_jump > 0 and self._last_raw_xy is not None:
            dx = abs(cx - self._last_raw_xy[0])
            dy = abs(cy - self._last_raw_xy[1])
            if dx > self._max_jump or dy > self._max_jump:
                self._last_raw_xy = None
                return None, None, 0.0
        self._last_raw_xy = (cx, cy)

        if self._use_head_comp:
            hx, hy = head_center(lm)
            if hx is not None and hy is not None:
                self._update_head_origin(hx, hy)
                if self._head_origin is not None:
                    ox, oy = self._head_origin
                    cx -= (hx - ox) * self._head_gain_x
                    cy -= (hy - oy) * self._head_gain_y

        if self._auto_gain:
            cx, cy = self._apply_auto_gain(cx, cy)

        if self._flip_x:
            cx = 1.0 - cx
        if self._flip_y:
            cy = 1.0 - cy

        # Expand dynamics if the raw range is too narrow.
        if self._gain_x and self._gain_x != 1.0:
            cx = 0.5 + (cx - 0.5) * self._gain_x
        if self._gain_y and self._gain_y != 1.0:
            cy = 0.5 + (cy - 0.5) * self._gain_y

        cx = float(max(0.0, min(1.0, cx)))
        cy = float(max(0.0, min(1.0, cy)))
        conf = float(max(0.0, min(1.0, conf if conf is not None else 0.0)))
        return cx, cy, conf

    def _apply_auto_gain(self, cx: float, cy: float) -> tuple[float, float]:
        ax = self._auto_gain_alpha
        pad = self._auto_gain_margin
        if self._range_x is None:
            self._range_x = (cx, cx)
            self._range_y = (cy, cy)
            return cx, cy

        min_x, max_x = self._range_x
        min_y, max_y = self._range_y if self._range_y is not None else (cy, cy)
        min_x = min((1 - ax) * min_x + ax * cx, min_x, cx)
        max_x = max((1 - ax) * max_x + ax * cx, max_x, cx)
        min_y = min((1 - ax) * min_y + ax * cy, min_y, cy)
        max_y = max((1 - ax) * max_y + ax * cy, max_y, cy)
        self._range_x = (min_x, max_x)
        self._range_y = (min_y, max_y)

        span_x = max(1e-4, max_x - min_x)
        span_y = max(1e-4, max_y - min_y)
        cxn = (cx - min_x) / span_x
        cyn = (cy - min_y) / span_y
        # add a little padding to avoid sticking to borders
        cxn = (cxn - 0.5) * (1 + pad * 2) + 0.5
        cyn = (cyn - 0.5) * (1 + pad * 2) + 0.5
        return float(max(0.0, min(1.0, cxn))), float(max(0.0, min(1.0, cyn)))

    def _update_head_origin(self, hx: float, hy: float):
        if self._head_origin is None:
            self._head_origin = (hx, hy)
        else:
            ox, oy = self._head_origin
            self._head_origin = (
                ox * (1 - self._head_alpha) + hx * self._head_alpha,
                oy * (1 - self._head_alpha) + hy * self._head_alpha,
            )

    def _apply_model(self, xn: float, yn: float) -> tuple[Optional[float], Optional[float]]:
        if self._model is None:
            if self._out_w and self._out_h:
                return xn * self._out_w, yn * self._out_h
            return None, None

        if self._model.model_type == "poly2":
            try:
                coef_x = np.array(self._model.params["coef_x"], dtype=float)
                coef_y = np.array(self._model.params["coef_y"], dtype=float)
                ix = float(self._model.params["intercept_x"])
                iy = float(self._model.params["intercept_y"])
            except Exception:
                return None, None

            x1 = float(xn)
            x2 = float(yn)
            # PolynomialFeatures(degree=2, include_bias=True) manual expansion:
            feats = np.array([1.0, x1, x2, x1 * x1, x1 * x2, x2 * x2], dtype=float)
            xp = float(np.dot(feats, coef_x) + ix)
            yp = float(np.dot(feats, coef_y) + iy)
            return xp, yp

        # other model types can be implemented later
        return None, None
