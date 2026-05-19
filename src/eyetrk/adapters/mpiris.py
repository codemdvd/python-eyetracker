# src/eyetrk/adapters/mpiris.py
from __future__ import annotations

import json
import time
import threading
from pathlib import Path
from typing import Optional, cast, Any

import cv2
import numpy as np

_MP_OK = False
mp: Any | None = None  # always declared

try:
    import mediapipe as _mp  # type: ignore[import]
    mp = _mp
    _MP_OK = True
except Exception:
    mp = None
    _MP_OK = False

from .iris_common import aggregate_eyes, eye_from_landmarks, head_center, estimate_head_pose_pnp, iris_gaze_hf
from ..core.tracker import Tracker
from ..core.types import Sample, CalibModel, TrackerInfo, sanitize_predicted_point


class MpirisAdapter(Tracker):
    """
    Webcam-based tracker using MediaPipe FaceMesh (iris landmarks).
    Emits normalized gaze proxy (x_norm, y_norm). If a calibration model
    is set via set_external_model, also emits x_px, y_px.
    """

    def __init__(self) -> None:
        self._tracker_id = "mpiris"
        self.uses_internal_calibration = False
        self._session_id: Optional[str] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # capture / runtime
        self._cam_index = 0
        self._camera_source = "webcam"
        self._camera_backend = "auto"  # "auto", "dshow", "msmf"
        self._daheng_index = 1
        self._daheng_exposure_us = 5000.0
        self._daheng_gain_db = 12.0
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
        self._video_path: str | None = None
        self._video_start_timestamp_ms: int | None = None
        self._record_video_path: str | None = None
        self._video_writer = None

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
        self._auto_gain_frozen = False
        self._frozen_range_x: Optional[tuple[float, float]] = None
        self._frozen_range_y: Optional[tuple[float, float]] = None
        self._last_head_x: Optional[float] = None
        self._last_head_y: Optional[float] = None
        self._last_yaw: Optional[float] = None
        self._last_pitch: Optional[float] = None
        self._last_iris_abs_xy: Optional[tuple[float, float]] = None
        self._last_gaze_x_hf: Optional[float] = None
        self._last_gaze_y_hf: Optional[float] = None
        self._unmirror_cam = True
        self._raw_debug_xy: Optional[tuple[float, float]] = None
        self._corrected_debug_xy: Optional[tuple[float, float]] = None

    # ---- public API -----------------------------------------------------

    def initialize(self, cfg: dict) -> TrackerInfo:
        """cfg: {fps?: float, camera_index?: int, width?: int, height?: int}"""
        self._target_fps = float(cfg.get("fps", self._target_fps))
        self._cam_index = int(cfg.get("camera_index", self._cam_index))
        self._camera_source = str(cfg.get("camera_source", self._camera_source))
        self._camera_backend = str(cfg.get("camera_backend", self._camera_backend))
        self._daheng_index = int(cfg.get("daheng_device_index", self._daheng_index))
        self._daheng_exposure_us = float(cfg.get("daheng_exposure_us", self._daheng_exposure_us))
        self._daheng_gain_db = float(cfg.get("daheng_gain_db", self._daheng_gain_db))
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
        self._video_path = cfg.get("video_path") or None
        self._video_start_timestamp_ms = _maybe_int(cfg.get("video_start_timestamp_ms"))
        self._record_video_path = cfg.get("record_video_path") or None
        self._sleep = max(0.0, 1.0 / self._target_fps - 0.001)
        self._unmirror_cam = bool(cfg.get("cam_unmirror", self._unmirror_cam))

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
        version = "mediapipe" if _MP_OK else "unavailable"
        return TrackerInfo(name=self._tracker_id, version=version, reported_fps=self._target_fps)

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
        self._set_auto_gain_frozen(False)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self._close_video_writer()

    def on_event(self, event: str, payload: dict | None = None) -> None:
        if event in {"calibration_start", "tasks_start"}:
            self._set_auto_gain_frozen(True)
            return
        if event in {"calibration_done", "tasks_done"}:
            self._set_auto_gain_frozen(False)
            return
        return

    # ---- internal -------------------------------------------------------

    def _run_loop(self, callback) -> None:
        cap = None
        if self._video_path:
            cap = cv2.VideoCapture(self._video_path, cv2.CAP_ANY)
        elif self._camera_source == "daheng":
            from .daheng_capture import DahengCapture
            cap = DahengCapture(self._daheng_index, self._daheng_exposure_us, self._daheng_gain_db)
        else:
            _backend_map = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}
            for attempt in range(12):  # retry ~6s total
                if self._camera_backend in _backend_map:
                    backend = _backend_map[self._camera_backend]
                else:  # "auto"
                    backend = cv2.CAP_DSHOW if attempt % 2 == 0 else cv2.CAP_ANY
                idx = self._cam_index
                cap = cv2.VideoCapture(idx, backend)
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                    cap.set(cv2.CAP_PROP_FPS, self._target_fps)
                    cap.set(cv2.CAP_PROP_FOURCC, cast(Any, cv2).VideoWriter_fourcc(*"MJPG"))
                    # Re-enable auto-exposure: DSHOW resets it on open, causing ~1s flicker.
                    # 0.75 = auto for DSHOW; MSMF uses 3 — try both, ignore errors.
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
                except Exception:
                    pass
                if cap.isOpened():
                    break
                if idx != 0:
                    cap.release()
                    cap = cv2.VideoCapture(0, backend)
                    if cap.isOpened():
                        break
                cap.release()
                cap = None
                time.sleep(0.5)

        if not cap or not cap.isOpened():
            if self._video_path:
                print(f"[mpiris] Failed to open video source: {self._video_path}")
            else:
                print(f"[mpiris] Failed to open camera index {self._cam_index}")
            return

        ret_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        ret_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if self._video_path:
            print(f"[mpiris] Video source opened ({ret_w:.0f}x{ret_h:.0f}): {self._video_path}")
        else:
            print(f"[mpiris] Camera opened (index={self._cam_index}, {ret_w:.0f}x{ret_h:.0f})")
        source_start_ms = self._video_start_timestamp_ms if self._video_path else int(time.time() * 1000)

        frame_id = 0
        misses = 0
        reopen_attempts = 0
        while self._running:
            ok, frame = cap.read()
            if not ok:
                if self._video_path:
                    break
                time.sleep(0.05)
                continue

            if self._unmirror_cam and self._camera_source != "daheng":
                frame = cv2.flip(frame, 1)

            self._write_video_frame(frame, source_start_ms=source_start_ms, cap=cap)

            h, w = frame.shape[:2]

            # MediaPipe expects RGB
            xn, yn, conf, hx, hy = self._infer_norm(frame, w, h)
            self._last_head_x = hx
            self._last_head_y = hy
            if xn is None or yn is None:
                misses += 1
                self._raw_debug_xy = None
                self._corrected_debug_xy = None
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
            x_px, y_px = sanitize_predicted_point(
                x_px,
                y_px,
                screen_w=self._out_w,
                screen_h=self._out_h,
                x_norm=xn_out,
                y_norm=yn_out,
            )

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
                timestamp_ms=self._compute_timestamp_ms(cap, frame_id, source_start_ms),
                frame_id=frame_id,
                head_x=float(hx) if hx is not None else None,
                head_y=float(hy) if hy is not None else None,
                head_z=None,
                yaw=self._last_yaw,
                pitch=self._last_pitch,
                roll=None,
                x_norm=float(xn_out) if xn_out is not None else None,
                y_norm=float(yn_out) if yn_out is not None else None,
                raw_x_norm=float(self._raw_debug_xy[0]) if self._raw_debug_xy is not None else None,
                raw_y_norm=float(self._raw_debug_xy[1]) if self._raw_debug_xy is not None else None,
                corrected_x_norm=float(self._corrected_debug_xy[0]) if self._corrected_debug_xy is not None else None,
                corrected_y_norm=float(self._corrected_debug_xy[1]) if self._corrected_debug_xy is not None else None,
                iris_abs_x_norm=float(self._last_iris_abs_xy[0]) if self._last_iris_abs_xy is not None else None,
                iris_abs_y_norm=float(self._last_iris_abs_xy[1]) if self._last_iris_abs_xy is not None else None,
                gaze_x_hf=self._last_gaze_x_hf,
                gaze_y_hf=self._last_gaze_y_hf,
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
        self._close_video_writer()
        if self._preview:
            try:
                cv2.destroyWindow("mpiris preview")
            except Exception:
                pass

    def _infer_norm(self, bgr_frame: np.ndarray, w: int, h: int) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Return (x_norm, y_norm, confidence, head_x, head_y)."""
        if not _MP_OK or self._face_mesh is None:
            return None, None, None, None, None

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        res = self._face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            if not self._no_face_warned:
                print("[mpiris] No face detected. Check lighting and framing.")
                self._no_face_warned = True
            self._head_origin = None
            self._last_raw_xy = None
            self._raw_debug_xy = None
            self._corrected_debug_xy = None
            self._last_yaw = None
            self._last_pitch = None
            self._last_iris_abs_xy = None
            self._last_gaze_x_hf = None
            self._last_gaze_y_hf = None
            return None, None, 0.0, None, None
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

        # Absolute iris position in camera frame (same convention as optimeyes)
        iris_pts = [
            p.iris_xy
            for p in (rel_left, rel_right)
            if p is not None and p.iris_xy[0] is not None
        ]
        if iris_pts:
            self._last_iris_abs_xy = (
                float(np.mean([pt[0] for pt in iris_pts])),
                float(np.mean([pt[1] for pt in iris_pts])),
            )
        else:
            self._last_iris_abs_xy = None

        # Head pose angles using same 2D geometry as optimeyes
        yaw, pitch = self._head_pose(lm)
        self._last_yaw = float(yaw) if yaw is not None else None
        self._last_pitch = float(pitch) if pitch is not None else None

        # 3D head-frame gaze direction (rotation-invariant)
        _R, _ = estimate_head_pose_pnp(lm, w, h)
        if _R is not None:
            self._last_gaze_x_hf, self._last_gaze_y_hf = iris_gaze_hf(lm, _R, w, h)
        else:
            self._last_gaze_x_hf = None
            self._last_gaze_y_hf = None

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
                self._raw_debug_xy = None
                self._corrected_debug_xy = None
                return None, None, 0.0, None, None
            cx = float(np.mean([p[0] for p in fallback]))
            cy = float(np.mean([p[1] for p in fallback]))
            conf = 0.35

        if self._max_jump > 0 and self._last_raw_xy is not None:
            dx = abs(cx - self._last_raw_xy[0])
            dy = abs(cy - self._last_raw_xy[1])
            if dx > self._max_jump or dy > self._max_jump:
                self._last_raw_xy = None
                self._raw_debug_xy = None
                self._corrected_debug_xy = None
                return None, None, 0.0, None, None
        self._last_raw_xy = (cx, cy)
        self._raw_debug_xy = (cx, cy)

        hx, hy = head_center(lm)
        if self._use_head_comp:
            if hx is not None and hy is not None:
                self._update_head_origin(hx, hy)
                if self._head_origin is not None:
                    ox, oy = self._head_origin
                    cx -= (hx - ox) * self._head_gain_x
                    cy -= (hy - oy) * self._head_gain_y
        self._corrected_debug_xy = (cx, cy)

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
        return cx, cy, conf, hx, hy

    def _head_pose(self, lm) -> tuple[Optional[float], Optional[float]]:
        """Compute 2D-geometry head pose angles (same convention as optimeyes)."""
        try:
            nose = lm[1]
            left_eye = lm[33]
            right_eye = lm[263]
            brow = lm[9] if len(lm) > 9 else nose
            chin = lm[152] if len(lm) > 152 else nose
        except Exception:
            return None, None
        mid_x = (left_eye.x + right_eye.x) * 0.5
        mid_y = (left_eye.y + right_eye.y) * 0.5
        eye_span = max(abs(right_eye.x - left_eye.x), 1e-4)
        yaw = (nose.x - mid_x) / eye_span
        vertical_span = max(abs(chin.y - brow.y), 1e-4)
        pitch = (nose.y - mid_y) / vertical_span
        return float(max(-1.5, min(1.5, yaw))), float(max(-1.5, min(1.5, pitch)))

    def _apply_auto_gain(self, cx: float, cy: float) -> tuple[float, float]:
        ax = self._auto_gain_alpha
        pad = self._auto_gain_margin
        range_x = self._frozen_range_x if self._auto_gain_frozen else self._range_x
        range_y = self._frozen_range_y if self._auto_gain_frozen else self._range_y
        if range_x is None:
            self._range_x = (cx, cx)
            self._range_y = (cy, cy)
            if self._auto_gain_frozen:
                self._frozen_range_x = self._range_x
                self._frozen_range_y = self._range_y
            return cx, cy

        min_x, max_x = range_x
        min_y, max_y = range_y if range_y is not None else (cy, cy)
        if not self._auto_gain_frozen:
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

    def _set_auto_gain_frozen(self, frozen: bool) -> None:
        if not self._auto_gain:
            return
        self._auto_gain_frozen = bool(frozen)
        if frozen:
            self._frozen_range_x = self._range_x
            self._frozen_range_y = self._range_y
        else:
            self._frozen_range_x = None
            self._frozen_range_y = None

    def _compute_timestamp_ms(self, cap, frame_id: int, source_start_ms: int) -> int:
        if not self._video_path:
            return int(time.time() * 1000)
        rel_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        if rel_ms <= 0.0:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or self._target_fps or 30.0)
            fps = max(1.0, fps)
            rel_ms = (frame_id * 1000.0) / fps
        return int(source_start_ms + rel_ms)

    def _write_video_frame(self, frame: np.ndarray, *, source_start_ms: int, cap) -> None:
        if not self._record_video_path:
            return
        writer = self._video_writer
        if writer is None:
            path = Path(self._record_video_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            fps = float(cap.get(cv2.CAP_PROP_FPS) or self._target_fps or 30.0)
            fps = max(1.0, fps)
            fourcc = cast(Any, cv2).VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
            if not writer.isOpened():
                print(f"[mpiris] Failed to open video writer: {path}")
                writer.release()
                return
            self._video_writer = writer
            print(f"[mpiris] Recording video to: {path}")
            meta_path = Path(str(path) + ".json")
            meta_path.write_text(
                json.dumps(
                    {
                        "video_path": str(path),
                        "start_timestamp_ms": int(source_start_ms),
                        "fps": float(fps),
                        "width": int(width),
                        "height": int(height),
                        "cam_unmirror_applied": bool(self._unmirror_cam),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        writer.write(frame)

    def _close_video_writer(self) -> None:
        writer = self._video_writer
        self._video_writer = None
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass

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
            params = self._model.params or {}
            # New format (arbitrary feature set)
            if "powers" in params and "input_features" in params:
                _iabs_x = self._last_iris_abs_xy[0] if self._last_iris_abs_xy is not None else None
                _iabs_y = self._last_iris_abs_xy[1] if self._last_iris_abs_xy is not None else None
                feats = _build_poly_features(
                    params,
                    {
                        "x_norm": float(xn),
                        "y_norm": float(yn),
                        "raw_x_norm": self._raw_debug_xy[0] if self._raw_debug_xy is not None else None,
                        "raw_y_norm": self._raw_debug_xy[1] if self._raw_debug_xy is not None else None,
                        "corrected_x_norm": self._corrected_debug_xy[0] if self._corrected_debug_xy is not None else None,
                        "corrected_y_norm": self._corrected_debug_xy[1] if self._corrected_debug_xy is not None else None,
                        "iris_abs_x_norm": _iabs_x,
                        "iris_abs_y_norm": _iabs_y,
                        "iris_from_head_x_norm": (_iabs_x - self._last_head_x) if (_iabs_x is not None and self._last_head_x is not None) else None,
                        "iris_from_head_y_norm": (_iabs_y - self._last_head_y) if (_iabs_y is not None and self._last_head_y is not None) else None,
                        "head_x": self._last_head_x,
                        "head_y": self._last_head_y,
                        "head_z": None,
                        "yaw": self._last_yaw,
                        "pitch": self._last_pitch,
                        "roll": None,
                        "gaze_x_hf": self._last_gaze_x_hf,
                        "gaze_y_hf": self._last_gaze_y_hf,
                    },
                )
                if feats is None:
                    if self._out_w and self._out_h:
                        return xn * self._out_w, yn * self._out_h
                    return None, None
                try:
                    coef_x = np.array(params["coef_x"], dtype=float)
                    coef_y = np.array(params["coef_y"], dtype=float)
                    ix = float(params["intercept_x"])
                    iy = float(params["intercept_y"])
                except Exception:
                    return None, None
                xp = float(np.dot(feats, coef_x) + ix)
                yp = float(np.dot(feats, coef_y) + iy)
                return xp, yp

            # Legacy 2-feature model
            try:
                coef_x = np.array(params["coef_x"], dtype=float)
                coef_y = np.array(params["coef_y"], dtype=float)
                ix = float(params["intercept_x"])
                iy = float(params["intercept_y"])
            except Exception:
                return None, None

            x1 = float(xn)
            x2 = float(yn)
            feats = np.array([1.0, x1, x2, x1 * x1, x1 * x2, x2 * x2], dtype=float)
            xp = float(np.dot(feats, coef_x) + ix)
            yp = float(np.dot(feats, coef_y) + iy)
            return xp, yp

        # other model types can be implemented later
        return None, None


def _build_poly_features(params: dict, feats_map: dict) -> Optional[np.ndarray]:
    powers = params.get("powers")
    inputs = params.get("input_features")
    if not powers or not inputs:
        return None
    values = [feats_map.get(name) for name in inputs]
    if any(v is None for v in values):
        return None
    out = []
    for term in powers:
        term_val = 1.0
        for v, pwr in zip(values, term):
            if v is None:
                term_val = None
                break
            term_val *= float(v) ** int(pwr)
        if term_val is None or not np.isfinite(term_val):
            return None
        out.append(term_val)
    return np.array(out, dtype=float)


def _maybe_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None

