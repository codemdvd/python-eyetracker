from __future__ import annotations

import json
import threading
import time
from pathlib import Path
import cv2
import numpy as np

from .iris_common import aggregate_eyes, eye_from_landmarks, head_center
from ..core.tracker import Tracker
from ..core.types import Sample, TrackerInfo, CalibModel, sanitize_predicted_point

try:
    import mediapipe as mp  # type: ignore

    _MP_OK = True
except Exception:
    _MP_OK = False


class OptimeyesAdapter(Tracker):
    """
    OptiMeyes-inspired adapter with a head-pose-aware iris pipeline.
    Uses MediaPipe FaceMesh landmarks but applies its own pose compensation and
    temporal smoothing instead of mirroring the mpiris adapter.
    """

    def __init__(self):
        self._cb = None
        self._stop = threading.Event()
        self._thread = None
        self._cap = None
        self._fps = 30.0
        self._session_id: str | None = None
        self._cam_index = 0
        self._camera_source = "webcam"
        self._daheng_index = 1
        self._daheng_exposure_us = 5000.0
        self._daheng_gain_db = 12.0
        self._width = 1280
        self._height = 720
        self._flip_x = False
        self._flip_y = False
        self._gain_x = 1.05
        self._gain_y = 1.0
        self._out_w = 1280
        self._out_h = 720
        self._ema_alpha = 0.2
        self._ema_xy: tuple[float, float] | None = None
        self._max_jump = 0.12
        self._min_eye_w = 0.02
        self._min_eye_h = 0.01
        self._blink_ratio = 0.16
        self._min_conf = 0.65
        self._yaw_gain = 0.0
        self._pitch_gain = 0.0
        self._head_gain_x = 0.0
        self._head_gain_y = 0.0
        self._pose_alpha = 0.02
        self._yaw_origin: float | None = None
        self._pitch_origin: float | None = None
        self._last_raw_xy: tuple[float, float] | None = None
        self._last_stage_raw_xy: tuple[float, float] | None = None
        self._last_stage_pose_xy: tuple[float, float] | None = None
        self._last_stage_corrected_xy: tuple[float, float] | None = None
        self._last_iris_abs_xy: tuple[float, float] | None = None
        self._last_left_eye = None
        self._last_right_eye = None
        self._last_head_x: float | None = None
        self._last_head_y: float | None = None
        self._last_yaw: float | None = None
        self._last_pitch: float | None = None
        self._preview = False
        self._win_name = "optimeyes preview"
        self._auto_gain = True
        self._auto_gain_alpha = 0.02
        self._auto_gain_margin = 0.05
        self._range_x: tuple[float, float] | None = None
        self._range_y: tuple[float, float] | None = None
        self._auto_gain_frozen = False
        self._frozen_range_x: tuple[float, float] | None = None
        self._frozen_range_y: tuple[float, float] | None = None
        self._video_path: str | None = None
        self._video_start_timestamp_ms: int | None = None
        self._record_video_path: str | None = None
        self._video_writer = None
        self._face_mesh = None
        self._model: CalibModel | None = None

    def initialize(self, config: dict) -> TrackerInfo:
        if not _MP_OK:
            raise RuntimeError("mediapipe is required for optimeyes fallback.")
        self._fps = float(config.get("fps", 30.0))
        self._cam_index = int(config.get("camera_index", 0))
        self._camera_source = str(config.get("camera_source", self._camera_source))
        self._daheng_index = int(config.get("daheng_device_index", self._daheng_index))
        self._daheng_exposure_us = float(config.get("daheng_exposure_us", self._daheng_exposure_us))
        self._daheng_gain_db = float(config.get("daheng_gain_db", self._daheng_gain_db))
        self._width = int(config.get("width", 1280))
        self._height = int(config.get("height", 720))
        self._flip_x = bool(config.get("flip_x", False))
        self._flip_y = bool(config.get("flip_y", False))
        self._gain_x = float(config.get("gain_x", 1.0))
        self._gain_y = float(config.get("gain_y", 1.0))
        self._out_w = int(config.get("out_width", self._out_w))
        self._out_h = int(config.get("out_height", self._out_h))
        self._ema_alpha = float(config.get("ema_alpha", self._ema_alpha))
        self._max_jump = float(config.get("max_jump_norm", self._max_jump))
        self._min_eye_w = float(config.get("min_eye_w", self._min_eye_w))
        self._min_eye_h = float(config.get("min_eye_h", self._min_eye_h))
        self._blink_ratio = float(config.get("blink_ratio", self._blink_ratio))
        self._min_conf = float(config.get("min_conf", self._min_conf))
        self._yaw_gain = float(config.get("yaw_gain", self._yaw_gain))
        self._pitch_gain = float(config.get("pitch_gain", self._pitch_gain))
        self._head_gain_x = float(config.get("head_gain_x", self._head_gain_x))
        self._head_gain_y = float(config.get("head_gain_y", self._head_gain_y))
        self._pose_alpha = float(config.get("pose_alpha", self._pose_alpha))
        self._preview = bool(config.get("preview", self._preview))
        self._auto_gain = bool(config.get("auto_gain", self._auto_gain))
        self._auto_gain_alpha = float(config.get("auto_gain_alpha", self._auto_gain_alpha))
        self._auto_gain_margin = float(config.get("auto_gain_margin", self._auto_gain_margin))
        self._video_path = config.get("video_path") or None
        self._video_start_timestamp_ms = _maybe_int(config.get("video_start_timestamp_ms"))
        self._record_video_path = config.get("record_video_path") or None
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        return TrackerInfo(name="optimeyes", version="mediapipe", reported_fps=self._fps)

    def set_external_model(self, model: CalibModel) -> None:
        self._model = model

    def on_event(self, event: str, payload: dict | None = None) -> None:
        if event in {"calibration_start", "tasks_start"}:
            self._set_auto_gain_frozen(True)
            return
        if event in {"calibration_done", "tasks_done"}:
            self._set_auto_gain_frozen(False)
            return

    def start_stream(self, callback, session_id: str | None = None):
        self._cb = callback
        if session_id is not None:
            self._session_id = session_id
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._set_auto_gain_frozen(False)
        if self._cap:
            self._cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if self._thread and self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        self._close_video_writer()

    def _loop(self):
        if self._video_path:
            self._cap = cv2.VideoCapture(self._video_path, cv2.CAP_ANY)
        elif self._camera_source == "daheng":
            from .daheng_capture import DahengCapture
            self._cap = DahengCapture(self._daheng_index, self._daheng_exposure_us, self._daheng_gain_db)
        else:
            self._cap = cv2.VideoCapture(self._cam_index)
            try:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                self._cap.set(cv2.CAP_PROP_FPS, self._fps)
                self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # DSHOW: 0.75 = auto
                self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)     # MSMF: 3 = auto
            except Exception:
                pass
        if not self._cap.isOpened():
            if self._video_path:
                print(f"[OptimeyesAdapter] Could not open video: {self._video_path}")
            else:
                print("[OptimeyesAdapter] Could not open camera.")
            return

        w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if self._video_path:
            print(f"[OptimeyesAdapter] Video source opened ({w:.0f}x{h:.0f}): {self._video_path}")
        else:
            print(f"[OptimeyesAdapter] Webcam opened ({w:.0f}x{h:.0f})")
        source_start_ms = self._video_start_timestamp_ms if self._video_path else int(time.time() * 1000)

        frame_id = 0
        misses = 0
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                if self._video_path:
                    break
                time.sleep(0.01)
                continue

            self._write_video_frame(frame, source_start_ms=source_start_ms)

            xn, yn, conf = self._infer_norm(frame)
            if xn is None or yn is None:
                misses += 1
                if misses > self._fps:
                    self._ema_xy = None
                    self._last_raw_xy = None
                    self._last_stage_raw_xy = None
                    self._last_stage_pose_xy = None
                    self._last_stage_corrected_xy = None
                    self._last_iris_abs_xy = None
                    self._last_left_eye = None
                    self._last_right_eye = None
                    self._last_head_x = None
                    self._last_head_y = None
                    self._last_yaw = None
                    self._last_pitch = None
                    self._yaw_origin = None
                    self._pitch_origin = None
                continue
            misses = 0

            if self._ema_xy is None:
                self._ema_xy = (xn, yn)
            else:
                ex, ey = self._ema_xy
                self._ema_xy = (
                    ex * (1 - self._ema_alpha) + xn * self._ema_alpha,
                    ey * (1 - self._ema_alpha) + yn * self._ema_alpha,
                )
            xn_s, yn_s = self._ema_xy

            xp, yp = self._apply_model(xn_s, yn_s)
            xp, yp = sanitize_predicted_point(
                xp,
                yp,
                screen_w=self._out_w,
                screen_h=self._out_h,
                x_norm=xn_s,
                y_norm=yn_s,
            )
            validity = 0 if (conf is None or conf >= self._min_conf) else 1

            if self._preview:
                disp = frame.copy()
                h, w = disp.shape[:2]
                cv2.circle(disp, (int(xn_s * w), int(yn_s * h)), 6, (0, 255, 0), -1)
                cv2.imshow(self._win_name, disp)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop
                    self._stop.set()
                    break

            s = Sample(
                session_id=self._session_id or "demo",
                tracker_id="optimeyes",
                timestamp_ms=self._compute_timestamp_ms(frame_id, source_start_ms),
                frame_id=frame_id,
                head_x=float(self._last_head_x) if self._last_head_x is not None else None,
                head_y=float(self._last_head_y) if self._last_head_y is not None else None,
                x_norm=float(xn_s),
                y_norm=float(yn_s),
                raw_x_norm=float(self._last_stage_raw_xy[0]) if self._last_stage_raw_xy is not None else None,
                raw_y_norm=float(self._last_stage_raw_xy[1]) if self._last_stage_raw_xy is not None else None,
                pose_x_norm=float(self._last_stage_pose_xy[0]) if self._last_stage_pose_xy is not None else None,
                pose_y_norm=float(self._last_stage_pose_xy[1]) if self._last_stage_pose_xy is not None else None,
                corrected_x_norm=float(self._last_stage_corrected_xy[0]) if self._last_stage_corrected_xy is not None else None,
                corrected_y_norm=float(self._last_stage_corrected_xy[1]) if self._last_stage_corrected_xy is not None else None,
                iris_abs_x_norm=float(self._last_iris_abs_xy[0]) if self._last_iris_abs_xy is not None else None,
                iris_abs_y_norm=float(self._last_iris_abs_xy[1]) if self._last_iris_abs_xy is not None else None,
                left_eye_x_norm=float(self._last_left_eye.x) if self._last_left_eye is not None else None,
                left_eye_y_norm=float(self._last_left_eye.y) if self._last_left_eye is not None else None,
                right_eye_x_norm=float(self._last_right_eye.x) if self._last_right_eye is not None else None,
                right_eye_y_norm=float(self._last_right_eye.y) if self._last_right_eye is not None else None,
                left_iris_abs_x_norm=float(self._last_left_eye.iris_xy[0]) if self._last_left_eye is not None else None,
                left_iris_abs_y_norm=float(self._last_left_eye.iris_xy[1]) if self._last_left_eye is not None else None,
                right_iris_abs_x_norm=float(self._last_right_eye.iris_xy[0]) if self._last_right_eye is not None else None,
                right_iris_abs_y_norm=float(self._last_right_eye.iris_xy[1]) if self._last_right_eye is not None else None,
                left_eye_w_norm=float(self._last_left_eye.w) if self._last_left_eye is not None else None,
                left_eye_h_norm=float(self._last_left_eye.h) if self._last_left_eye is not None else None,
                right_eye_w_norm=float(self._last_right_eye.w) if self._last_right_eye is not None else None,
                right_eye_h_norm=float(self._last_right_eye.h) if self._last_right_eye is not None else None,
                yaw=float(self._last_yaw) if self._last_yaw is not None else None,
                pitch=float(self._last_pitch) if self._last_pitch is not None else None,
                x_px=float(xp) if xp is not None else None,
                y_px=float(yp) if yp is not None else None,
                confidence=float(conf) if conf is not None else None,
                validity=validity,
                stim_id=None,
                event="stream",
            )
            if self._cb:
                self._cb(s)

            frame_id += 1
            if not self._video_path:
                time.sleep(max(0.0, 1 / self._fps - 0.001))

        if self._preview:
            try:
                cv2.destroyWindow(self._win_name)
            except Exception:
                pass
        self._close_video_writer()

    def _infer_norm(self, frame):
        if self._face_mesh is None:
            return None, None, None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self._face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None, None, 0.0
        lm = res.multi_face_landmarks[0].landmark

        left = eye_from_landmarks(
            lm,
            iris_idx=[468, 469, 470, 471],
            corners=[33, 133],
            lids=[159, 145],
            blink_ratio=self._blink_ratio,
        )
        right = eye_from_landmarks(
            lm,
            iris_idx=[473, 474, 475, 476],
            corners=[362, 263],
            lids=[386, 374],
            blink_ratio=self._blink_ratio,
        )

        self._last_left_eye = left
        self._last_right_eye = right
        cx, cy, conf, _used = aggregate_eyes(
            (left, right),
            min_eye_w=self._min_eye_w,
            min_eye_h=self._min_eye_h,
        )
        iris_pts = [
            eye.iris_xy
            for eye in (left, right)
            if eye is not None and eye.iris_xy[0] is not None and eye.iris_xy[1] is not None
        ]
        if iris_pts:
            self._last_iris_abs_xy = (
                float(np.mean([pt[0] for pt in iris_pts])),
                float(np.mean([pt[1] for pt in iris_pts])),
            )
        else:
            self._last_iris_abs_xy = None
        if cx is None or cy is None:
            fallback = [p.iris_xy for p in (left, right) if p is not None]
            fallback = [p for p in fallback if p[0] is not None and p[1] is not None]
            if not fallback:
                self._last_raw_xy = None
                self._last_stage_raw_xy = None
                self._last_stage_pose_xy = None
                self._last_stage_corrected_xy = None
                self._last_iris_abs_xy = None
                self._last_left_eye = None
                self._last_right_eye = None
                return None, None, 0.0
            cx = float(np.mean([p[0] for p in fallback]))
            cy = float(np.mean([p[1] for p in fallback]))
            conf = 0.35
        self._last_stage_raw_xy = (float(cx), float(cy))

        yaw, pitch = self._head_pose(lm)
        self._last_yaw = float(yaw) if yaw is not None else None
        self._last_pitch = float(pitch) if pitch is not None else None
        if yaw is not None:
            if self._yaw_origin is None:
                self._yaw_origin = yaw
            else:
                self._yaw_origin = (1 - self._pose_alpha) * self._yaw_origin + self._pose_alpha * yaw
            cx -= (yaw - self._yaw_origin) * self._yaw_gain
        if pitch is not None:
            if self._pitch_origin is None:
                self._pitch_origin = pitch
            else:
                self._pitch_origin = (1 - self._pose_alpha) * self._pitch_origin + self._pose_alpha * pitch
            cy -= (pitch - self._pitch_origin) * self._pitch_gain
        self._last_stage_pose_xy = (float(cx), float(cy))

        # Quick reject of wild jumps.
        if self._max_jump > 0 and self._last_raw_xy is not None:
            dx = abs(cx - self._last_raw_xy[0])
            dy = abs(cy - self._last_raw_xy[1])
            if dx > self._max_jump or dy > self._max_jump:
                self._last_raw_xy = None
                return None, None, 0.0
        self._last_raw_xy = (cx, cy)

        # Soft head translation correction to keep eyes centered if user drifts.
        hx, hy = head_center(lm)
        self._last_head_x = float(hx) if hx is not None else None
        self._last_head_y = float(hy) if hy is not None else None
        if hx is not None and hy is not None:
            cx -= (hx - 0.5) * self._head_gain_x
            cy -= (hy - 0.5) * self._head_gain_y

        if self._auto_gain:
            cx, cy = self._apply_auto_gain(cx, cy)
        self._last_stage_corrected_xy = (float(cx), float(cy))

        if self._flip_x:
            cx = 1.0 - cx
        if self._flip_y:
            cy = 1.0 - cy
        if self._gain_x and self._gain_x != 1.0:
            cx = 0.5 + (cx - 0.5) * self._gain_x
        if self._gain_y and self._gain_y != 1.0:
            cy = 0.5 + (cy - 0.5) * self._gain_y

        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        conf = float(max(0.0, min(1.0, conf if conf is not None else 0.0)))
        return cx, cy, conf

    def _head_pose(self, lm) -> tuple[float | None, float | None]:
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
        yaw = float(max(-1.5, min(1.5, yaw)))
        pitch = float(max(-1.5, min(1.5, pitch)))
        return yaw, pitch

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

    def _compute_timestamp_ms(self, frame_id: int, source_start_ms: int) -> int:
        if not self._video_path:
            return int(time.time() * 1000)
        rel_ms = float(self._cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) if self._cap is not None else 0.0
        if rel_ms <= 0.0:
            fps = float(self._cap.get(cv2.CAP_PROP_FPS) or self._fps or 30.0) if self._cap is not None else float(self._fps or 30.0)
            fps = max(1.0, fps)
            rel_ms = (frame_id * 1000.0) / fps
        return int(source_start_ms + rel_ms)

    def _write_video_frame(self, frame, *, source_start_ms: int) -> None:
        if not self._record_video_path:
            return
        writer = self._video_writer
        if writer is None:
            path = Path(self._record_video_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            fps = float(self._cap.get(cv2.CAP_PROP_FPS) or self._fps or 30.0) if self._cap is not None else float(self._fps or 30.0)
            fps = max(1.0, fps)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
            if not writer.isOpened():
                print(f"[OptimeyesAdapter] Failed to open video writer: {path}")
                writer.release()
                return
            self._video_writer = writer
            print(f"[OptimeyesAdapter] Recording video to: {path}")
            meta_path = Path(str(path) + ".json")
            meta_path.write_text(
                json.dumps(
                    {
                        "video_path": str(path),
                        "start_timestamp_ms": int(source_start_ms),
                        "fps": float(fps),
                        "width": int(width),
                        "height": int(height),
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

    def _apply_model(self, xn: float, yn: float):
        if self._model is None:
            return xn * self._out_w, yn * self._out_h
        try:
            if self._model.model_type != "poly2":
                return None, None
            params = self._model.params or {}
            if "powers" in params and "input_features" in params:
                _iabs_x = self._last_iris_abs_xy[0] if self._last_iris_abs_xy is not None else None
                _iabs_y = self._last_iris_abs_xy[1] if self._last_iris_abs_xy is not None else None
                feats = _build_poly_features(
                    params,
                    {
                        "x_norm": float(xn),
                        "y_norm": float(yn),
                        "right_eye_x_norm": float(self._last_right_eye.x) if self._last_right_eye is not None else None,
                        "right_eye_y_norm": float(self._last_right_eye.y) if self._last_right_eye is not None else None,
                        "right_eye_w_norm": float(self._last_right_eye.w) if self._last_right_eye is not None else None,
                        "right_eye_h_norm": float(self._last_right_eye.h) if self._last_right_eye is not None else None,
                        "left_eye_x_norm": float(self._last_left_eye.x) if self._last_left_eye is not None else None,
                        "left_eye_y_norm": float(self._last_left_eye.y) if self._last_left_eye is not None else None,
                        "left_eye_w_norm": float(self._last_left_eye.w) if self._last_left_eye is not None else None,
                        "left_eye_h_norm": float(self._last_left_eye.h) if self._last_left_eye is not None else None,
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
                    },
                )
                if feats is None:
                    return xn * self._out_w, yn * self._out_h
                coef_x = np.array(params["coef_x"], dtype=float)
                coef_y = np.array(params["coef_y"], dtype=float)
                ix = float(params["intercept_x"])
                iy = float(params["intercept_y"])
                xp = float(np.dot(feats, coef_x) + ix)
                yp = float(np.dot(feats, coef_y) + iy)
                return xp, yp

            coef_x = np.array(params["coef_x"], dtype=float)
            coef_y = np.array(params["coef_y"], dtype=float)
            ix = float(params["intercept_x"])
            iy = float(params["intercept_y"])
            x1 = float(xn)
            x2 = float(yn)
            feats = np.array([1.0, x1, x2, x1 * x1, x1 * x2, x2 * x2], dtype=float)
            xp = float(np.dot(feats, coef_x) + ix)
            yp = float(np.dot(feats, coef_y) + iy)
            return xp, yp
        except Exception:
            return None, None
        return None, None


def _build_poly_features(params: dict, feats_map: dict) -> np.ndarray | None:
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
        for value, pwr in zip(values, term):
            if value is None:
                return None
            term_val *= float(value) ** int(pwr)
        if not np.isfinite(term_val):
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
