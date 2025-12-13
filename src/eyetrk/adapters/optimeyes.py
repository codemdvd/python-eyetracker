from __future__ import annotations

import threading
import time
import cv2
import numpy as np

from .iris_common import aggregate_eyes, eye_from_landmarks, head_center
from ..core.tracker import Tracker
from ..core.types import Sample, TrackerInfo, CalibModel

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
        self._yaw_gain = 0.32
        self._pitch_gain = 0.15
        self._pose_alpha = 0.02
        self._yaw_origin: float | None = None
        self._pitch_origin: float | None = None
        self._last_raw_xy: tuple[float, float] | None = None
        self._preview = False
        self._win_name = "optimeyes preview"
        self._auto_gain = True
        self._auto_gain_alpha = 0.02
        self._auto_gain_margin = 0.05
        self._range_x: tuple[float, float] | None = None
        self._range_y: tuple[float, float] | None = None
        self._face_mesh = None
        self._model: CalibModel | None = None

    def initialize(self, config: dict) -> TrackerInfo:
        if not _MP_OK:
            raise RuntimeError("mediapipe is required for optimeyes fallback.")
        self._fps = float(config.get("fps", 30.0))
        self._cam_index = int(config.get("camera_index", 0))
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
        self._pose_alpha = float(config.get("pose_alpha", self._pose_alpha))
        self._preview = bool(config.get("preview", self._preview))
        self._auto_gain = bool(config.get("auto_gain", self._auto_gain))
        self._auto_gain_alpha = float(config.get("auto_gain_alpha", self._auto_gain_alpha))
        self._auto_gain_margin = float(config.get("auto_gain_margin", self._auto_gain_margin))
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

    def start_stream(self, callback, session_id: str | None = None):
        self._cb = callback
        if session_id is not None:
            self._session_id = session_id
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._cap:
            self._cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if self._thread and self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)

    def _loop(self):
        self._cap = cv2.VideoCapture(self._cam_index)
        try:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        except Exception:
            pass
        if not self._cap.isOpened():
            print("[OptimeyesAdapter] Could not open camera.")
            return

        w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[OptimeyesAdapter] Webcam opened ({w:.0f}x{h:.0f})")

        frame_id = 0
        misses = 0
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            xn, yn, conf = self._infer_norm(frame)
            if xn is None or yn is None:
                misses += 1
                if misses > self._fps:
                    self._ema_xy = None
                    self._last_raw_xy = None
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
                timestamp_ms=int(time.time() * 1000),
                frame_id=frame_id,
                x_norm=float(xn_s),
                y_norm=float(yn_s),
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
            time.sleep(max(0.0, 1 / self._fps - 0.001))

        if self._preview:
            try:
                cv2.destroyWindow(self._win_name)
            except Exception:
                pass

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

        cx, cy, conf, _used = aggregate_eyes(
            (left, right),
            min_eye_w=self._min_eye_w,
            min_eye_h=self._min_eye_h,
        )
        if cx is None or cy is None:
            fallback = [p.iris_xy for p in (left, right) if p is not None]
            fallback = [p for p in fallback if p[0] is not None and p[1] is not None]
            if not fallback:
                self._last_raw_xy = None
                return None, None, 0.0
            cx = float(np.mean([p[0] for p in fallback]))
            cy = float(np.mean([p[1] for p in fallback]))
            conf = 0.35

        yaw, pitch = self._head_pose(lm)
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
        if hx is not None and hy is not None:
            cx -= (hx - 0.5) * 0.12
            cy -= (hy - 0.5) * 0.1

        if self._auto_gain:
            cx, cy = self._apply_auto_gain(cx, cy)

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
        cxn = (cxn - 0.5) * (1 + pad * 2) + 0.5
        cyn = (cyn - 0.5) * (1 + pad * 2) + 0.5
        return float(max(0.0, min(1.0, cxn))), float(max(0.0, min(1.0, cyn)))

    def _apply_model(self, xn: float, yn: float):
        if self._model is None:
            return xn * self._out_w, yn * self._out_h
        try:
            if self._model.model_type == "poly2":
                coef_x = np.array(self._model.params["coef_x"], dtype=float)
                coef_y = np.array(self._model.params["coef_y"], dtype=float)
                ix = float(self._model.params["intercept_x"])
                iy = float(self._model.params["intercept_y"])
                x1 = float(xn)
                x2 = float(yn)
                feats = np.array([1.0, x1, x2, x1 * x1, x1 * x2, x2 * x2], dtype=float)
                xp = float(np.dot(feats, coef_x) + ix)
                yp = float(np.dot(feats, coef_y) + iy)
                return xp, yp
        except Exception:
            return None, None
        return None, None
