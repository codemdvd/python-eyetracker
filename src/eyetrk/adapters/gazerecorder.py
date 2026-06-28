from __future__ import annotations

import numpy as np
from typing import Callable, Optional

from ..core.tracker import Tracker
from ..core.types import Sample, TrackerInfo, CalibModel, sanitize_predicted_point


class GazerecorderAdapter(Tracker):
    """
    Receives samples from the FastAPI WebSocket bridge.
    Browser page (tools/gazerecorder_bridge.html) must send normalized gaze:
      { tracker_id:"gazerecorder", session_id, timestamp_ms, x_norm, y_norm, confidence, ... }
    """

    def __init__(self):
        self._cb: Optional[Callable[[Sample], None]] = None
        self._session_id: str | None = None
        self._model: CalibModel | None = None
        self._out_w: int = 1280
        self._out_h: int = 720
        self.uses_internal_calibration: bool = True
        self._sdk_calibrated: bool = False

    def initialize(self, config: dict) -> TrackerInfo:
        """Store output resolution and calibration mode from config and return tracker metadata."""
        self._out_w = int(config.get("out_width", self._out_w))
        self._out_h = int(config.get("out_height", self._out_h))
        self.uses_internal_calibration = bool(config.get("use_internal_calibration", False))
        version = config.get("version", "unknown")
        return TrackerInfo(name="gazerecorder", version=version)

    def set_external_model(self, model: CalibModel) -> None:
        """Store a correction model applied on top of GR's cloud-calibrated output."""
        self._model = model

    def start_stream(self, callback: Callable[[Sample], None], session_id: str | None = None) -> None:
        """Register the sample callback; GR samples arrive via emit() from the WebSocket bridge."""
        self._cb = callback
        if session_id:
            self._session_id = session_id

    def stop(self) -> None:
        """Clear the callback, session id, and SDK calibration flag; WebSocket is managed by the bridge server."""
        self._cb = None
        self._session_id = None
        self._sdk_calibrated = False

    def on_event(self, event: str, payload: dict | None = None) -> None:
        if event == "bridge_start":
            payload = payload or {}
            screen_w = payload.get("screen_w")
            screen_h = payload.get("screen_h")
            if screen_w:
                try:
                    self._out_w = int(screen_w)
                except Exception:
                    pass
            if screen_h:
                try:
                    self._out_h = int(screen_h)
                except Exception:
                    pass
            try:
                from eyetrk.web_bridge import server as bridge_server

                bridge_server.push_event(
                    "gazerecorder",
                    {
                        "type": "start",
                        "session_id": self._session_id,
                        "screen_w": self._out_w,
                        "screen_h": self._out_h,
                        "phase": payload.get("phase", "calibration"),
                    },
                )
            except Exception:
                pass
            return
        if event == "stim_off":
            payload = payload or {}
            try:
                from eyetrk.web_bridge import server as bridge_server
                bridge_server.push_event("gazerecorder", {"type": "stim_off", "stim_id": payload.get("id")})
            except Exception:
                pass
            return
        if event != "stim":
            return
        payload = payload or {}

        stim_id = payload.get("stim_id") or payload.get("id")
        x_px = payload.get("target_x_px")
        y_px = payload.get("target_y_px")
        if x_px is None:
            x_px = payload.get("x_px")
        if y_px is None:
            y_px = payload.get("y_px")
        if x_px is None and payload.get("x_norm") is not None:
            x_px = float(payload.get("x_norm")) * self._out_w
        if y_px is None and payload.get("y_norm") is not None:
            y_px = float(payload.get("y_norm")) * self._out_h

        try:
            from eyetrk.web_bridge import server as bridge_server

            bridge_server.push_event(
                "gazerecorder",
                {
                    "type": "stim",
                    "stim_id": stim_id,
                    "x_px": x_px,
                    "y_px": y_px,
                    "target_x_px": x_px,
                    "target_y_px": y_px,
                    "screen_w": self._out_w,
                    "screen_h": self._out_h,
                },
            )
        except Exception:
            return

    def emit(self, sample: Sample):
        # Called by web_bridge server when a browser sample arrives.
        if sample.event == "internal_calibration":
            self.uses_internal_calibration = True
            return
        if sample.event == "sdk_calibrated":
            self._sdk_calibrated = True
            return
        if sample.event == "sdk_load_failed":
            print(
                "[gazerecorder] ERROR: GazeCloudAPI.js failed to load from CDN. "
                "Check internet connection. GazeRecorder requires network access.",
                flush=True,
            )
            return
        if self._session_id:
            if sample.session_id and sample.session_id != self._session_id:
                return
            if not sample.session_id:
                sample.session_id = self._session_id

        if sample.x_norm is not None and sample.y_norm is not None:
            xp, yp = self._apply_model(sample.x_norm, sample.y_norm)
            xp, yp = sanitize_predicted_point(
                xp,
                yp,
                screen_w=self._out_w,
                screen_h=self._out_h,
                x_norm=sample.x_norm,
                y_norm=sample.y_norm,
            )
            sample.x_px = xp
            sample.y_px = yp
            if sample.validity is None:
                sample.validity = 0

        if self._cb:
            self._cb(sample)

    def _apply_model(self, xn: float, yn: float) -> tuple[Optional[float], Optional[float]]:
        if self._model is None:
            return xn * self._out_w, yn * self._out_h

        if self._model.model_type == "poly2":
            params = self._model.params or {}
            # New format: uses stored powers and input_features for arbitrary feature sets.
            if "powers" in params and "input_features" in params:
                feats = _build_poly_features(
                    params,
                    {
                        "x_norm": float(xn),
                        "y_norm": float(yn),
                    },
                )
                if feats is None:
                    return xn * self._out_w, yn * self._out_h
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

            # Legacy 2-feature poly2
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
