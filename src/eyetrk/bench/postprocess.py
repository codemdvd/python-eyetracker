from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from ..core.types import Sample


@dataclass
class _TaskBiasState:
    bias_x: float = 0.0
    bias_y: float = 0.0
    segment_key: tuple[str, str | None, float, float] | None = None
    segment_start_ms: int = 0
    recent_points: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=5))
    recent_errors: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=5))


class BenchmarkBiasCorrector:
    """
    Online bias correction for controlled benchmark tasks.

    This intentionally applies only to tasks where the expected stimulus is known
    and the participant is instructed to follow it. The goal is to compensate for
    slow session drift, not to replace calibration.
    """

    def __init__(
        self,
        *,
        settle_ms: int = 120,
        update_alpha: float = 0.12,
        stability_px: float = 120.0,
        pursuit_alpha: float = 0.05,
        pursuit_window: int = 5,
    ) -> None:
        self._settle_ms = int(settle_ms)
        self._update_alpha = float(update_alpha)
        self._stability_px = float(stability_px)
        self._pursuit_alpha = float(pursuit_alpha)
        self._pursuit_window = max(3, int(pursuit_window))
        self._states: dict[str, _TaskBiasState] = {}

    def apply(self, sample: Sample) -> Sample:
        """Subtract the current running bias from the sample's x_px/y_px in-place and update bias from the new residual."""
        task_name = getattr(sample, "task_name", None)
        if task_name not in {"fixation-grid", "step-saccades", "smooth-pursuit"}:
            return sample
        x_px = getattr(sample, "x_px", None)
        y_px = getattr(sample, "y_px", None)
        target_x = getattr(sample, "target_x_px", None)
        target_y = getattr(sample, "target_y_px", None)
        timestamp_ms = int(getattr(sample, "timestamp_ms", 0) or 0)
        if x_px is None or y_px is None or target_x is None or target_y is None:
            return sample

        task_name = str(task_name)
        state = self._states.setdefault(task_name, _TaskBiasState())
        if task_name == "smooth-pursuit":
            segment_key = (task_name, getattr(sample, "stim_id", None), 0.0, 0.0)
        else:
            segment_key = (task_name, getattr(sample, "stim_id", None), float(target_x), float(target_y))
        if state.segment_key != segment_key:
            state.segment_key = segment_key
            state.segment_start_ms = timestamp_ms
            state.recent_points.clear()
            state.recent_errors.clear()

        corrected_x = float(x_px) - state.bias_x
        corrected_y = float(y_px) - state.bias_y
        sample.x_px = corrected_x
        sample.y_px = corrected_y

        state.recent_points.append((corrected_x, corrected_y))
        state.recent_errors.append((corrected_x - float(target_x), corrected_y - float(target_y)))
        age_ms = max(0, timestamp_ms - state.segment_start_ms)
        if task_name == "smooth-pursuit":
            if len(state.recent_errors) < self._pursuit_window:
                return sample
            err = np.asarray(state.recent_errors, dtype=float)
            med_err_x = float(np.median(err[:, 0]))
            med_err_y = float(np.median(err[:, 1]))
            alpha = self._pursuit_alpha
            state.bias_x = (1.0 - alpha) * state.bias_x + alpha * med_err_x
            state.bias_y = (1.0 - alpha) * state.bias_y + alpha * med_err_y
            return sample

        if age_ms < self._settle_ms or len(state.recent_points) < 5:
            return sample
        pts = np.asarray(state.recent_points, dtype=float)
        if float(np.std(pts[:, 0])) > self._stability_px or float(np.std(pts[:, 1])) > self._stability_px:
            return sample
        err_x = float(np.median(pts[:, 0]) - float(target_x))
        err_y = float(np.median(pts[:, 1]) - float(target_y))
        alpha = self._update_alpha
        state.bias_x = (1.0 - alpha) * state.bias_x + alpha * err_x
        state.bias_y = (1.0 - alpha) * state.bias_y + alpha * err_y
        return sample


class BenchmarkAnchorRecalibrator:
    """
    Online benchmark-time recalibration using stable fixation anchors from
    fixation-grid. Once enough anchors are collected, a small 2D polynomial map
    is fitted and applied to subsequent benchmark samples.
    """

    def __init__(self, *, settle_ms: int = 120, min_anchors: int = 6, ridge_alpha: float = 1.0) -> None:
        self._settle_ms = int(settle_ms)
        self._min_anchors = int(min_anchors)
        self._ridge_alpha = float(ridge_alpha)
        self._anchor_points: list[tuple[float, float, float, float]] = []
        self._model: tuple[PolynomialFeatures, Ridge, Ridge] | None = None
        self._segment_key: tuple[str | None, str | None] | None = None
        self._segment_points: list[tuple[int, float, float, float, float]] = []

    def apply(self, sample: Sample) -> Sample:
        """Apply the fitted polynomial warp (if available) to x_px/y_px and collect fixation anchors for future fitting."""
        task_name = getattr(sample, "task_name", None)
        stim_id = getattr(sample, "stim_id", None)
        key = (task_name, stim_id)
        if key != self._segment_key:
            self._finalize_segment()
            self._segment_key = key
            self._segment_points = []

        x_px = getattr(sample, "x_px", None)
        y_px = getattr(sample, "y_px", None)
        target_x = getattr(sample, "target_x_px", None)
        target_y = getattr(sample, "target_y_px", None)
        timestamp_ms = int(getattr(sample, "timestamp_ms", 0) or 0)
        if x_px is None or y_px is None or target_x is None or target_y is None:
            return sample

        raw_x = float(x_px)
        raw_y = float(y_px)
        if task_name == "fixation-grid":
            self._segment_points.append((timestamp_ms, raw_x, raw_y, float(target_x), float(target_y)))

        if self._model is None:
            return sample

        poly, regx, regy = self._model
        feats = poly.transform(np.array([[raw_x, raw_y]], dtype=float))
        sample.x_px = float(regx.predict(feats)[0])
        sample.y_px = float(regy.predict(feats)[0])
        return sample

    def _finalize_segment(self) -> None:
        """Compute the median stable gaze position for the completed segment and add it as a calibration anchor."""
        if not self._segment_points:
            return
        task_name, _stim_id = self._segment_key if self._segment_key is not None else (None, None)
        if task_name != "fixation-grid":
            return
        start_ms = self._segment_points[0][0]
        stable = [row for row in self._segment_points if row[0] - start_ms >= self._settle_ms]
        if len(stable) < 5:
            return
        mx = float(np.median([row[1] for row in stable]))
        my = float(np.median([row[2] for row in stable]))
        tx = float(np.median([row[3] for row in stable]))
        ty = float(np.median([row[4] for row in stable]))
        self._anchor_points.append((mx, my, tx, ty))
        if len(self._anchor_points) >= self._min_anchors:
            self._fit_model()

    def _fit_model(self) -> None:
        """Fit a degree-2 polynomial Ridge regression from raw gaze coordinates to target coordinates using all collected anchors."""
        X = np.asarray([[row[0], row[1]] for row in self._anchor_points], dtype=float)
        Y = np.asarray([[row[2], row[3]] for row in self._anchor_points], dtype=float)
        poly = PolynomialFeatures(2, include_bias=True)
        Z = poly.fit_transform(X)
        regx = Ridge(alpha=self._ridge_alpha, fit_intercept=False).fit(Z, Y[:, 0])
        regy = Ridge(alpha=self._ridge_alpha, fit_intercept=False).fit(Z, Y[:, 1])
        self._model = (poly, regx, regy)
