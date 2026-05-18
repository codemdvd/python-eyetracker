# core/types.py

import math
from typing import Optional, Literal
from pydantic import BaseModel


class Sample(BaseModel):
    session_id: str
    tracker_id: str
    timestamp_ms: int
    frame_id: Optional[int] = None
    head_x: Optional[float] = None
    head_y: Optional[float] = None
    head_z: Optional[float] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    x_px: Optional[float] = None
    y_px: Optional[float] = None
    x_norm: Optional[float] = None
    y_norm: Optional[float] = None
    raw_x_norm: Optional[float] = None
    raw_y_norm: Optional[float] = None
    pose_x_norm: Optional[float] = None
    pose_y_norm: Optional[float] = None
    corrected_x_norm: Optional[float] = None
    corrected_y_norm: Optional[float] = None
    iris_abs_x_norm: Optional[float] = None
    iris_abs_y_norm: Optional[float] = None
    gaze_x_hf: Optional[float] = None
    gaze_y_hf: Optional[float] = None
    left_eye_x_norm: Optional[float] = None
    left_eye_y_norm: Optional[float] = None
    right_eye_x_norm: Optional[float] = None
    right_eye_y_norm: Optional[float] = None
    left_iris_abs_x_norm: Optional[float] = None
    left_iris_abs_y_norm: Optional[float] = None
    right_iris_abs_x_norm: Optional[float] = None
    right_iris_abs_y_norm: Optional[float] = None
    left_eye_w_norm: Optional[float] = None
    left_eye_h_norm: Optional[float] = None
    right_eye_w_norm: Optional[float] = None
    right_eye_h_norm: Optional[float] = None
    confidence: Optional[float] = None
    validity: int = 0
    stim_id: Optional[str] = None
    event: Optional[str] = None
    task_name: Optional[str] = None
    target_x_px: Optional[float] = None
    target_y_px: Optional[float] = None
    # GazeRecorder diagnostics (browser-side viewport state + raw SDK values)
    inner_h: Optional[int] = None
    inner_w: Optional[int] = None
    fullscreen: Optional[int] = None
    gr_gaze_x: Optional[float] = None
    gr_gaze_y: Optional[float] = None
    gr_doc_x: Optional[float] = None
    gr_doc_y: Optional[float] = None


SAMPLE_CSV_FIELDS = tuple(Sample.model_fields.keys())


def sample_to_csv_row(sample: "Sample") -> dict[str, object]:
    data = sample.model_dump(mode="python")
    return {field: data.get(field) for field in SAMPLE_CSV_FIELDS}


def sanitize_predicted_point(
    x_px: float | None,
    y_px: float | None,
    *,
    screen_w: int,
    screen_h: int,
    x_norm: float | None = None,
    y_norm: float | None = None,
    margin_ratio: float = 0.2,
) -> tuple[float | None, float | None]:
    fallback_x = float(x_norm * screen_w) if x_norm is not None else None
    fallback_y = float(y_norm * screen_h) if y_norm is not None else None

    if x_px is None or y_px is None:
        return fallback_x, fallback_y
    if not math.isfinite(x_px) or not math.isfinite(y_px):
        return fallback_x, fallback_y

    margin_x = max(1.0, float(screen_w) * margin_ratio)
    margin_y = max(1.0, float(screen_h) * margin_ratio)
    out_of_guard = (
        x_px < -margin_x
        or x_px > screen_w + margin_x
        or y_px < -margin_y
        or y_px > screen_h + margin_y
    )
    if out_of_guard:
        return fallback_x, fallback_y

    return (
        float(min(float(screen_w), max(0.0, x_px))),
        float(min(float(screen_h), max(0.0, y_px))),
    )


class CalibModel(BaseModel):
    model_type: Literal["poly2","homography","native"]
    params: dict
    fit_error_px: float | None = None


class TrackerInfo(BaseModel):
    name: str
    version: str
    reported_fps: float | None = None
