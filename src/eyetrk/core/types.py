# core/types.py

from typing import Optional, Literal
from pydantic import BaseModel


class Sample(BaseModel):
    session_id: str
    tracker_id: str
    timestamp_ms: int
    frame_id: Optional[int] = None
    x_px: Optional[float] = None
    y_px: Optional[float] = None
    x_norm: Optional[float] = None
    y_norm: Optional[float] = None
    confidence: Optional[float] = None
    validity: int = 0
    stim_id: Optional[str] = None
    event: Optional[str] = None
    task_name: Optional[str] = None
    target_x_px: Optional[float] = None
    target_y_px: Optional[float] = None


class CalibModel(BaseModel):
    model_type: Literal["poly2","homography","native"]
    params: dict
    fit_error_px: float | None = None


class TrackerInfo(BaseModel):
    name: str
    version: str
    reported_fps: float | None = None
