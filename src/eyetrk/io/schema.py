from pydantic import BaseModel
from typing import Optional


class SessionMeta(BaseModel):
    session_id: str
    os: str
    browser: Optional[str] = None
    width_px: int
    height_px: int
    ppi: float
    distance_cm: float
    protocol: str
    dwell_ms: int
    refresh_hz: Optional[float] = None
    lighting: Optional[str] = None
    webcam_fps_reported: Optional[float] = None
    tracker_version: Optional[str] = None