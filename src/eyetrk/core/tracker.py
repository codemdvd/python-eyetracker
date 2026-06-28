# core/tracker.py

from abc import ABC, abstractmethod
from typing import Callable

from .types import CalibModel, Sample, TrackerInfo


class Tracker(ABC):
    """Abstract base class every tracker adapter must implement. One instance per tracker per session."""

    uses_internal_calibration: bool = False
    """True for browser trackers that run their own calibration wizard (GazeRecorder). False for webcam trackers."""

    @abstractmethod
    def initialize(self, config: dict) -> TrackerInfo:
        """Apply config dict and return tracker metadata. Called once before start_stream."""
        ...

    @abstractmethod
    def start_stream(self, callback: Callable[[Sample], None], session_id: str | None = None) -> None:
        """Launch background thread/loop. Calls callback(sample) for every captured frame."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Signal the background thread to exit and block until it joins."""
        ...

    def on_event(self, event: str, payload: dict | None = None) -> None:
        """React to calibration lifecycle events (calib_point_start, calibration_done, tasks_done, etc.). Optional."""
        return None

    def set_external_model(self, model: CalibModel) -> None:
        """Inject a polynomial model fitted offline. The tracker uses it to convert raw gaze to x_px/y_px."""
        return None
