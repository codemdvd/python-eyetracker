# core/tracker.py

from abc import ABC, abstractmethod
from typing import Callable

from .types import CalibModel, Sample, TrackerInfo


class Tracker(ABC):
    uses_internal_calibration: bool = False

    @abstractmethod
    def initialize(self, config: dict) -> TrackerInfo:
        ...

    @abstractmethod
    def start_stream(self, callback: Callable[[Sample], None], session_id: str | None = None) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    def on_event(self, event: str, payload: dict | None = None) -> None:
        """Optional: handle events like calib_point_start/end, calibration_done."""
        return None

    def set_external_model(self, model: CalibModel) -> None:
        return None
