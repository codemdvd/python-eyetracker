# core/tracker.py


from abc import ABC, abstractmethod
from typing import Callable, Iterable
from .types import Sample, CalibModel, TrackerInfo


class Tracker(ABC):
    @abstractmethod
    def initialize(self, config: dict) -> TrackerInfo: ...


    @abstractmethod
    def start_stream(self, callback: Callable[[Sample], None], session_id: str | None = None) -> None: ...


    @abstractmethod
    def stop(self) -> None: ...



    def on_event(self, event: str, payload: dict | None = None) -> None:
        """Optional: handle events like calib_point_start/end, calibration_done."""
        return None



    def set_external_model(self, model: CalibModel) -> None:
        return None
