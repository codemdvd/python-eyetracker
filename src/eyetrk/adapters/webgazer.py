from __future__ import annotations

from typing import Callable, Optional

from ..core.tracker import Tracker
from ..core.types import Sample, TrackerInfo, CalibModel


class WebGazerAdapter(Tracker):
    """
    Адаптер для WebGazer.

    ВАЖНО: WebGazer сам калибруется в браузере.
    Здесь мы:
      - просто принимаем x_norm, y_norm из веба,
      - переводим их в пиксели по размеру стимула,
      - логируем в общий пайплайн.

    Никаких внешних Poly2-моделей для webgazer больше НЕ применяем.
    """

    def __init__(self):
        self._cb: Optional[Callable[[Sample], None]] = None
        self._session_id: str | None = None
        self._out_w: int = 1280
        self._out_h: int = 720

    # ------------------------------------------------------------------ #
    def initialize(self, config: dict) -> TrackerInfo:
        self._out_w = int(config.get("out_width", self._out_w))
        self._out_h = int(config.get("out_height", self._out_h))
        version = config.get("version", "unknown")
        return TrackerInfo(name="webgazer", version=version)

    def set_external_model(self, model: CalibModel) -> None:
        """
        WebGazer использует свою встроенную браузерную калибровку.
        Внешнюю модель просто игнорируем.
        """
        return None

    def start_stream(self, callback: Callable[[Sample], None], session_id: str | None = None) -> None:
        self._cb = callback
        if session_id:
            self._session_id = session_id

    def stop(self) -> None:
        self._cb = None
        self._session_id = None

    def on_event(self, event: str, payload: dict | None = None) -> None:
        """
        Получаем события от Orchestrator (calib_point_start, calib_point_end, stim_on/off/move)
        и, для webgazer, пересылаем их в браузер через web_bridge.
        """
        payload = payload or {}
        # Нас интересуют в первую очередь калибровочные события, но можно слать и task-события
        interesting = {
            "calib_point_start",
            "calib_point_end",
            "stim_on",
            "stim_off",
            "stim_move",
            "task_start",
            "task_end",
        }
        if event not in interesting:
            return

        try:
            from eyetrk.web_bridge import server as bridge_server

            bridge_server.push_event(
                "webgazer",
                {
                    "type": event,
                    "payload": payload,
                },
            )
        except Exception:
            # не роняем пайплайн, если веб-мост не поднят
            return

    # ------------------------------------------------------------------ #
    def emit(self, sample: Sample):
        """
        Вызывается web_bridge-сервером, когда приходит сэмпл из браузера.
        """
        if self._session_id and not sample.session_id:
            sample.session_id = self._session_id

        if sample.x_norm is not None and sample.y_norm is not None:
            xp, yp = self._apply_model(sample.x_norm, sample.y_norm)
            sample.x_px = xp
            sample.y_px = yp
            if sample.validity is None:
                sample.validity = 0

        if self._cb:
            self._cb(sample)

    # ------------------------------------------------------------------ #
    def _apply_model(self, xn: float, yn: float) -> tuple[float, float]:
        """
        Никакой внешней модели: просто масштабируем нормированные координаты.
        Считаем, что webgazer выдает уже откалиброванный x_norm/y_norm.
        """
        return xn * self._out_w, yn * self._out_h
