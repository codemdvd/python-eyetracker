from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..core.tracker import Tracker
from ..core.types import Sample, TrackerInfo, CalibModel


class TurkerGazeAdapter(Tracker):
    """
    Browser-based tracker using the TurkerGaze JS library.

    Поток данных:
      JS (turkergaze_bridge.html) ->
      FastAPI WebSocket bridge (web_bridge/server.py) ->
      adapters_registry["turkergaze"].emit(Sample(...))

    Мы здесь:
      - принимаем Sample из web_bridge через .emit(...)
      - при наличии внешней калибровочной модели (poly2) применяем её
      - если модели нет — просто масштабируем x_norm,y_norm в пиксели экрана.
    """

    def __init__(self):
        self._cb: Optional[Callable[[Sample], None]] = None
        self._session_id: str | None = None
        self._model: CalibModel | None = None
        self._out_w: int = 1280
        self._out_h: int = 720

    # ------------------------------------------------------------------ #
    def initialize(self, config: dict) -> TrackerInfo:
        """
        Вызывается cli._prepare_adapters при старте.

        config может содержать:
          - out_width / out_height: логическое разрешение экрана, в которое
            мы проецируем сырые нормализованные координаты.
        """
        self._out_w = int(config.get("out_width", self._out_w))
        self._out_h = int(config.get("out_height", self._out_h))
        version = config.get("version", "unknown")
        return TrackerInfo(name="turkergaze", version=version)

    def set_external_model(self, model: CalibModel) -> None:
        """
        Позволяет загрузить нашу poly2-модель (если мы её всё-таки хотим
        поверх TurkerGaze использовать). Можно игнорировать, задав self._model = None.
        """
        self._model = model

    def start_stream(self, callback: Callable[[Sample], None], session_id: str | None = None) -> None:
        """
        Никакого собственного потока мы здесь не запускаем — данные приходят
        из web_bridge по WebSocket. Нам нужно только запомнить callback.
        """
        self._cb = callback
        if session_id:
            self._session_id = session_id

    def stop(self) -> None:
        """
        Для браузерных трекеров особой остановки не нужно: достаточно
        перестать обрабатывать коллбек.
        """
        self._cb = None
        self._session_id = None

    def on_event(self, event: str, payload: dict | None = None) -> None:
        # На события (calib_point_start/end, task_start, ...) можно не реагировать.
        return

    # ------------------------------------------------------------------ #
    def emit(self, sample: Sample) -> None:
        """
        Вызвается из web_bridge.server, когда приходят данные по WS.

        Здесь:
          - проставляем session_id, если его нет
          - при необходимости применяем модель
          - вызываем self._cb(sample), чтобы Orchestrator получил данные.
        """
        if self._session_id and not sample.session_id:
            sample.session_id = self._session_id

        if sample.x_norm is not None and sample.y_norm is not None:
            xp, yp = self._apply_model(sample.x_norm, sample.y_norm)
            sample.x_px = xp
            sample.y_px = yp
            # По умолчанию считаем выборку валидной, если validity не задана.
            if sample.validity is None:
                sample.validity = 0

        if self._cb:
            self._cb(sample)

    # ------------------------------------------------------------------ #
    def _apply_model(self, xn: float, yn: float) -> tuple[Optional[float], Optional[float]]:
        """
        Применяем либо нашу poly2-модель, либо простой linear scaling.
        """
        # 1) Если внешнюю модель решили не использовать — просто масштабируем.
        if self._model is None:
            return xn * self._out_w, yn * self._out_h

        # 2) Poly2 — тот же формат, что мы используем в Poly2Fitter.
        if self._model.model_type == "poly2":
            try:
                coef_x = np.array(self._model.params["coef_x"], dtype=float)
                coef_y = np.array(self._model.params["coef_y"], dtype=float)
                ix = float(self._model.params["intercept_x"])
                iy = float(self._model.params["intercept_y"])
            except Exception:
                return None, None

            x1 = float(xn)
            x2 = float(yn)
            feats = np.array([1.0, x1, x2, x1 * x1, x1 * x2, x2 * x2], dtype=float)
            xp = float(np.dot(feats, coef_x) + ix)
            yp = float(np.dot(feats, coef_y) + iy)
            return xp, yp

        # Другие типы моделей пока не поддерживаем
        return None, None
