from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..core.types import Sample


@dataclass
class TimelineState:
    task_name: str | None = None
    stim_id: str | None = None
    target_x_px: float | None = None
    target_y_px: float | None = None


def load_timeline(path: Path) -> list[dict]:
    events: list[dict] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            item = json.loads(text)
        except Exception:
            continue
        if isinstance(item, dict):
            events.append(item)
    events.sort(key=lambda item: int(item.get("t_ms", 0)))
    return events


def load_video_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_frame_labels(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if "frame_id" not in df.columns:
        return {}
    frame_labels: dict[int, dict] = {}
    subset_cols = [c for c in ("frame_id", "task_name", "stim_id", "target_x_px", "target_y_px") if c in df.columns]
    if "frame_id" not in subset_cols:
        return {}
    for row in df[subset_cols].itertuples(index=False):
        try:
            frame_id = int(getattr(row, "frame_id"))
        except Exception:
            continue
        frame_labels[frame_id] = {
            "task_name": _as_text(getattr(row, "task_name", None)),
            "stim_id": _as_text(getattr(row, "stim_id", None)),
            "target_x_px": _as_float(getattr(row, "target_x_px", None)),
            "target_y_px": _as_float(getattr(row, "target_y_px", None)),
        }
    return frame_labels


class SessionTimelineLabeler:
    def __init__(self, events: Iterable[dict]):
        self._events = list(events)
        self._index = 0
        self._state = TimelineState()

    def apply(self, sample: Sample) -> Sample:
        ts = int(getattr(sample, "timestamp_ms", 0) or 0)
        while self._index < len(self._events):
            event = self._events[self._index]
            event_ts = int(event.get("t_ms", 0) or 0)
            if event_ts > ts:
                break
            self._consume_event(event)
            self._index += 1

        if self._state.task_name and not sample.task_name:
            sample.task_name = self._state.task_name
        if self._state.stim_id and not sample.stim_id:
            sample.stim_id = self._state.stim_id
        if self._state.target_x_px is not None:
            sample.target_x_px = self._state.target_x_px
        if self._state.target_y_px is not None:
            sample.target_y_px = self._state.target_y_px
        return sample

    def _consume_event(self, event: dict) -> None:
        kind = str(event.get("event", ""))
        if kind in {"task_start"}:
            self._state.task_name = _as_text(event.get("task"))
            return
        if kind in {"task_end"}:
            self._state.task_name = None
            self._state.stim_id = None
            self._state.target_x_px = None
            self._state.target_y_px = None
            return
        if kind == "stim":
            self._state.stim_id = _as_text(event.get("stim_id"))
            self._state.target_x_px = _as_float(event.get("target_x_px"))
            self._state.target_y_px = _as_float(event.get("target_y_px"))
            if event.get("task_name") is not None:
                self._state.task_name = _as_text(event.get("task_name"))
            return
        if kind == "stim_off":
            self._state.stim_id = None
            self._state.target_x_px = None
            self._state.target_y_px = None
            return
        if kind == "calibration_start":
            self._state.task_name = "calibration"
            return
        if kind == "calibration_done":
            self._state.task_name = None
            self._state.stim_id = None
            self._state.target_x_px = None
            self._state.target_y_px = None
            return


class FrameLabeler:
    def __init__(self, frame_labels: dict[int, dict]):
        self._frame_labels = dict(frame_labels)

    def apply(self, sample: Sample) -> Sample:
        frame_id = getattr(sample, "frame_id", None)
        if frame_id is None:
            return sample
        label = self._frame_labels.get(int(frame_id))
        if not label:
            return sample
        if label.get("task_name") is not None:
            sample.task_name = label["task_name"]
        if label.get("stim_id") is not None:
            sample.stim_id = label["stim_id"]
        if label.get("target_x_px") is not None:
            sample.target_x_px = label["target_x_px"]
        if label.get("target_y_px") is not None:
            sample.target_y_px = label["target_y_px"]
        return sample


def _as_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except Exception:
        return None
