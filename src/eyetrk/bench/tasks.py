from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass
class Event:
    """One step in a task timeline: type is 'stim_on'/'stim_off'/'stim_move'/'wait_ms', payload carries coordinates or duration."""
    type: str
    payload: dict


@dataclass
class Task:
    """A named benchmark task with an ordered list of stimulus events to play back."""
    name: str
    timeline: List[Event]


GRID_POINTS = [
    (x, y)
    for y in (0.1, 0.3, 0.5, 0.7, 0.9)
    for x in (0.1, 0.3, 0.5, 0.7, 0.9)
]


def fixation_grid(dwell_ms: int = 800, gap_ms: int = 200, repeats: int = 2) -> Task:
    """5×5 grid of 25 stationary fixation targets. Each point shown for dwell_ms, repeated `repeats` times."""
    events: List[Event] = []
    rep = max(1, int(repeats))
    for r in range(rep):
        for idx, (x, y) in enumerate(GRID_POINTS, start=1):
            stim_id = f"fix_r{r+1}_{idx:02}"
            events.append(Event("stim_on", {"id": stim_id, "x_norm": x, "y_norm": y}))
            events.append(Event("wait_ms", {"ms": dwell_ms}))
            events.append(Event("stim_off", {"id": stim_id}))
            events.append(Event("wait_ms", {"ms": gap_ms}))
    return Task(name="fixation-grid", timeline=events)


def step_saccades(
    cycles: int = 12,
    dwell_ms: int = 600,
    gap_ms: int = 250,
    left_x: float = 0.2,
    right_x: float = 0.8,
    y: float = 0.5,
) -> Task:
    """Horizontal left–right alternating targets at fixed y. Tests saccadic eye movement response speed and accuracy."""
    events: List[Event] = []
    stim_idx = 1
    for cycle in range(cycles):
        for label, x in (("L", left_x), ("R", right_x)):
            stim_id = f"sacc_{stim_idx:02}_{label}"
            events.append(Event("stim_on", {"id": stim_id, "x_norm": x, "y_norm": y}))
            events.append(Event("wait_ms", {"ms": dwell_ms}))
            events.append(Event("stim_off", {"id": stim_id}))
            events.append(Event("wait_ms", {"ms": gap_ms}))
            stim_idx += 1
    return Task(name="step-saccades", timeline=events)


def smooth_pursuit(
    revolutions: int = 4,
    seconds_per_rev: float = 4.0,
    fps: int = 60,
    radius: float = 0.25,
    center_x: float = 0.5,
    center_y: float = 0.5,
    laps: int = 2,
) -> Task:
    """Circular moving target. Tests smooth pursuit tracking ability — how well the tracker follows continuous motion."""
    events: List[Event] = []
    stim_id = "pursuit_circle"
    step_ms = max(10, int(1000 / max(1, fps)))
    total_steps = max(1, int(revolutions * seconds_per_rev * fps))

    def point(angle: float) -> tuple[float, float]:
        return (
            center_x + radius * math.cos(angle),
            center_y + radius * math.sin(angle),
        )

    start_x, start_y = point(0.0)
    events.append(Event("stim_on", {"id": stim_id, "x_norm": start_x, "y_norm": start_y}))
    events.append(Event("wait_ms", {"ms": 300}))

    for lap in range(max(1, int(laps))):
        for step in range(1, total_steps + 1):
            angle = 2.0 * math.pi * (step / (seconds_per_rev * fps))
            x, y = point(angle)
            events.append(Event("stim_move", {"id": f"{stim_id}_{lap}", "x_norm": x, "y_norm": y}))
            events.append(Event("wait_ms", {"ms": step_ms}))
        events.append(Event("wait_ms", {"ms": 300}))

    events.append(Event("stim_off", {"id": stim_id}))
    return Task(name="smooth-pursuit", timeline=events)


TASK_REGISTRY: Dict[str, Callable[[], Task]] = {
    "fixation-grid": fixation_grid,
    "step-saccades": step_saccades,
    "smooth-pursuit": smooth_pursuit,
}


def resolve_tasks(names: List[str]) -> List[Task]:
    """Look up task names in TASK_REGISTRY and return instantiated Task objects. Raises ValueError for unknown names."""
    tasks: List[Task] = []
    for name in names:
        factory = TASK_REGISTRY.get(name)
        if not factory:
            raise ValueError(f"Unknown benchmark task: {name}")
        tasks.append(factory())
    return tasks
