# calib/protocols.py


from dataclasses import dataclass
from typing import List


@dataclass
class Point:
    id: str
    x_norm: float
    y_norm: float
    dwell_ms: int = 1000
    gap_ms: int = 300


@dataclass
class Sequence:
    points: List[Point]




def generate_9pt_grid(dwell_ms: int = 1000, gap_ms: int = 300) -> Sequence:
    coords = [
    (0.5,0.5), (0.1,0.1), (0.9,0.1), (0.9,0.9), (0.1,0.9), (0.5,0.1), (0.9,0.5), (0.5,0.9), (0.1,0.5)
    ]
    pts = [Point(id=f"calib_{i+1:02}", x_norm=x, y_norm=y, dwell_ms=dwell_ms, gap_ms=gap_ms) for i,(x,y) in enumerate(coords)]
    return Sequence(points=pts)