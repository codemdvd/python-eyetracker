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
        (0.5, 0.5),   # center
        (0.1, 0.1),   # tl
        (0.9, 0.9),   # br
        (0.9, 0.1),   # tr
        (0.1, 0.9),   # bl
        (0.5, 0.1),   # top
        (0.5, 0.9),   # bottom
        (0.9, 0.5),   # right
        (0.1, 0.5),   # left
    ]
    pts = [Point(id=f"calib_{i+1:02}", x_norm=x, y_norm=y, dwell_ms=dwell_ms, gap_ms=gap_ms) for i,(x,y) in enumerate(coords)]
    return Sequence(points=pts)


def generate_25pt_grid(dwell_ms: int = 1000, gap_ms: int = 200) -> Sequence:
    """5×5 grid matching the fixation-grid task. Gives WebGazer training data at
    every task fixation position so its JS regression needn't extrapolate."""
    import random
    xs = [0.1, 0.3, 0.5, 0.7, 0.9]
    ys = [0.1, 0.3, 0.5, 0.7, 0.9]
    coords = [(x, y) for y in ys for x in xs]
    # Randomise order to avoid drift artefacts from reading-order fixation bias
    rng = random.Random(42)
    rng.shuffle(coords)
    pts = [Point(id=f"calib_{i+1:02}", x_norm=x, y_norm=y, dwell_ms=dwell_ms, gap_ms=gap_ms)
           for i, (x, y) in enumerate(coords)]
    return Sequence(points=pts)
