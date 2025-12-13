# calib/acceptance.py
from __future__ import annotations
from typing import Iterable, Optional
import math


def acceptance(
    samples: Iterable,
    *,
    screen_w_px: int,
    screen_h_px: int,
    min_valid_ms: int = 800,        # было 600
    max_disp_px: float = 40.0,
    min_valid_rate: float = 0.80,   # было 0.70
    target_xy_px: Optional[tuple[float,float]] = None,
    max_offset_px: float = 80.0,    # новый критерий
) -> bool:
    samples = list(samples)
    if not samples:
        return False

    valid = [s for s in samples if getattr(s, "validity", 0) == 0]
    if not valid:
        return False

    tvals = sorted(getattr(s, "timestamp_ms", 0) for s in valid)
    valid_ms = (tvals[-1] - tvals[0]) if len(tvals) > 1 else 0

    def xy_px(s):
        x_px = getattr(s, "x_px", None)
        y_px = getattr(s, "y_px", None)
        if x_px is not None and y_px is not None:
            return float(x_px), float(y_px)
        xn = getattr(s, "x_norm", None)
        yn = getattr(s, "y_norm", None)
        if xn is None or yn is None:
            return None
        return float(xn) * screen_w_px, float(yn) * screen_h_px

    coords = [xy_px(s) for s in valid]
    coords = [c for c in coords if c is not None]
    if len(coords) < 3:
        return False

    cx = sum(c[0] for c in coords) / len(coords)
    cy = sum(c[1] for c in coords) / len(coords)
    sq = [(c[0] - cx) ** 2 + (c[1] - cy) ** 2 for c in coords]
    rms_sd = math.sqrt(sum(sq) / len(sq))

    # новый критерий: близость к целевой точке
    offset_ok = True
    if target_xy_px is not None:
        tx, ty = target_xy_px
        dx = cx - tx
        dy = cy - ty
        offset = math.hypot(dx, dy)
        offset_ok = (offset <= max_offset_px)

    valid_rate = len(valid) / len(samples)

    return (valid_ms >= min_valid_ms) and (rms_sd <= max_disp_px) and (valid_rate >= min_valid_rate) and offset_ok