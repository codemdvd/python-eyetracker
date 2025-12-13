from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass
class EyeFeatures:
    x: float
    y: float
    w: float
    h: float
    blink: bool
    iris_xy: Tuple[float, float]


def _landmark_center(lm, indices: Sequence[int]) -> tuple[Optional[float], Optional[float]]:
    xs, ys = [], []
    for i in indices:
        if i < len(lm):
            xs.append(lm[i].x)
            ys.append(lm[i].y)
    if not xs:
        return None, None
    return float(np.mean(xs)), float(np.mean(ys))


def eye_from_landmarks(
    lm,
    *,
    iris_idx: Sequence[int],
    corners: Sequence[int],
    lids: Sequence[int],
    blink_ratio: float = 0.18,
) -> EyeFeatures | None:
    ix, iy = _landmark_center(lm, iris_idx)
    if ix is None or iy is None:
        return None

    outer = _landmark_center(lm, [corners[0]])
    inner = _landmark_center(lm, [corners[1]])
    upper = _landmark_center(lm, [lids[0]])
    lower = _landmark_center(lm, [lids[1]])
    if None in (*outer, *inner, *upper, *lower):
        return None

    ox, oy = outer
    inner_x, inner_y = inner
    _, upper_y = upper
    _, lower_y = lower

    horiz_span = inner_x - ox
    vert_span = lower_y - upper_y
    if abs(horiz_span) < 1e-4 or abs(vert_span) < 1e-4:
        return None

    x_rel = (ix - ox) / horiz_span
    y_rel = (iy - upper_y) / vert_span
    if horiz_span < 0:
        x_rel = 1.0 - x_rel

    # Eye aspect ratio is a cheap blink / squint indicator.
    ear = abs(vert_span) / abs(horiz_span)
    blink = ear < blink_ratio

    return EyeFeatures(
        x=float(max(0.0, min(1.0, x_rel))),
        y=float(max(0.0, min(1.0, y_rel))),
        w=float(abs(horiz_span)),
        h=float(abs(vert_span)),
        blink=blink,
        iris_xy=(float(ix), float(iy)),
    )


def aggregate_eyes(
    eyes: Iterable[EyeFeatures | None],
    *,
    min_eye_w: float,
    min_eye_h: float,
) -> tuple[Optional[float], Optional[float], float, int]:
    usable = []
    for eye in eyes:
        if eye is None or eye.blink:
            continue
        if eye.w < min_eye_w or eye.h < min_eye_h:
            continue
        quality = min(1.0, min(eye.w / max(min_eye_w, 1e-4), eye.h / max(min_eye_h, 1e-4)))
        usable.append((eye, quality))

    if not usable:
        return None, None, 0.0, 0

    total_q = sum(q for _, q in usable)
    if total_q <= 0:
        return None, None, 0.0, 0

    cx = sum(eye.x * q for eye, q in usable) / total_q
    cy = sum(eye.y * q for eye, q in usable) / total_q

    # Confidence grows with usable eyes and their geometric quality.
    conf = min(1.0, (total_q / len(usable)) * (1.0 if len(usable) == 2 else 0.8))

    return float(cx), float(cy), float(conf), len(usable)


def head_center(lm, indices: Sequence[int] = (33, 133, 362, 263, 1)) -> tuple[Optional[float], Optional[float]]:
    xs, ys = [], []
    for i in indices:
        if i < len(lm):
            xs.append(lm[i].x)
            ys.append(lm[i].y)
    if not xs:
        return None, None
    return float(np.mean(xs)), float(np.mean(ys))
