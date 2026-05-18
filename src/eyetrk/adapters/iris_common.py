from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import cv2
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


# ---- 3D head pose via solvePnP -----------------------------------------
#
# Generic 3D face model (nose tip at origin, mm scale).
# Coordinates: +x = right in image (after cam un-mirror), +y = up, +z = toward camera.
# Landmark correspondence (MediaPipe Face Mesh, un-mirrored frame):
#   lm[1]   nose tip
#   lm[152] chin
#   lm[33]  right eye outer corner  (right side of un-mirrored image)
#   lm[263] left eye outer corner   (left side of un-mirrored image)
#   lm[61]  right mouth corner
#   lm[291] left mouth corner
_FACE_3D_PNP = np.array([
    [  0.0,   0.0,   0.0],
    [  0.0, -63.6, -12.5],
    [ 43.3,  32.7, -26.0],
    [-43.3,  32.7, -26.0],
    [ 28.9, -28.9, -24.1],
    [-28.9, -28.9, -24.1],
], dtype=np.float64)
_PNP_LM = [1, 152, 33, 263, 61, 291]


def estimate_head_pose_pnp(
    lm, img_w: int, img_h: int
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (R 3×3, tvec) from solvePnP with an approximate camera model, or (None, None)."""
    try:
        pts_2d = np.array(
            [[lm[i].x * img_w, lm[i].y * img_h] for i in _PNP_LM], dtype=np.float64
        )
        f = float(img_w)
        K = np.array([[f, 0.0, img_w * 0.5], [0.0, f, img_h * 0.5], [0.0, 0.0, 1.0]])
        ok, rvec, tvec = cv2.solvePnP(
            _FACE_3D_PNP, pts_2d, K, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return None, None
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec
    except Exception:
        return None, None


def iris_gaze_hf(
    lm, R: np.ndarray, img_w: int, img_h: int
) -> tuple[Optional[float], Optional[float]]:
    """
    Head-frame gaze direction from the average of both iris landmarks (468, 473).

    Projects the iris midpoint as a ray in camera frame, then un-rotates by R.T
    to get the gaze direction in head frame — rotation-invariant by construction.
    Returns (gaze_x, gaze_y) clamped to [-1, 1], or (None, None) on failure.
    """
    try:
        ix = ((lm[468].x + lm[473].x) * 0.5) * img_w
        iy = ((lm[468].y + lm[473].y) * 0.5) * img_h
        f = float(img_w)
        d_cam = np.array([(ix - img_w * 0.5) / f, (iy - img_h * 0.5) / f, 1.0])
        d_cam /= np.linalg.norm(d_cam)
        d_head = R.T @ d_cam
        if abs(d_head[2]) < 1e-6:
            return None, None
        gx = float(d_head[0] / d_head[2])
        gy = float(-d_head[1] / d_head[2])
        return float(max(-1.0, min(1.0, gx))), float(max(-1.0, min(1.0, gy)))
    except Exception:
        return None, None
