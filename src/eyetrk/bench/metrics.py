import math

import numpy as np


def mean_absolute_error(pred: np.ndarray, truth: np.ndarray) -> float:
    """Mean Euclidean distance between predicted and ground-truth gaze points, in pixels."""
    return float(np.mean(np.linalg.norm(pred - truth, axis=1)))


def root_mean_squared_error(pred: np.ndarray, truth: np.ndarray) -> float:
    """Root mean squared Euclidean error between predicted and ground-truth gaze, in pixels."""
    return float(np.sqrt(np.mean(np.sum((pred - truth) ** 2, axis=1))))


def precision_rms_sd(points: np.ndarray) -> float:
    """RMS spatial dispersion of a set of gaze points around their centroid — measures repeatability independent of accuracy."""
    c = np.mean(points, axis=0)
    return float(np.sqrt(np.mean(np.sum((points - c) ** 2, axis=1))))


def drop_rate(valid_mask: np.ndarray) -> float:
    """Fraction of frames where the tracker produced no valid gaze sample (validity != 0)."""
    return float(1.0 - np.mean(valid_mask))


def px_to_deg(px: float, *, ppi: float = 96.0, distance_cm: float = 60.0) -> float:
    """Convert a pixel distance to degrees of visual angle.

    Uses the standard formula: α = 2·atan(size_cm / (2·distance_cm)).
    ppi is the logical (CSS) pixels-per-inch reported by the display session.
    """
    cm_per_px = 2.54 / ppi
    return float(math.degrees(2.0 * math.atan(px * cm_per_px / 2.0 / distance_cm)))