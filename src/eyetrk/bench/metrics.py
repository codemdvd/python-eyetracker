import numpy as np
from typing import Dict


def mean_absolute_error(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred - truth, axis=1)))


def precision_rms_sd(points: np.ndarray) -> float:
    c = np.mean(points, axis=0)
    return float(np.sqrt(np.mean(np.sum((points - c)**2, axis=1))))


def drop_rate(valid_mask: np.ndarray) -> float:
    return float(1.0 - np.mean(valid_mask))