# calib/validation.py


import numpy as np
from typing import Dict


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred - truth, axis=1)))


def rms_sd(points: np.ndarray) -> float:
    c = np.mean(points, axis=0)
    return float(np.sqrt(np.mean(np.sum((points - c)**2, axis=1))))


def drift(err_first: float, err_last: float) -> float:
    return float(err_last - err_first)