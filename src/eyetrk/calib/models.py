# calib/models.py
from .protocols import Sequence
from ..core.types import CalibModel



def fit_homography(pupil_xy, screen_xy) -> CalibModel:

    return CalibModel(model_type="homography", params={}, fit_error_px=None)