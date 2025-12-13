# core/calibrator.py

import numpy as np
from .types import CalibModel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class Poly2Fitter:
    def fit(self, X: np.ndarray, Y: np.ndarray) -> CalibModel:
        poly = PolynomialFeatures(2, include_bias=True)
        Xd = poly.fit_transform(X)
        regx = LinearRegression().fit(Xd, Y[:,0])
        regy = LinearRegression().fit(Xd, Y[:,1])
        params = {
        "poly_features": poly.get_params(),
        "coef_x": regx.coef_.tolist(),
        "intercept_x": float(regx.intercept_),
        "coef_y": regy.coef_.tolist(),
        "intercept_y": float(regy.intercept_),
        }
        return CalibModel(model_type="poly2", params=params, fit_error_px=None)