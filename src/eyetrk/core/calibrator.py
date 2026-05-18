# core/calibrator.py

import math

import numpy as np
from .types import CalibModel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, RidgeCV


class Poly2Fitter:
    def __init__(
        self,
        alphas: tuple[float, ...] | None = None,
        candidate_degrees: tuple[int, ...] | None = None,
        simpler_within_ratio: float = 0.05,
    ):
        self.alphas = alphas or (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
        self.candidate_degrees = tuple(sorted(set(candidate_degrees or (1, 2))))
        self.simpler_within_ratio = max(0.0, float(simpler_within_ratio))

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        input_features: list[str] | None = None,
        degree: int | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> CalibModel:
        """Fit a polynomial model and select degree by leave-one-out CV when not fixed."""
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if input_features is None:
            input_features = [f"x{i}" for i in range(X.shape[1])]
        if len(input_features) != X.shape[1]:
            raise ValueError("input_features length must match X columns")
        weights = self._normalize_weights(sample_weight, len(X))
        if degree is None:
            degree, selection = self._select_degree(X, Y, weights)
        else:
            selection = self._degree_metrics(X, Y, int(degree), weights)

        poly = PolynomialFeatures(int(degree), include_bias=True)
        Xd = poly.fit_transform(X)
        regx = RidgeCV(alphas=self.alphas, fit_intercept=False).fit(Xd, Y[:, 0], sample_weight=weights)
        regy = RidgeCV(alphas=self.alphas, fit_intercept=False).fit(Xd, Y[:, 1], sample_weight=weights)
        params = {
            "poly_features": poly.get_params(),
            "input_features": list(input_features),
            "degree": int(degree),
            "powers": poly.powers_.tolist(),
            "coef_x": regx.coef_.tolist(),
            "intercept_x": 0.0,
            "coef_y": regy.coef_.tolist(),
            "intercept_y": 0.0,
            "ridge_alpha_x": float(regx.alpha_),
            "ridge_alpha_y": float(regy.alpha_),
            "candidate_degrees": list(self.candidate_degrees),
            "selection_cv_mae_px": float(selection["cv_mae_px"]),
            "selection_train_mae_px": float(selection["train_mae_px"]),
            "weighting_enabled": bool(weights is not None),
        }
        return CalibModel(model_type="poly2", params=params, fit_error_px=None)

    def _select_degree(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> tuple[int, dict[str, float]]:
        metrics_by_degree: dict[int, dict[str, float]] = {}
        for degree in self.candidate_degrees:
            if not self._is_degree_supported(degree, n_features=X.shape[1], n_samples=len(X)):
                continue
            metrics_by_degree[int(degree)] = self._degree_metrics(X, Y, int(degree), sample_weight)

        if not metrics_by_degree:
            fallback_degree = 1
            return fallback_degree, self._degree_metrics(X, Y, fallback_degree, sample_weight)

        best_cv = min(m["cv_mae_px"] for m in metrics_by_degree.values())
        tolerance = best_cv * (1.0 + self.simpler_within_ratio)
        eligible = [deg for deg, metric in metrics_by_degree.items() if metric["cv_mae_px"] <= tolerance]
        chosen_degree = min(eligible) if eligible else min(metrics_by_degree, key=lambda deg: metrics_by_degree[deg]["cv_mae_px"])
        return int(chosen_degree), metrics_by_degree[int(chosen_degree)]

    def _degree_metrics(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        degree: int,
        sample_weight: np.ndarray | None,
    ) -> dict[str, float]:
        poly = PolynomialFeatures(degree, include_bias=True)
        Xd = poly.fit_transform(X)
        regx = RidgeCV(alphas=self.alphas, fit_intercept=False).fit(Xd, Y[:, 0], sample_weight=sample_weight)
        regy = RidgeCV(alphas=self.alphas, fit_intercept=False).fit(Xd, Y[:, 1], sample_weight=sample_weight)
        pred_x = regx.predict(Xd)
        pred_y = regy.predict(Xd)
        train_err = np.sqrt((pred_x - Y[:, 0]) ** 2 + (pred_y - Y[:, 1]) ** 2)
        if sample_weight is None:
            train_mae = float(np.mean(train_err))
        else:
            train_mae = float(np.average(train_err, weights=sample_weight))
        cv_mae = self._loo_cv_mae(X, Y, degree, sample_weight)
        return {
            "cv_mae_px": float(cv_mae),
            "train_mae_px": train_mae,
        }

    def _loo_cv_mae(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        degree: int,
        sample_weight: np.ndarray | None,
    ) -> float:
        n = len(X)
        if n < 3:
            return float("inf")
        if not self._is_degree_supported(degree, n_features=X.shape[1], n_samples=n - 1):
            return float("inf")

        # Pre-select alpha on the full dataset so each LOO fold uses a consistent
        # regularization strength. Running RidgeCV inside every fold (n nested CVs
        # on n-1 points) is numerically noisy, especially for degree=2 where the
        # feature count is close to the sample count.
        poly_full = PolynomialFeatures(degree, include_bias=True)
        Xd_full = poly_full.fit_transform(X)
        regx_full = RidgeCV(alphas=self.alphas, fit_intercept=False).fit(
            Xd_full, Y[:, 0], sample_weight=sample_weight
        )
        regy_full = RidgeCV(alphas=self.alphas, fit_intercept=False).fit(
            Xd_full, Y[:, 1], sample_weight=sample_weight
        )
        alpha_x = regx_full.alpha_
        alpha_y = regy_full.alpha_

        errs: list[float] = []
        for hold_idx in range(n):
            mask = np.ones(n, dtype=bool)
            mask[hold_idx] = False
            X_train = X[mask]
            Y_train = Y[mask]
            w_train = sample_weight[mask] if sample_weight is not None else None
            poly = PolynomialFeatures(degree, include_bias=True)
            X_train_d = poly.fit_transform(X_train)
            regx = Ridge(alpha=alpha_x, fit_intercept=False).fit(X_train_d, Y_train[:, 0], sample_weight=w_train)
            regy = Ridge(alpha=alpha_y, fit_intercept=False).fit(X_train_d, Y_train[:, 1], sample_weight=w_train)
            X_test_d = poly.transform(X[hold_idx : hold_idx + 1])
            pred_x = float(regx.predict(X_test_d)[0])
            pred_y = float(regy.predict(X_test_d)[0])
            dx = pred_x - float(Y[hold_idx, 0])
            dy = pred_y - float(Y[hold_idx, 1])
            errs.append(math.sqrt(dx * dx + dy * dy))
        return float(np.mean(errs)) if errs else float("inf")

    def _is_degree_supported(self, degree: int, *, n_features: int, n_samples: int) -> bool:
        if degree < 1:
            return False
        n_terms = math.comb(n_features + degree, degree)
        return n_samples > n_terms

    def _normalize_weights(self, sample_weight: np.ndarray | None, n_rows: int) -> np.ndarray | None:
        if sample_weight is None:
            return None
        weights = np.asarray(sample_weight, dtype=float).reshape(-1)
        if len(weights) != n_rows:
            raise ValueError("sample_weight length must match X rows")
        if not np.all(np.isfinite(weights)):
            raise ValueError("sample_weight must be finite")
        weights = np.clip(weights, 1e-6, None)
        return weights
