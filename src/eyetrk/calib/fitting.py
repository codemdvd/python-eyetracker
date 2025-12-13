from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .protocols import generate_9pt_grid
from ..core.calibrator import Poly2Fitter
from ..core.types import CalibModel


@dataclass
class FitOutput:
    model: CalibModel
    diag: Dict[str, float | int | bool]
    per_stim: pd.DataFrame | None


def fit_dataframe(
    df: pd.DataFrame,
    width: int,
    height: int,
    per_stim_median: bool = False,
) -> FitOutput:
    df = df.copy()
    df = df[df["validity"] == 0]
    df = df.dropna(subset=["x_norm", "y_norm"])
    raw_n = len(df)

    use_direct_targets = {"target_x_px", "target_y_px"}.issubset(df.columns)
    if use_direct_targets:
        df = df.dropna(subset=["stim_id", "target_x_px", "target_y_px"])
        df["stim_key"] = df["stim_id"].astype(str)
        df["target_x"] = df["target_x_px"]
        df["target_y"] = df["target_y_px"]
    else:
        df = df.dropna(subset=["stim_id"])
        smap = _stim_mapping(width, height)
        df["stim_norm"] = df["stim_id"].astype(str).apply(_base_stim_id)
        df = df[df["stim_norm"].apply(lambda s: s in smap)]
        df["target_x"] = df["stim_norm"].apply(lambda s: smap[s][0])
        df["target_y"] = df["stim_norm"].apply(lambda s: smap[s][1])
        df["stim_key"] = df["stim_norm"]

    if df.empty:
        raise RuntimeError("No valid calibration samples found (check stim_id and validity).")

    if per_stim_median:
        grp = (
            df.groupby("stim_key", as_index=False)[["x_norm", "y_norm", "target_x", "target_y"]]
            .median()
        )
        X = grp[["x_norm", "y_norm"]].to_numpy(dtype=float)
        Y = grp[["target_x", "target_y"]].to_numpy(dtype=float)
        used = grp
    else:
        X = df[["x_norm", "y_norm"]].to_numpy(dtype=float)
        Y = df[["target_x", "target_y"]].to_numpy(dtype=float)
        used = df

    model = Poly2Fitter().fit(X, Y)

    feats = np.column_stack(
        [
            np.ones(len(X)),
            X[:, 0],
            X[:, 1],
            X[:, 0] * X[:, 0],
            X[:, 0] * X[:, 1],
            X[:, 1] * X[:, 1],
        ]
    )
    coef_x = np.asarray(model.params["coef_x"], dtype=float)
    coef_y = np.asarray(model.params["coef_y"], dtype=float)
    ix = float(model.params["intercept_x"])
    iy = float(model.params["intercept_y"])
    pred_x = feats @ coef_x + ix
    pred_y = feats @ coef_y + iy

    err_x = pred_x - Y[:, 0]
    err_y = pred_y - Y[:, 1]
    err = np.sqrt(err_x**2 + err_y**2)

    def _r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    diag = {
        "r2_x": float(_r2(Y[:, 0], pred_x)),
        "r2_y": float(_r2(Y[:, 1], pred_y)),
        "rmse_px": float(np.sqrt(np.mean(err**2))),
        "mae_px": float(np.mean(np.abs(err))),
        "n_samples": int(len(X)),
        "n_raw_samples": int(raw_n),
        "per_stim_median": bool(per_stim_median),
    }

    try:
        used = used.copy()
        used["pred_x"] = pred_x
        used["pred_y"] = pred_y
        used["err_px"] = np.sqrt(
            (used["pred_x"] - used["target_x"]) ** 2 + (used["pred_y"] - used["target_y"]) ** 2
        )
        per_stim = (
            used.groupby("stim_key")["err_px"].agg(["count", "mean", "median", "max"]).reset_index()
        )
    except Exception:
        per_stim = None

    return FitOutput(model=model, diag=diag, per_stim=per_stim)


def _stim_mapping(width: int, height: int) -> Dict[str, Tuple[float, float]]:
    pts = generate_9pt_grid().points
    m: Dict[str, Tuple[float, float]] = {}
    for p in pts:
        m[p.id] = (float(p.x_norm * width), float(p.y_norm * height))
        m[f"{p.id}_retry"] = m[p.id]
    return m


def _base_stim_id(stim_id: str) -> str:
    return stim_id.replace("__", "_").replace("-", "_").split("_retry")[0]
