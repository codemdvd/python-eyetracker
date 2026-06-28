from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..bench.tasks import TASK_REGISTRY
from .protocols import generate_9pt_grid
from ..core.calibrator import Poly2Fitter
from ..core.types import CalibModel


@dataclass
class FitOutput:
    """Result of fit_dataframe(): the fitted model, diagnostic stats dict, and per-stimulus error breakdown."""
    model: CalibModel
    diag: Dict[str, float | int | bool]
    per_stim: pd.DataFrame | None


def fit_dataframe(
    df: pd.DataFrame,
    width: int,
    height: int,
    per_stim_median: bool = False,
    allow_extra_features: bool = False,
    validation_df: pd.DataFrame | None = None,
) -> FitOutput:
    """Fit a polynomial gaze correction model from calibration samples.

    Tries multiple input feature sets (e.g. [x_norm, y_norm], [iris_from_head_x, y, pitch])
    and polynomial degrees (1–2), selects the best via leave-one-out CV MAE.
    If validation_df (task samples with known targets) is provided, ranks candidates by
    transfer accuracy instead of training accuracy.
    Returns FitOutput with the winning CalibModel, diagnostics, and per-point error stats.
    """
    df = df.copy()
    df = df[df["validity"] == 0]
    df = df.dropna(subset=["x_norm", "y_norm"])
    raw_n = len(df)
    tracker_name = None
    if "tracker_id" in df.columns:
        trackers = [str(v) for v in df["tracker_id"].dropna().astype(str).unique().tolist()]
        if len(trackers) == 1:
            tracker_name = trackers[0]
    if "stim_id" in df.columns:
        df["stim_id"] = df["stim_id"].astype(str)
        df = df[df["stim_id"].str.startswith("calib_")]

    use_direct_targets = {"target_x_px", "target_y_px"}.issubset(df.columns)
    if use_direct_targets:
        df = df.dropna(subset=["stim_id", "target_x_px", "target_y_px"])
        df["stim_key"] = df["stim_id"].astype(str).apply(_base_stim_id)
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

    df = _select_best_calibration_attempts(df, width=width, height=height)
    if "stim_id" in df.columns:
        df["stim_key"] = df["stim_id"].astype(str).apply(_base_stim_id)
    # For GazeRecorder the cloud model should already be calibrated after ShowCalibration.
    # Stims where the raw (pre-correction) gaze is far from the target indicate the
    # GR model was still updating during that stim (online learning artifact).
    # Exclude such stims before polynomial fitting.
    if tracker_name == "gazerecorder":
        df = _filter_gr_outlier_stims(df, width=width, height=height)

    df = _derive_features(df)
    if validation_df is not None:
        validation_df = _derive_features(validation_df)

    candidate_feature_sets = _candidate_feature_sets(
        df,
        tracker_name=tracker_name,
        allow_extra_features=allow_extra_features,
    )

    chosen = None
    for feature_cols in candidate_feature_sets:
        try:
            candidate = _fit_with_features(
                df,
                feature_cols=feature_cols,
                width=width,
                height=height,
                per_stim_median=per_stim_median,
            )
        except RuntimeError:
            continue
        validation_metrics = _score_validation_tasks(
            validation_df,
            model=candidate["model"],
            feature_cols=feature_cols,
        )
        if validation_metrics is not None:
            candidate["validation"] = validation_metrics
            rank = (
                float(validation_metrics["validation_task_mae_px"]),
                float(validation_metrics["validation_task_rmse_px"]),
                abs(float(validation_metrics["validation_task_bias_y_px"])),
                abs(float(validation_metrics["validation_task_bias_x_px"])),
                float(candidate["model"].params.get("selection_cv_mae_px", float("inf"))),
                float(candidate["err"].mean()),
            )
        else:
            rank = (
                float(candidate["model"].params.get("selection_cv_mae_px", float("inf"))),
                float(candidate["err"].mean()),
                float(np.sqrt(np.mean(candidate["err"] ** 2))),
            )
        if chosen is None or rank < chosen["rank"]:
            chosen = {**candidate, "rank": rank}

    if chosen is None:
        raise RuntimeError("No usable calibration rows after filtering.")

    feature_cols = chosen["feature_cols"]
    agg = chosen["agg"]
    used = chosen["used"]
    X = chosen["X"]
    Y = chosen["Y"]
    model = chosen["model"]
    pred_x = chosen["pred_x"]
    pred_y = chosen["pred_y"]
    err = chosen["err"]

    err_x = pred_x - Y[:, 0]
    err_y = pred_y - Y[:, 1]

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
        "n_points": int(used["stim_key"].nunique()) if "stim_key" in used.columns else int(len(X)),
        "per_stim_median": bool(per_stim_median),
        "input_features": list(feature_cols),
        "selected_degree": int(model.params.get("degree", 2)),
        "ridge_alpha_x": float(model.params.get("ridge_alpha_x", 0.0)),
        "ridge_alpha_y": float(model.params.get("ridge_alpha_y", 0.0)),
        "selection_cv_mae_px": float(model.params.get("selection_cv_mae_px", 0.0)),
        "selection_train_mae_px": float(model.params.get("selection_train_mae_px", 0.0)),
        "selection_mode": "validation-task-transfer" if chosen.get("validation") is not None else "calibration-only",
    }
    validation_metrics = chosen.get("validation")
    if validation_metrics is not None:
        diag.update(validation_metrics)
    if agg is not None and not agg.empty:
        diag.update(
            {
                "robust_point_fit": True,
                "trimmed_raw_samples": int(agg["trimmed_rows"].sum()),
                "kept_raw_samples": int(agg["kept_rows"].sum()),
                "mean_point_dispersion_px": float(agg["dispersion_px"].mean()),
                "max_point_dispersion_px": float(agg["dispersion_px"].max()),
            }
        )

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


def _derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features from existing columns in-place (returns copy)."""
    df = df.copy()
    # Head-relative iris position: iris_abs minus face centre.
    # Translation-invariant across sessions — if the user sits at a different
    # x/y position between calibration and task recordings, both iris_abs and
    # head_x shift by the same amount, so their difference is stable.
    if all(c in df.columns for c in ("iris_abs_x_norm", "head_x")):
        df["iris_from_head_x_norm"] = df["iris_abs_x_norm"] - df["head_x"]
    if all(c in df.columns for c in ("iris_abs_y_norm", "head_y")):
        df["iris_from_head_y_norm"] = df["iris_abs_y_norm"] - df["head_y"]
    return df


def _candidate_feature_sets(
    df: pd.DataFrame,
    *,
    tracker_name: str | None,
    allow_extra_features: bool,
) -> list[list[str]]:
    """Return ordered list of feature column sets to try for this tracker. Tracker-specific curated sets come after the universal [x_norm, y_norm] baseline."""
    candidates: list[list[str]] = [["x_norm", "y_norm"]]
    if tracker_name == "mpiris":
        curated = [
            # 2-feature sets — allow degree=2 (6 terms, feasible with 9 calib pts)
            ["raw_x_norm", "raw_y_norm"],
            ["corrected_x_norm", "corrected_y_norm"],
            # iris_from_head: translation-invariant, large signal range — preferred over iris_abs
            ["iris_from_head_x_norm", "iris_from_head_y_norm"],
            # 3D head-frame gaze: rotation-invariant by construction (R.T @ d_cam)
            ["gaze_x_hf", "gaze_y_hf"],
            ["gaze_x_hf", "gaze_y_hf", "head_x", "head_y"],
            # 3-feature sets with head angles — pitch encodes look-up/down compensation
            ["iris_from_head_x_norm", "iris_from_head_y_norm", "pitch"],
            ["raw_x_norm", "raw_y_norm", "pitch"],
            ["raw_x_norm", "raw_y_norm", "yaw", "pitch"],
            # legacy head-translation features (kept for backwards compat)
            ["raw_x_norm", "raw_y_norm", "head_y"],
            ["corrected_x_norm", "corrected_y_norm", "head_y"],
            ["raw_x_norm", "raw_y_norm", "head_x", "head_y"],
        ]
        for cols in curated:
            if all(col in df.columns and df[col].notna().sum() >= max(9, int(len(df) * 0.4)) for col in cols):
                candidates.append(cols)
    if tracker_name == "optimeyes":
        curated = [
            # 2-feature sets — degree=2 is feasible (6 terms with 9 calib pts)
            ["right_eye_x_norm", "right_eye_h_norm"],
            # iris_from_head: translation-invariant large-signal-range gaze proxy
            ["iris_from_head_x_norm", "iris_from_head_y_norm"],
            # 3-feature sets
            ["right_eye_x_norm", "right_eye_y_norm"],
            ["iris_from_head_x_norm", "iris_from_head_y_norm", "pitch"],
            ["right_eye_x_norm", "right_eye_h_norm", "pitch"],
            ["right_eye_x_norm", "right_eye_y_norm", "right_eye_w_norm"],
            ["right_eye_x_norm", "right_eye_w_norm"],
            # 4-feature sets (degree=2 not feasible, degree=1 only)
            ["right_eye_x_norm", "right_eye_w_norm", "right_eye_h_norm"],
            ["right_eye_x_norm", "right_eye_y_norm", "right_eye_h_norm"],
            ["right_eye_x_norm", "right_eye_y_norm", "right_eye_w_norm", "right_eye_h_norm"],
            ["right_eye_x_norm", "right_eye_y_norm", "right_eye_w_norm", "pitch"],
            ["right_eye_x_norm", "right_eye_w_norm", "right_eye_h_norm", "pitch"],
        ]
        for cols in curated:
            if all(col in df.columns and df[col].notna().sum() >= max(9, int(len(df) * 0.4)) for col in cols):
                candidates.append(cols)
        return candidates

    if allow_extra_features and df["stim_key"].nunique() >= 18:
        feature_candidates = ["head_x", "head_y", "head_z", "yaw", "pitch", "roll"]
        extra = [
            f
            for f in feature_candidates
            if f in df.columns and df[f].notna().sum() >= max(18, int(len(df) * 0.7))
        ]
        if extra:
            candidates.append(["x_norm", "y_norm"] + extra)
    return candidates


def _fit_with_features(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    width: int,
    height: int,
    per_stim_median: bool,
) -> dict:
    """Fit one candidate model using the given feature columns. Returns a dict with model, predictions, and per-sample errors."""
    agg = None
    if per_stim_median:
        agg = _aggregate_calibration_points(df, feature_cols, width=width, height=height)
        agg = agg.dropna(subset=feature_cols + ["target_x", "target_y"])
        X = agg[feature_cols].to_numpy(dtype=float)
        Y = agg[["target_x", "target_y"]].to_numpy(dtype=float)
        used = agg
    else:
        work = df.dropna(subset=feature_cols + ["target_x", "target_y"])
        X = work[feature_cols].to_numpy(dtype=float)
        Y = work[["target_x", "target_y"]].to_numpy(dtype=float)
        used = work
    if X.size == 0 or Y.size == 0:
        raise RuntimeError("No usable calibration rows after filtering.")

    model = Poly2Fitter().fit(X, Y, input_features=feature_cols)
    powers = np.asarray(model.params["powers"], dtype=int)
    feats = np.ones((len(X), len(powers)), dtype=float)
    for term_idx, pow_vec in enumerate(powers):
        term = np.ones(len(X), dtype=float)
        for col_idx, pwr in enumerate(pow_vec):
            if pwr == 0:
                continue
            term *= np.power(X[:, col_idx], pwr)
        feats[:, term_idx] = term

    coef_x = np.asarray(model.params["coef_x"], dtype=float)
    coef_y = np.asarray(model.params["coef_y"], dtype=float)
    ix = float(model.params["intercept_x"])
    iy = float(model.params["intercept_y"])
    pred_x = feats @ coef_x + ix
    pred_y = feats @ coef_y + iy
    err = np.sqrt((pred_x - Y[:, 0]) ** 2 + (pred_y - Y[:, 1]) ** 2)
    return {
        "feature_cols": feature_cols,
        "agg": agg,
        "used": used,
        "X": X,
        "Y": Y,
        "model": model,
        "pred_x": pred_x,
        "pred_y": pred_y,
        "err": err,
    }


def _stim_mapping(width: int, height: int) -> Dict[str, Tuple[float, float]]:
    """Build a dict from stim_id (e.g. 'calib_01') to target pixel coordinates for the 9-point grid."""
    pts = generate_9pt_grid().points
    m: Dict[str, Tuple[float, float]] = {}
    for p in pts:
        m[p.id] = (float(p.x_norm * width), float(p.y_norm * height))
        m[f"{p.id}_retry"] = m[p.id]
    return m


def _base_stim_id(stim_id: str) -> str:
    """Strip retry suffixes from a stim_id to get the canonical base name (e.g. 'calib_01_retry' → 'calib_01')."""
    return stim_id.replace("__", "_").replace("-", "_").split("_retry")[0]


def _select_best_calibration_attempts(df: pd.DataFrame, *, width: int, height: int) -> pd.DataFrame:
    """For each calibration point that has multiple attempts (original + retries), keep the attempt with the lowest spatial dispersion."""
    if "stim_id" not in df.columns or df.empty:
        return df

    work = df.copy()
    work["stim_id"] = work["stim_id"].astype(str)
    work["stim_base"] = work["stim_id"].apply(_base_stim_id)
    calib_mask = work["stim_base"].str.startswith("calib_")
    if not calib_mask.any():
        return work.drop(columns=["stim_base"], errors="ignore")

    calib_df = work.loc[calib_mask].copy()

    scored_attempts: list[tuple[str, str, float, int, int]] = []
    for stim_id, grp in calib_df.groupby("stim_id"):
        coords = grp[["x_norm", "y_norm"]].dropna()
        if len(coords) < 3:
            dispersion = float("inf")
        else:
            x_px = coords["x_norm"].to_numpy(dtype=float) * width
            y_px = coords["y_norm"].to_numpy(dtype=float) * height
            cx = float(np.median(x_px))
            cy = float(np.median(y_px))
            dispersion = float(np.sqrt(np.mean((x_px - cx) ** 2 + (y_px - cy) ** 2)))
        retry_rank = 1 if "_retry" in stim_id else 0
        scored_attempts.append((grp["stim_base"].iloc[0], stim_id, dispersion, len(coords), retry_rank))

    chosen_ids: set[str] = set()
    for stim_base, grp in pd.DataFrame(
        scored_attempts,
        columns=["stim_base", "stim_id", "dispersion", "valid_count", "retry_rank"],
    ).groupby("stim_base"):
        ranked = grp.sort_values(
            by=["dispersion", "valid_count", "retry_rank"],
            ascending=[True, False, False],
            kind="stable",
        )
        chosen_ids.add(str(ranked.iloc[0]["stim_id"]))

    kept = calib_df[calib_df["stim_id"].isin(chosen_ids)].copy()
    kept["stim_key"] = kept["stim_base"]
    return kept.drop(columns=["stim_base"], errors="ignore")


def _filter_gr_outlier_stims(
    df: pd.DataFrame,
    *,
    width: int,
    height: int,
    max_raw_err_px: float = 380.0,
    min_stims: int = 5,
) -> pd.DataFrame:
    """Drop stims where GR cloud prediction is still far from target (model still updating).

    GazeRecorder's cloud model should be accurate post-ShowCalibration. Large raw
    errors indicate the model hadn't converged yet during that stim window.
    """
    if "stim_key" not in df.columns or "target_x" not in df.columns:
        return df
    per_stim = (
        df.groupby("stim_key")
        .agg(
            x_norm_med=("x_norm", "median"),
            y_norm_med=("y_norm", "median"),
            target_x=("target_x", "first"),
            target_y=("target_y", "first"),
            n=("x_norm", "count"),
        )
        .reset_index()
    )
    per_stim["raw_err_px"] = np.sqrt(
        (per_stim["x_norm_med"] * width - per_stim["target_x"]) ** 2
        + (per_stim["y_norm_med"] * height - per_stim["target_y"]) ** 2
    )
    bad_mask = per_stim["raw_err_px"] > max_raw_err_px
    bad_keys = set(per_stim.loc[bad_mask, "stim_key"].tolist())
    if not bad_keys:
        return df
    good_count = int(bad_mask.shape[0]) - int(bad_mask.sum())
    # If keeping only good stims leaves too few, drop only the worst offenders.
    if good_count < min_stims:
        n_to_drop = max(0, len(per_stim) - min_stims)
        if n_to_drop == 0:
            return df
        worst = per_stim.nlargest(n_to_drop, "raw_err_px")["stim_key"]
        bad_keys = set(worst.tolist())
    result = df[~df["stim_key"].isin(bad_keys)].copy()
    if len(result) < 3:
        return df
    return result


def _aggregate_calibration_points(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    width: int,
    height: int,
) -> pd.DataFrame:
    """Collapse all samples per calibration point to a single row using robust median after outlier trimming."""
    rows: list[dict[str, float | int | str]] = []
    required = feature_cols + ["target_x", "target_y"]
    for stim_key, grp in df.groupby("stim_key", sort=True):
        clean = grp.dropna(subset=required).copy()
        if clean.empty:
            continue
        keep_mask = _robust_feature_inliers(clean[feature_cols])
        kept = clean.loc[keep_mask].copy()
        if kept.empty:
            kept = clean.copy()
        dispersion_px = _point_dispersion_px(kept, width=width, height=height)
        row: dict[str, float | int | str] = {
            "stim_key": str(stim_key),
            "target_x": float(kept["target_x"].median()),
            "target_y": float(kept["target_y"].median()),
            "source_rows": int(len(clean)),
            "kept_rows": int(len(kept)),
            "trimmed_rows": int(len(clean) - len(kept)),
            "dispersion_px": float(dispersion_px),
        }
        for col in feature_cols:
            row[col] = float(kept[col].median())
        rows.append(row)
    return pd.DataFrame(rows)


def _robust_feature_inliers(df: pd.DataFrame) -> np.ndarray:
    """Return boolean mask keeping the 65–85% of rows closest to the feature-space median (discards gaze outliers)."""
    values = df.to_numpy(dtype=float)
    n_rows = len(values)
    if n_rows <= 6:
        return np.ones(n_rows, dtype=bool)

    med = np.nanmedian(values, axis=0)
    dist = np.sqrt(np.sum(np.square(values - med), axis=1))
    keep_ratio = 0.65 if n_rows >= 20 else (0.75 if n_rows >= 10 else 0.85)
    keep_n = max(4, int(np.ceil(n_rows * keep_ratio)))
    ranked = np.argsort(dist, kind="stable")
    keep = np.zeros(n_rows, dtype=bool)
    keep[ranked[:keep_n]] = True
    return keep


def _point_dispersion_px(df: pd.DataFrame, *, width: int, height: int) -> float:
    """RMS distance of each sample from the median gaze position for one calibration point, in pixels."""
    coords = df[["x_norm", "y_norm"]].dropna()
    if len(coords) < 2:
        return 0.0
    x_px = coords["x_norm"].to_numpy(dtype=float) * width
    y_px = coords["y_norm"].to_numpy(dtype=float) * height
    cx = float(np.median(x_px))
    cy = float(np.median(y_px))
    return float(np.sqrt(np.mean((x_px - cx) ** 2 + (y_px - cy) ** 2)))


def _score_validation_tasks(
    df: pd.DataFrame | None,
    *,
    model: CalibModel,
    feature_cols: list[str],
) -> dict[str, float | int] | None:
    """Apply model to task-phase samples and return transfer MAE/RMSE/bias. Returns None if no task data available."""
    if df is None or df.empty:
        return None
    if "task_name" not in df.columns:
        return None
    if "target_x_px" not in df.columns or "target_y_px" not in df.columns:
        return None

    work = df.copy()
    if "validity" in work.columns:
        work = work[work["validity"] == 0]
    work = work.dropna(subset=["task_name", "target_x_px", "target_y_px"])
    if work.empty:
        return None
    work["task_name"] = work["task_name"].astype(str)
    work = work[work["task_name"].isin(TASK_REGISTRY.keys())]
    if work.empty:
        return None
    work = work.dropna(subset=feature_cols)
    if work.empty:
        return None

    pred_x, pred_y = _predict_pixels(work, model=model, feature_cols=feature_cols)
    err_x = pred_x - work["target_x_px"].to_numpy(dtype=float)
    err_y = pred_y - work["target_y_px"].to_numpy(dtype=float)
    err = np.sqrt(err_x**2 + err_y**2)
    return {
        "validation_task_samples": int(len(work)),
        "validation_task_mae_px": float(np.mean(err)),
        "validation_task_rmse_px": float(np.sqrt(np.mean(err**2))),
        "validation_task_bias_x_px": float(np.mean(err_x)),
        "validation_task_bias_y_px": float(np.mean(err_y)),
    }


def _predict_pixels(
    df: pd.DataFrame,
    *,
    model: CalibModel,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a fitted CalibModel to a DataFrame and return (pred_x_px, pred_y_px) arrays."""
    params = model.params
    powers = np.asarray(params["powers"], dtype=int)
    X = df[feature_cols].to_numpy(dtype=float)
    feats = np.ones((len(X), len(powers)), dtype=float)
    for term_idx, pow_vec in enumerate(powers):
        term = np.ones(len(X), dtype=float)
        for col_idx, pwr in enumerate(pow_vec):
            if pwr == 0:
                continue
            term *= np.power(X[:, col_idx], pwr)
        feats[:, term_idx] = term
    coef_x = np.asarray(params["coef_x"], dtype=float)
    coef_y = np.asarray(params["coef_y"], dtype=float)
    ix = float(params.get("intercept_x", 0.0))
    iy = float(params.get("intercept_y", 0.0))
    pred_x = feats @ coef_x + ix
    pred_y = feats @ coef_y + iy
    return pred_x, pred_y
