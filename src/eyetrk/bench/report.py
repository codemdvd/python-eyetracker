from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .metrics import drop_rate, mean_absolute_error, precision_rms_sd, root_mean_squared_error
from .tasks import TASK_REGISTRY


@dataclass
class TaskMetrics:
    """Computed accuracy and reliability metrics for one tracker on one benchmark task in one session."""
    session_id: str
    tracker_id: str
    task_name: str
    samples: int
    valid_samples: int
    mae_px: float | None
    rmse_px: float | None
    precision_px: float | None
    drop_rate: float
    ppi: float = field(default=96.0)
    distance_cm: float = field(default=60.0)


def compute_session_metrics(session_dir: Path) -> List[TaskMetrics]:
    """Load session.json and all samples_*.csv files from a session directory and return TaskMetrics for every tracker × task combination."""
    session_dir = Path(session_dir)
    session_id = session_dir.name
    meta_path = session_dir / "session.json"
    if not meta_path.exists():
        return []

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return []

    screen_w = int(meta.get("width_px", 1280))
    screen_h = int(meta.get("height_px", 720))
    ppi = float(meta.get("ppi", 96.0))
    distance_cm = float(meta.get("distance_cm", 60.0))

    results: List[TaskMetrics] = []
    for csv_path in sorted(session_dir.glob("samples_*.csv")):
        tracker_id = csv_path.stem.replace("samples_", "")
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        results.extend(
            _metrics_for_tracker(
                df=df,
                session_id=session_id,
                tracker_id=tracker_id,
                screen_w=screen_w,
                screen_h=screen_h,
                ppi=ppi,
                distance_cm=distance_cm,
            )
        )

    return results


def _metrics_for_tracker(
    df: pd.DataFrame,
    *,
    session_id: str,
    tracker_id: str,
    screen_w: int,
    screen_h: int,
    ppi: float = 96.0,
    distance_cm: float = 60.0,
) -> List[TaskMetrics]:
    """Group a tracker's sample DataFrame by task_name and compute MAE, RMSE, precision, and drop-rate for each group."""
    if df.empty or "task_name" not in df.columns:
        return []

    if "target_x_px" not in df.columns or "target_y_px" not in df.columns:
        return []

    subset = df.dropna(subset=["task_name", "target_x_px", "target_y_px"]).copy()
    if subset.empty:
        return []
    subset["task_name"] = subset["task_name"].astype(str)
    subset = subset[subset["task_name"].isin(TASK_REGISTRY.keys())]
    if subset.empty:
        return []

    preds = _predict_pixels(subset, screen_w, screen_h)
    if preds is None:
        return []
    subset["pred_x_px"], subset["pred_y_px"] = preds
    subset = subset.dropna(subset=["pred_x_px", "pred_y_px"])
    if subset.empty:
        return []

    if "validity" in subset.columns:
        subset["valid_mask"] = subset["validity"] == 0
    else:
        subset["valid_mask"] = True

    out: List[TaskMetrics] = []
    for task_name, grp in subset.groupby("task_name"):
        mask = grp["valid_mask"].to_numpy(dtype=bool)
        preds_arr = grp[["pred_x_px", "pred_y_px"]].to_numpy(dtype=float)
        targets_arr = grp[["target_x_px", "target_y_px"]].to_numpy(dtype=float)
        if preds_arr.size == 0:
            continue

        if mask.any():
            mae_val = mean_absolute_error(preds_arr[mask], targets_arr[mask])
            rmse_val = root_mean_squared_error(preds_arr[mask], targets_arr[mask])
            precision_val = precision_rms_sd(preds_arr[mask]) if mask.sum() >= 2 else None
        else:
            mae_val = None
            rmse_val = None
            precision_val = None

        drop = drop_rate(mask.astype(float))

        out.append(
            TaskMetrics(
                session_id=session_id,
                tracker_id=tracker_id,
                task_name=str(task_name),
                samples=int(len(grp)),
                valid_samples=int(mask.sum()),
                mae_px=mae_val,
                rmse_px=rmse_val,
                precision_px=precision_val,
                drop_rate=float(drop),
                ppi=ppi,
                distance_cm=distance_cm,
            )
        )

    return out


def _predict_pixels(df: pd.DataFrame, screen_w: int, screen_h: int):
    """Return (x_px, y_px) Series from the DataFrame, falling back from pixel to normalised columns and filling gaps."""
    x_series = df["x_px"].copy() if "x_px" in df.columns else None
    y_series = df["y_px"].copy() if "y_px" in df.columns else None

    if x_series is None:
        if "x_norm" not in df.columns:
            return None
        x_series = df["x_norm"] * screen_w
    else:
        if "x_norm" in df.columns:
            x_series = x_series.fillna(df["x_norm"] * screen_w)

    if y_series is None:
        if "y_norm" not in df.columns:
            return None
        y_series = df["y_norm"] * screen_h
    else:
        if "y_norm" in df.columns:
            y_series = y_series.fillna(df["y_norm"] * screen_h)

    return x_series, y_series
