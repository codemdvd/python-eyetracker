#!/usr/bin/env python3
"""
Offline fitter for poly2 calibration models based on logged CSVs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from eyetrk.calib.fitting import fit_dataframe


def _find_latest_csv(in_path: Path) -> Path | None:
    csvs = sorted(in_path.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None


def main():
    ap = argparse.ArgumentParser(description="Fit poly2 calibration model from logs")
    ap.add_argument("--csv", type=Path, default=None, help="Path to a specific CSV log")
    ap.add_argument("--in_path", type=Path, default=Path("runs"), help="Directory with CSV logs")
    ap.add_argument("--out", type=Path, default=Path("models/calib_model.json"), help="Output path for model JSON")
    ap.add_argument("--width", type=int, default=1280, help="Screen width in pixels")
    ap.add_argument("--height", type=int, default=720, help="Screen height in pixels")
    ap.add_argument(
        "--per_stim_median",
        dest="per_stim_median",
        action="store_true",
        default=False,
        help="Aggregate one median sample per stim (default: off)",
    )
    ap.add_argument(
        "--no-per_stim_median",
        dest="per_stim_median",
        action="store_false",
        help="Disable per-stim median aggregation",
    )
    args = ap.parse_args()

    csv_path = args.csv or _find_latest_csv(args.in_path)
    if csv_path is None:
        raise SystemExit(f"No CSV found. Provide --csv or put logs into {args.in_path}")

    df = pd.read_csv(csv_path)
    fit_out = fit_dataframe(
        df,
        width=args.width,
        height=args.height,
        per_stim_median=args.per_stim_median,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(fit_out.model.model_dump(), f, indent=2)

    report_path = args.out.with_suffix(".report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"CSV: {csv_path}\n")
        f.write(json.dumps(fit_out.diag, indent=2))
        f.write("\n")
        if fit_out.per_stim is not None:
            f.write("\nPer-stim error (px):\n")
            f.write(fit_out.per_stim.to_string(index=False))
            f.write("\n")

    print(f"\n[fit_calib_model] Model saved: {args.out}")
    print(f"[fit_calib_model] Report: {report_path}")
    print("[fit_calib_model] Metrics:")
    for k, v in fit_out.diag.items():
        print(f"  {k}: {v}")
    if fit_out.per_stim is not None:
        print("\nPer-stim sample counts and errors (px):")
        print(fit_out.per_stim.to_string(index=False))


if __name__ == "__main__":
    main()
