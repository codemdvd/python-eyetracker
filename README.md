# python-eyetracker

Unified calibration + benchmarking pipeline for WebGazer, TurkerGaze, Pupil Labs Core, Optimeyes, and Mpiris (webcam + MediaPipe iris).


## 1. Installation
```bash
python -m venv .venv
.venv\Scripts\activate           # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e .
```

## 2. Optional web bridge
Required only for browser trackers (WebGazer / TurkerGaze).
```bash
uvicorn eyetrk.web_bridge.server:app --reload
```

## 3. Calibration
Runs the on-screen 9-point protocol, fits per-tracker models, and stores raw samples under `runs/<session_id>/`.
```bash
eyetrack calibrate -t mpiris                               # single tracker
eyetrack calibrate -t mpiris --fullscreen                  # same, but stimulus window covers whole display
eyetrack calibrate --trackers mpiris pupilcore optimeyes
eyetrack calibrate -t mpiris --dwell-ms 1500 --preview     # longer dwell, webcam preview
```
If `models/<tracker>.json` (or legacy `models/calib_model.json` for Mpiris) exists it is injected automatically. After each calibration run the freshly fitted model plus a `.report.txt` with diagnostics are copied to `models/<tracker>.json` / `models/<tracker>.report.txt` (and `models/calib_model.*` for compatibility).

## 3b. Quickstart (calibration + tasks in one go)
Run a full pass (calibration first, then tasks with the fitted models) with a single command:
```bash
eyetrack quickstart -t mpiris pupilcore --tasks fixation-grid step-saccades smooth-pursuit
```

## 4. Benchmark task suite
Shows three canonical tasks (fixation grid, step saccades, smooth pursuit) and logs gaze responses aligned with target coordinates.
```bash
eyetrack run-tasks -t mpiris
eyetrack run-tasks --trackers mpiris pupilcore --tasks fixation-grid smooth-pursuit
```

## 5. Metrics / reporting
Reads the recorded sessions and prints MAE, precision RMS, and drop-rate per task / tracker.
```bash
eyetrack report --runs runs                  # latest sessions
eyetrack report --session <uuid> --tracker mpiris
```

## 6. Re-fitting a calibration model offline
Use any `runs/<session>/samples_<tracker>.csv` (containing stim IDs or task targets) to generate a new poly2 model.
```bash
python tools/fit_calib_model.py --csv runs/<session>/samples_mpiris.csv \
    --out models/calib_model.json --per_stim_median --width 1280 --height 720
```

## 7. Standalone real-time logging
Quick script for Mpiris without the full CLI orchestration (logs to `runs/<uuid>/samples_mpiris.csv`).
```bash
python src/eyetrk/run_realtime.py
```

---
- Runs are written to `runs/<session_id>/`.
- Tracker adapters live in `src/eyetrk/adapters/`.
- Stimulus engine is backed by pygame (`src/eyetrk/stim/engine.py`).
