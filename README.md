# python-eyetracker

Unified calibration + benchmarking pipeline for WebGazer, GazeRecorder, Optimeyes, and Mpiris (webcam + MediaPipe iris).

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.11** | `mediapipe 0.10.x` does not support 3.12+. Use exactly 3.11. |
| **Windows 10/11** | Camera backend uses `cv2.CAP_DSHOW` (Windows-only). |
| **Webcam** | Required for `mpiris` and `optimeyes` trackers. |
| **Chrome or Edge** | Required for `webgazer` and `gazerecorder` trackers. |

---

## 1. Setup on a new machine

```powershell
# 1. Clone
git clone https://github.com/codemdvd/python-eyetracker.git
cd python-eyetracker

# 2. Create virtualenv with Python 3.11 specifically
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1     # PowerShell
# or: .venv\Scripts\activate     # CMD

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the package itself (editable)
pip install -e .
```

Verify:
```powershell
eyetrack --help
```

## 2. Optional web bridge
Required only for browser trackers (WebGazer / GazeRecorder).
```bash
uvicorn eyetrk.web_bridge.server:app --reload
```
If you use multiple web trackers at once, the CLI prints URLs for each tracker page; open each in a browser tab.

## 2b. GUI
Desktop control panel with tracker/task selection, parameter fields, run buttons, standalone bridge controls, and a report table.
```bash
eyetrack-gui
```
The GUI wraps the same scenarios as the CLI: `calibrate`, `quickstart`, `run-tasks`, plus report viewing. For `run-tasks` with browser trackers, set a fixed `session_id`, start the bridge in the GUI, open the tracker page, and then launch the task run.
The `Models` tab shows current calibration and transfer models, their status, key metrics, and source CSVs. It also lets you build transfer models from a calibration session plus a task session without using the terminal.

## 3. Calibration
Runs the on-screen 9-point protocol, fits per-tracker models, and stores raw samples under `runs/<session_id>/`.
```bash
eyetrack calibrate -t mpiris                               # single tracker
eyetrack calibrate -t mpiris --fullscreen                  # same, but stimulus window covers whole display
eyetrack calibrate --trackers mpiris optimeyes
eyetrack calibrate -t mpiris --dwell-ms 1500 --preview     # longer dwell, webcam preview
eyetrack calibrate -t webgazer --start-bridge --bridge-open-browser --fullscreen
eyetrack calibrate -t gazerecorder --start-bridge --bridge-open-browser --fullscreen

```
If `models/<tracker>.json` (or legacy `models/calib_model.json` for Mpiris) exists it is injected automatically. After each calibration run the freshly fitted model plus a `.report.txt` with diagnostics are copied to `models/<tracker>.json` / `models/<tracker>.report.txt` (and `models/calib_model.*` for compatibility).

Calibration commands use `models/<tracker>.json`. Task and replay commands prefer `models/<tracker>.transfer.json` when available. Benchmark tasks additionally apply online benchmark-time recalibration for the controlled task protocol.

## 3b. Quickstart (calibration + tasks in one go)
Run a full pass (calibration first, then tasks with the fitted models) with a single command:
```bash
eyetrack quickstart -t mpiris optimeyes --tasks fixation-grid step-saccades smooth-pursuit
eyetrack quickstart -t gazerecorder --start-bridge --bridge-open-browser
```

## 4. Benchmark task suite
Shows three canonical tasks (fixation grid, step saccades, smooth pursuit) and logs gaze responses aligned with target coordinates.
```bash
eyetrack run-tasks -t mpiris
eyetrack run-tasks --trackers mpiris optimeyes --tasks fixation-grid smooth-pursuit
```

## 5. Metrics / reporting
Reads the recorded sessions and prints MAE, precision RMS, and drop-rate per task / tracker.
```bash
eyetrack report --runs runs                  # latest sessions
eyetrack report --session <uuid> --tracker mpiris
```

## 5b. Build transfer models from paired sessions
Use one calibration session plus one task session to build the task-oriented transfer model:
```bash
eyetrack optimize-session-pair --calib-session <calib_session> --task-session <task_session> -t mpiris
eyetrack optimize-session-pair --calib-session <calib_session> --task-session <task_session> -t optimeyes
```
This writes `models/<tracker>.transfer.json` and `models/<tracker>.transfer.report.txt`.

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
