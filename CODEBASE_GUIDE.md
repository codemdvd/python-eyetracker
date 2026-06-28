# Architecture Overview

This document explains the structure of the codebase for someone reading it for the first time.

---

## What This System Does

A **unified calibration and benchmarking framework** for four eye trackers running simultaneously:

| Tracker | Technology | Where it runs |
|---|---|---|
| **mpiris** | MediaPipe FaceMesh iris landmarks | Python thread, webcam |
| **optimeyes** | MediaPipe + head-pose compensation | Python thread, webcam |
| **webgazer** | WebGazer.js | Browser tab, sends data over WebSocket |
| **gazerecorder** | GazeCloudAPI.js (cloud-calibrated) | Browser tab, sends data over WebSocket |

**Typical session flow:**
1. Show a 9-point calibration grid on screen → collect raw gaze samples per tracker
2. Fit a polynomial regression model per tracker (maps raw gaze → screen pixels)
3. Show benchmark tasks (fixation grid, saccades, smooth pursuit) → collect gaze vs. target ground truth
4. Compute accuracy metrics (MAE, precision, drop rate) and print a report

---

## Directory Structure

```
python-eyetracker/
│
├── src/eyetrk/              ← main Python package (installed as "eyetrk")
│   ├── cli.py               ← entry point: all terminal commands live here
│   ├── gui.py               ← optional Tkinter desktop GUI (wraps cli.py calls)
│   ├── orchestrator.py      ← central coordinator: routes samples, manages timing
│   ├── run_realtime.py      ← minimal standalone script (mpiris only, no CLI)
│   │
│   ├── adapters/            ← one file per tracker; all implement the same interface
│   ├── calib/               ← calibration math: grid sequences, model fitting, quality checks
│   ├── bench/               ← benchmark tasks, metric calculations, session reports
│   ├── core/                ← shared data types and the Tracker abstract base class
│   ├── io/                  ← disk I/O: writing CSV samples, JSONL timeline, session metadata
│   ├── stim/                ← pygame stimulus display (the red dot on screen)
│   └── web_bridge/          ← FastAPI WebSocket server that browser tabs connect to
│
├── tools/                   ← browser-side files and offline utilities
│   ├── webgazer_bridge.html      ← HTML page opened in browser for WebGazer
│   ├── gazerecorder_bridge.html  ← HTML page opened in browser for GazeRecorder
│   ├── webgazer.js               ← WebGazer library (bundled, not fetched from CDN)
│   ├── GazeCloudAPI.js           ← GazeRecorder SDK (bundled)
│   └── fit_calib_model.py        ← offline script: refit a model from an existing CSV
│
├── models/                  ← saved calibration models (gitignored except committed examples)
│   ├── mpiris.json               ← active calibration model for mpiris
│   ├── mpiris.report.txt         ← diagnostics: R², MAE, feature set, etc.
│   ├── mpiris.transfer.json      ← transfer model (fitted on task data, used at benchmark time)
│   └── <tracker>.{json,report.txt,transfer.json}  ← same pattern for all 4 trackers
│
├── runs/                    ← session recordings (gitignored; created at runtime)
│   └── <uuid>_calib/        ← one directory per session (calibration or task)
│       ├── session.json          ← metadata (screen size, tracker version, etc.)
│       ├── samples_mpiris.csv    ← raw gaze samples for this tracker
│       ├── samples_webgazer.csv
│       └── timeline.jsonl        ← timestamped event log (stim_on, stim_off, tasks_start, …)
│
├── gxipy/                   ← Daheng industrial camera vendor SDK (not on PyPI)
├── requirements.txt         ← pinned runtime dependencies
├── requirements-dev.txt     ← adds pytest
└── pyproject.toml           ← package metadata; installs `eyetrack` and `eyetrack-gui` commands
```

---

## Entry Points

### Terminal
```
eyetrack calibrate   → run 9-point calibration
eyetrack quickstart  → calibration + benchmark tasks in one command
eyetrack run-tasks   → benchmark tasks only (uses existing model)
eyetrack report      → print accuracy metrics from runs/
eyetrack optimize-session-pair → build a transfer model from a calib+task session pair
```
All commands are defined in **`src/eyetrk/cli.py`** as Typer functions.

### Desktop GUI
```
eyetrack-gui
```
Defined in **`src/eyetrk/gui.py`**. Wraps the same `cli.py` functions — calls them directly as Python functions in a background thread and redirects stdout to a log widget.

---

## Component Descriptions

### `cli.py` — Command-Line Interface
The top-level coordinator for a user-facing session. Each command:
1. Parses arguments and builds per-tracker config dicts (`MPIRIS_DEFAULT_CFG`, `OPTIMEYES_DEFAULT_CFG`)
2. Creates adapter instances and a `RunLogger`
3. Instantiates `Orchestrator` and calls `calibrate_all()` / `run_tasks()`
4. After calibration: calls `fit_dataframe()` to build a model, saves it to `models/`

Key constants at the top of the file:
- `MPIRIS_DEFAULT_CFG` / `OPTIMEYES_DEFAULT_CFG` — default camera settings (fps, resolution, camera index, backend, etc.)
- `MODEL_QUALITY_GATES` — thresholds that decide whether a freshly fitted model is good enough to be promoted to `models/<tracker>.json`

---

### `orchestrator.py` — Central Coordinator
Sits between adapters and the stimulus engine. Responsibilities:
- **Starts and stops all adapter threads** (`start_streams()` / `stop_streams()`)
- **Runs the calibration sequence** (`calibrate_all()`): shows 9-point grid via `StimEngine`, collects samples from all adapters simultaneously, handles retries when a point is rejected by `acceptance()`
- **Runs benchmark tasks** (`run_tasks()`): shows task stimuli, routes samples to the logger with the correct `stim_id` and `target_x_px/y_px` labels
- **Retroactive labeling**: browser trackers have ~100ms WebSocket lag, so samples that arrive slightly late are labeled with the stim that was active at their `timestamp_ms`, not at the moment they arrive
- **Broadcasts events to browser trackers**: sends `stim_on` / `stim_off` / `tasks_start` messages through the web bridge so the browser page knows what's on screen

---

### `adapters/` — Tracker Implementations

All adapters implement the `Tracker` interface (`core/tracker.py`):

| Method | Purpose |
|---|---|
| `initialize(cfg)` | Apply config dict, return `TrackerInfo` |
| `start_stream(callback, session_id)` | Start background thread; call `callback(sample)` for every frame |
| `stop()` | Signal thread to exit, join |
| `on_event(event, payload)` | React to calibration lifecycle events (freeze gain, etc.) |
| `set_external_model(model)` | Accept a `CalibModel` fitted offline and use it for coordinate prediction |

#### `adapters/mpiris.py` — MpirisAdapter
Opens webcam with OpenCV, runs MediaPipe FaceMesh on every frame, extracts iris landmark centroids, applies the calibration polynomial to produce `x_px, y_px`. Supports DSHOW and MSMF camera backends.

#### `adapters/optimeyes.py` — OptimeyesAdapter
Same pipeline as mpiris but adds head-pose compensation: subtracts estimated head movement from the raw iris position before applying the polynomial.

#### `adapters/webgazer.py` — WebGazerAdapter
Does not open a camera. Listens for samples arriving from the browser via the WebSocket bridge. The browser tab runs WebGazer.js which does face tracking and gaze estimation internally.

#### `adapters/gazerecorder.py` — GazerecorderAdapter
Same pattern as WebGazer but wraps GazeCloudAPI.js (cloud-calibrated). The adapter handles the difference that GazeRecorder performs its own calibration wizard inside the browser.

#### `adapters/iris_common.py`
Shared math used by both mpiris and optimeyes: landmark index extraction, eye-center aggregation, head-pose center estimation.

#### `adapters/daheng_capture.py`
`cv2.VideoCapture`-compatible wrapper for Daheng industrial cameras via the `gxipy` vendor SDK. Drops in wherever `cv2.VideoCapture` is used.

---

### `calib/` — Calibration Math

#### `calib/protocols.py`
Defines calibration grid layouts. `generate_9pt_grid()` returns a `Sequence` of 9 `Point` objects (one center + 8 at corners/edges). Each `Point` has normalized screen coordinates `(x_norm, y_norm)` and timing parameters.

#### `calib/fitting.py` — `fit_dataframe()`
The core model-fitting function. Given a DataFrame of calibration samples:
1. Filters to valid samples (`validity == 0`)
2. Groups samples by calibration point (`stim_id`)
3. Tries multiple input feature sets (e.g., `[x_norm, y_norm]`, `[raw_x_norm, head_x, head_y]`)
4. For each feature set: fits polynomial regression (degree 1 or 2) with Ridge regularization, selects degree via leave-one-out cross-validation
5. Picks the best feature set (lowest CV MAE, or lowest transfer error if a task validation set is provided)
6. Returns a `FitOutput` containing the `CalibModel`, diagnostic stats, and per-point error breakdown

The fitted model maps `(feature_values) → (x_px, y_px)` using two independent Ridge regressors (one per axis).

#### `calib/acceptance.py` — `acceptance()`
After collecting samples for one calibration point, decides whether to accept or request a retry. Checks:
- Minimum number of valid samples
- Maximum spatial dispersion (samples too spread out → unstable fixation)
- Maximum offset from the target position (gaze too far from the dot)

---

### `bench/` — Benchmarking

#### `bench/tasks.py`
Defines the three standard benchmark tasks. Each task is a `Task` object with a `timeline` list of `Event` objects:
- **`fixation-grid`**: 5×5 grid of 25 fixation points, each shown for 800ms
- **`step-saccades`**: alternating left (x=0.2) / right (x=0.8) target at y=0.5, 12 cycles
- **`smooth-pursuit`**: target moves in a circle (radius=0.25, 4 revolutions at 4s/rev)

#### `bench/metrics.py`
Pure functions for accuracy metrics: `mean_absolute_error()`, `root_mean_squared_error()`, `precision_rms_sd()` (spatial consistency), `drop_rate()` (fraction of frames with no valid gaze).

#### `bench/report.py` — `compute_session_metrics()`
Reads a session directory, matches gaze samples to task targets by timestamp, and returns a list of `TaskMetrics` (one per tracker per task).

#### `bench/postprocess.py`
Online correction applied during task runs:
- `BenchmarkBiasCorrector`: estimates and removes slow drift (head movement, tracker drift) using exponential moving average of the error signal
- `BenchmarkAnchorRecalibrator`: uses the first few fixation-grid points as anchors to fit a task-time correction model on-the-fly

---

### `io/` — Data I/O

#### `io/logger.py` — `RunLogger`
Creates `runs/<session_id>/` and writes:
- `session.json` — `SessionMeta` (screen dimensions, tracker version, lighting condition, etc.)
- `samples_<tracker>.csv` — one row per gaze sample; ~50 columns (head pose, iris positions, calibrated output, stim context)
- `timeline.jsonl` — one JSON object per line, one per event (`stim_on`, `stim_off`, `calibration_start`, `task_start`, etc.)

#### `io/schema.py`
`SessionMeta` dataclass: the metadata stored in `session.json`. Includes screen size, PPI, viewing distance, protocol, dwell time, refresh rate, lighting condition, webcam FPS.

#### `io/replay.py`
Used when a second tracker is run offline against a recorded video. `SessionTimelineLabeler` replays the `timeline.jsonl` events and, for a given frame timestamp, returns what stimulus was on screen at that moment. Used by `FrameLabeler` to inject `stim_id` and `target_x_px/y_px` into replayed samples.

---

### `stim/engine.py` — Stimulus Display

`StimEngine` wraps a pygame window. Key behaviors:
- `wait_for_ready(msg, cam_index, ...)` — shows a camera preview + alignment overlay. The user fits their face in the frame and presses Space to proceed.
- Draws a red filled circle at the current target position (`(x_norm, y_norm)` → pixels)
- Handles dwell timing: holds the dot for `dwell_ms`, then fires `stim_off`
- For smooth pursuit: receives per-frame `stim_move` events and redraws every 16ms

Camera is opened here **only** for the preview screen. It is released before the adapter threads start so there is no camera contention.

---

### `web_bridge/server.py` — WebSocket Bridge

A FastAPI app with a single WebSocket endpoint `/ws/{tracker_id}`. Runs in a background thread (uvicorn).

**Browser → Python**: JSON objects with gaze sample fields (`x_norm`, `y_norm`, `timestamp_ms`, etc.). Sanitized and forwarded to the registered adapter callback.

**Python → Browser**: JSON events pushed via `push_event(tracker_id, payload)`. Event types:
- `"start"` — begin a new session (sends `session_id`, screen dimensions)
- `"stim_on"` / `"stim_off"` — tell the browser what's on screen (calibration or task point)
- `"bridge_start"` — switch phase (calibration → tasks)

The bridge also serves the HTML pages (`tools/webgazer_bridge.html`, `tools/gazerecorder_bridge.html`) over HTTP so the user just opens `http://localhost:8001` in a browser.

---

### `tools/` — Browser Pages and Offline Utilities

#### `tools/webgazer_bridge.html` + `webgazer.js`
A self-contained web page that loads WebGazer.js, runs face tracking in the browser, shows calibration dots received from Python over WebSocket, and streams gaze samples back.

#### `tools/gazerecorder_bridge.html` + `GazeCloudAPI.js`
Same idea for GazeRecorder. More complex because GazeRecorder performs its own server-side calibration wizard (`ShowCalibration()`) and caches models by hardware fingerprint. The bridge detects when the cloud returns a cached (potentially stale) model and forces a fresh calibration.

#### `tools/fit_calib_model.py`
Standalone CLI script:
```
python tools/fit_calib_model.py --csv runs/<session>/samples_mpiris.csv \
    --out models/calib_model.json --per_stim_median --width 1920 --height 1080
```
Useful for reprocessing old sessions or tuning model parameters without re-running calibration.

---

## Key Data Types

### `Sample` (`core/types.py`)
The universal data container — every gaze frame from every tracker is a `Sample`. Relevant fields:

| Field group | Fields | Meaning |
|---|---|---|
| Identity | `session_id`, `tracker_id`, `timestamp_ms`, `frame_id` | When and where this sample came from |
| Head pose | `head_x/y/z`, `yaw`, `pitch`, `roll` | 3D head position and orientation (normalized) |
| Raw gaze | `x_norm`, `y_norm` | Raw tracker output before calibration polynomial (0–1 screen coords) |
| Calibrated | `x_px`, `y_px` | Final gaze position in screen pixels |
| Eye geometry | `left_eye_w/h_norm`, `right_eye_w/h_norm`, `left/right_iris_abs_x/y_norm` | Eye openness and iris positions |
| Task context | `task_name`, `stim_id`, `target_x/y_px` | What was on screen when this sample was captured |
| Quality | `validity` (0=valid, 1=invalid), `confidence` | Sample quality flags |
| GazeRecorder | `gr_gaze_x/y`, `gr_doc_x/y` | Raw GazeRecorder output (before our polynomial) |

### `CalibModel` (`core/types.py`)
Serializable representation of a fitted calibration model. Stored as JSON in `models/`.
- `model_type`: `"poly2"` (polynomial regression) or `"native"` (tracker's own model)
- `params`: dict containing `powers`, `coef_x`, `coef_y`, `intercept_x/y`, `ridge_alpha_x/y`, `input_features`, `degree`
- `fit_error_px`: training MAE in pixels

### `SessionMeta` (`io/schema.py`)
Metadata written to `session.json`. Contains `session_id`, screen `width_px` / `height_px`, `ppi`, `distance_cm`, calibration `protocol` (e.g. `"9pt"`), `dwell_ms`, `refresh_hz`, `lighting`.

---

## Data Flow Diagram

```
User runs:  eyetrack calibrate -t mpiris -t webgazer --start-bridge --bridge-open-browser
                │
                ▼
          cli.py: calibrate()
                │
                ├─ starts web bridge (uvicorn thread)  ──► serves tools/webgazer_bridge.html
                │                                              │
                ├─ opens browser automatically                 │ (WebSocket ws://localhost:8000)
                │                                              │
                ├─ creates MpirisAdapter + WebGazerAdapter     │
                │                                              │
                └─ Orchestrator.calibrate_all()                │
                        │                                      │
                        ├─ StimEngine.wait_for_ready()         │
                        │   (camera preview, Space to start)   │
                        │                                      │
                        ├─ start_streams()                     │
                        │   ├─ MpirisAdapter thread: webcam → MediaPipe → callback(sample)
                        │   └─ WebGazerAdapter: registers callback, waits for WS samples ◄─┘
                        │
                        ├─ for each of 9 calibration points:
                        │   ├─ StimEngine draws dot at (x_norm, y_norm)
                        │   ├─ broadcast stim_on → WebSocket → browser shows dot
                        │   ├─ wait dwell_ms, collect samples from all trackers
                        │   ├─ acceptance() → accept or retry (up to 3×)
                        │   └─ broadcast stim_off → browser hides dot
                        │
                        └─ stop_streams()

          cli.py: _offline_refit_session()
                │
                ├─ reads runs/<session>/samples_mpiris.csv
                ├─ fit_dataframe() → CalibModel (poly2, auto degree/features)
                ├─ quality gate check → promote to models/mpiris.json if passes
                └─ write models/mpiris.report.txt (R², MAE, CV MAE, feature set)
```

---

## How Models Are Stored and Loaded

```
models/
  mpiris.json            ← calibration model (fitted on calib session)
  mpiris.report.txt      ← diagnostics text
  mpiris.transfer.json   ← transfer model (fitted on calib+task pair, preferred at task time)
  mpiris.transfer.report.txt
```

At task time (`run-tasks`), each adapter calls `set_external_model()`:
- If `models/<tracker>.transfer.json` exists → use it (better cross-session accuracy)
- Otherwise fall back to `models/<tracker>.json` (calibration model)
- If neither exists → tracker runs with its native model (browser trackers) or uncalibrated (webcam trackers)

Models are deliberately excluded from git (`models/*.json` in `.gitignore`). They must be generated fresh on each machine by running calibration.
