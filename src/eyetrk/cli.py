# eyetrk/cli.py
import json
import shutil
import sys
import uuid
import http.server
import threading
import webbrowser
from pathlib import Path
from typing import List, Tuple, Protocol

import pandas as pd
import typer
import uvicorn

from eyetrk.core.types import CalibModel
from eyetrk.web_bridge import server as bridge_server

from .adapters.mpiris import MpirisAdapter
from .adapters.optimeyes import OptimeyesAdapter
from .adapters.openseeface import OpenSeeFaceAdapter
from .adapters.pupilcore import PupilCoreAdapter
from .adapters.turkergaze import TurkerGazeAdapter
from .adapters.webgazer import WebGazerAdapter
from .bench.report import TaskMetrics, compute_session_metrics
from .bench.tasks import resolve_tasks
from .calib.fitting import fit_dataframe
from .io.logger import RunLogger
from .io.schema import SessionMeta
from .orchestrator import Orchestrator
from typing import List, Tuple, Protocol, Any

app = typer.Typer(help="Unified eye tracking CLI")


class TrackerAdapter(Protocol):
    def initialize(self, config: dict) -> Any:
        ...

    def set_external_model(self, model: CalibModel) -> None:
        ...

    def start_stream(self, callback, session_id: str | None = None) -> None:
        ...

    def stop(self) -> None:
        ...

    def on_event(self, event: str, payload: dict | None = None) -> None:
        ...

DEFAULT_TRACKERS = ["webgazer", "turkergaze", "pupilcore", "optimeyes", "openseeface", "mpiris"]
ROOT = Path(__file__).resolve().parents[2]
OPENSEEFACE_DEFAULT_CFG = {
    # Запускаем просто facetracker.py, а cwd установим в адаптере
    "cmd": ["python", "facetracker.py", "--gaze-tracking", "1", "--silent", "1"],
    "flip_x": True,
    "flip_y": False,
    "gain_x": 2.0,
    "gain_y": 2.0,
    "out_width": 1280,
    "out_height": 720,
    "min_conf": 0.6,
}
MPIRIS_DEFAULT_CFG = {
    "fps": 60.0,
    "camera_index": 0,
    "width": 1280,
    "height": 720,
    "flip_x": False,
    "flip_y": False,
    "gain_x": 1.0,
    "gain_y": 1.0,
    "ema_alpha": 0.2,
    "head_alpha": 0.05,
    "head_comp": False,
    "head_gain_x": 0.9,
    "head_gain_y": 0.15,
    "min_conf": 0.6,
    "max_jump_norm": 0.12,
    "min_eye_w": 0.02,
    "min_eye_h": 0.008,
    "blink_ratio": 0.16,
    "auto_gain": True,
    "auto_gain_alpha": 0.02,
    "auto_gain_margin": 0.05,
    "preview": False,
    "out_width": 1280,
    "out_height": 720,
}
OPTIMEYES_DEFAULT_CFG = {
    "fps": 60.0,
    "camera_index": 0,
    "width": 1280,
    "height": 720,
    "flip_x": True,
    "flip_y": True,
    "gain_x": 1.0,
    "gain_y": 1.4,
    "ema_alpha": 0.2,
    "max_jump_norm": 0.12,
    "min_eye_w": 0.02,
    "min_eye_h": 0.01,
    "blink_ratio": 0.16,
    "min_conf": 0.6,
    "yaw_gain": 0.32,
    "pitch_gain": 0.06,
    "pose_alpha": 0.02,
    "auto_gain": False,
    "auto_gain_alpha": 0.07,
    "auto_gain_margin": 0.1,
    "preview": False,
    "out_width": 1280,
    "out_height": 720,
}


@app.command()
def calibrate(
    trackers: List[str] = typer.Option(
        None,
        "--trackers",
        "-t",
        help="Trackers to use (repeatable). E.g.: -t pupilcore -t webgazer",
    ),
    fullscreen: bool = typer.Option(False, help="Show stimuli fullscreen"),
    width_px: int = typer.Option(1280, help="Screen width in pixels for calibration stimuli"),
    height_px: int = typer.Option(720, help="Screen height in pixels for calibration stimuli"),
    dwell_ms: int = typer.Option(1500, help="Dwell time per calibration point (ms)"),
    preview: bool = typer.Option(False, help="Show webcam preview during calibration (mpiris/optimeyes)"),
    pupil_host: str = typer.Option("127.0.0.1", help="Pupil Remote host"),
    pupil_req_port: int = typer.Option(50020, help="Pupil Remote REQ port"),
    session_id: str | None = typer.Option(None, help="Optional fixed session id (for web trackers bridge)"),
    start_bridge: bool = typer.Option(False, help="Auto-start embedded web bridge for web trackers"),
    bridge_ws_port: int = typer.Option(8000, help="WS port for embedded web bridge"),
    bridge_http_port: int = typer.Option(8001, help="HTTP port for embedded web bridge"),
    bridge_open_browser: bool = typer.Option(False, help="Open helper page when starting embedded bridge"),
    wait_ready: bool = typer.Option(False, help="Pause before showing stimuli (set up web trackers first)"),
):
    tracker_names = _normalize_tracker_names(trackers)
    if not tracker_names:
        raise typer.BadParameter("No trackers selected.")

    session_id = session_id or str(uuid.uuid4())
    logger = RunLogger()
    meta = SessionMeta(
        session_id=session_id,
        os="unknown",
        width_px=width_px,
        height_px=height_px,
        ppi=96.0,
        distance_cm=60.0,
        protocol="9pt",
        dwell_ms=dwell_ms,
    )
    logger.start_session(meta)
    typer.echo(f"[eyetrack] Session id: {session_id}")

    # Auto-start bridge if web trackers present
    stop_bridge = None
    if start_bridge and any(t in tracker_names for t in ("webgazer", "turkergaze")):
        url, stopper = _start_embedded_web_bridge(
            "0.0.0.0",
            bridge_ws_port,
            bridge_http_port,
            open_browser=bridge_open_browser,
            session_id=session_id,
            tracker_id="webgazer",
        )
        stop_bridge = stopper
        typer.echo(f"[eyetrack] Embedded web bridge at {url}")
        if wait_ready:
            try:
                input("[eyetrack] Press Enter to start calibration once web tracker page is streaming...")
            except EOFError:
                pass

    mpiris_cfg = dict(MPIRIS_DEFAULT_CFG)
    mpiris_cfg["out_width"] = width_px
    mpiris_cfg["out_height"] = height_px
    mpiris_cfg["preview"] = preview
    opt_cfg = dict(OPTIMEYES_DEFAULT_CFG)
    opt_cfg["out_width"] = width_px
    opt_cfg["out_height"] = height_px
    opt_cfg["preview"] = preview

    adapters = _prepare_adapters(
        tracker_names,
        pupil_host=pupil_host,
        pupil_req_port=pupil_req_port,
        mpiris_cfg=mpiris_cfg,
        optimeyes_cfg=opt_cfg,
        load_models=False,
    )

    orch = Orchestrator(adapters, logger, session_id, session_meta=meta)
    orch.start_streams()
    try:
        orch.calibrate_all(dwell_ms=meta.dwell_ms, gap_ms=300, fullscreen=fullscreen)
    finally:
        orch.stop_streams()
        if stop_bridge:
            stop_bridge()

    _finalize_calibration_outputs(
        session_id,
        tracker_names,
        width_px=meta.width_px,
        height_px=meta.height_px,
    )
    typer.echo(f"Calibration session created: {session_id}")


@app.command("quickstart")
def quickstart(
    trackers: List[str] = typer.Option(
        None,
        "--trackers",
        "-t",
        help="Trackers to use (repeatable). E.g.: -t pupilcore -t webgazer",
    ),
    tasks: List[str] = typer.Option(
        ["fixation-grid", "step-saccades", "smooth-pursuit"],
        "--tasks",
        "-k",
        help="Benchmark tasks to run after calibration",
    ),
    fullscreen: bool = typer.Option(False, help="Show stimuli fullscreen"),
    width_px: int = typer.Option(1280, help="Screen width in pixels for calibration and tasks"),
    height_px: int = typer.Option(720, help="Screen height in pixels for calibration and tasks"),
    dwell_ms: int = typer.Option(1500, help="Dwell time per calibration point (ms)"),
    preview: bool = typer.Option(False, help="Show webcam preview during calibration (mpiris/optimeyes)"),
    pupil_host: str = typer.Option("127.0.0.1", help="Pupil Remote host"),
    pupil_req_port: int = typer.Option(50020, help="Pupil Remote REQ port"),
    start_bridge: bool = typer.Option(False, help="Auto-start embedded web bridge for web trackers"),
    bridge_ws_port: int = typer.Option(8000, help="WS port for embedded web bridge"),
    bridge_http_port: int = typer.Option(8001, help="HTTP port for embedded web bridge"),
    bridge_open_browser: bool = typer.Option(False, help="Open helper page when starting embedded bridge"),
    wait_ready: bool = typer.Option(False, help="Pause before showing stimuli (set up web trackers first)"),
):
    """One-button flow: calibration first, then tasks with the fitted models."""
    tracker_names = _normalize_tracker_names(trackers)
    if not tracker_names:
        raise typer.BadParameter("No trackers selected.")
    try:
        task_objs = resolve_tasks(tasks)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    calib_session = str(uuid.uuid4())
    task_session = str(uuid.uuid4())

    # Optional bridge for web trackers (kept alive across both stages).
    stop_bridge = None
    if start_bridge and any(t in tracker_names for t in ("webgazer", "turkergaze")):
        url, stopper = _start_embedded_web_bridge("0.0.0.0", bridge_ws_port, bridge_http_port, open_browser=bridge_open_browser)
        stop_bridge = stopper
        typer.echo(f"[eyetrack] Embedded web bridge at {url} (session_id will be injected automatically)")
        if wait_ready:
            try:
                input("[eyetrack] Press Enter to start calibration once web tracker page is streaming...")
            except EOFError:
                pass

    mpiris_cfg = dict(MPIRIS_DEFAULT_CFG)
    mpiris_cfg["out_width"] = width_px
    mpiris_cfg["out_height"] = height_px
    mpiris_cfg["preview"] = preview
    opt_cfg = dict(OPTIMEYES_DEFAULT_CFG)
    opt_cfg["out_width"] = width_px
    opt_cfg["out_height"] = height_px
    opt_cfg["preview"] = preview

    try:
        # Calibration stage
        calib_logger = RunLogger()
        calib_meta = SessionMeta(
            session_id=calib_session,
            os="unknown",
            width_px=width_px,
            height_px=height_px,
            ppi=96.0,
            distance_cm=60.0,
            protocol="9pt",
            dwell_ms=dwell_ms,
        )
        calib_logger.start_session(calib_meta)
        adapters = _prepare_adapters(
            tracker_names,
            pupil_host=pupil_host,
            pupil_req_port=pupil_req_port,
            mpiris_cfg=mpiris_cfg,
            optimeyes_cfg=opt_cfg,
            load_models=False,
        )
        orch = Orchestrator(adapters, calib_logger, calib_session, session_meta=calib_meta)
        orch.start_streams()
        try:
            orch.calibrate_all(dwell_ms=calib_meta.dwell_ms, gap_ms=300, fullscreen=fullscreen)
        finally:
            orch.stop_streams()
        _finalize_calibration_outputs(
            calib_session,
            tracker_names,
            width_px=calib_meta.width_px,
            height_px=calib_meta.height_px,
        )
        typer.echo(f"[eyetrack] Calibration session created: {calib_session}")

        # Benchmark tasks stage (re-load adapters with freshly fitted models)
        adapters = _prepare_adapters(
            tracker_names,
            pupil_host=pupil_host,
            pupil_req_port=pupil_req_port,
            mpiris_cfg=mpiris_cfg,
            optimeyes_cfg=opt_cfg,
            load_models=True,
        )
        task_logger = RunLogger()
        task_meta = SessionMeta(
            session_id=task_session,
            os="unknown",
            width_px=width_px,
            height_px=height_px,
            ppi=96.0,
            distance_cm=60.0,
            protocol="bench",
            dwell_ms=0,
        )
        task_logger.start_session(task_meta)
        orch = Orchestrator(adapters, task_logger, task_session, session_meta=task_meta)
        orch.start_streams()
        try:
            orch.run_tasks(task_objs, fullscreen=fullscreen)
        finally:
            orch.stop_streams()
            if stop_bridge:
                stop_bridge()
                stop_bridge = None
        typer.echo(f"[eyetrack] Task session created: {task_session}")
    finally:
        if stop_bridge:
            stop_bridge()


@app.command("web-bridge")
def web_bridge(
    host: str = typer.Option("0.0.0.0", help="Host for WS and HTTP servers"),
    ws_port: int = typer.Option(8000, help="Port for WebSocket bridge (uvicorn)"),
    http_port: int = typer.Option(8001, help="Port for static HTTP (serves tools/webgazer_bridge.html)"),
    open_browser: bool = typer.Option(True, help="Open bridge page in default browser"),
):
    """Start WebSocket bridge (for webgazer/turkergaze) and serve the helper page."""
    root = Path(__file__).resolve().parents[2]
    tools_root = root

    # WS server (uvicorn) in a thread
    config = uvicorn.Config("eyetrk.web_bridge.server:app", host=host, port=ws_port, log_level="warning")
    ws_server = uvicorn.Server(config)
    ws_thread = threading.Thread(target=ws_server.run, daemon=True)

    # HTTP static server for tools/
    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(*args, directory=str(tools_root), **kwargs)
    httpd = http.server.ThreadingHTTPServer((host, http_port), handler)
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)

    typer.echo(f"[web-bridge] Starting WS on {host}:{ws_port} and HTTP on {host}:{http_port}")
    ws_thread.start()
    http_thread.start()

    url = f"http://localhost:{http_port}/tools/webgazer_bridge.html"
    typer.echo(f"[web-bridge] Open this URL in a browser: {url}")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        ws_thread.join()
    except KeyboardInterrupt:
        typer.echo("\n[web-bridge] Shutting down...")
    finally:
        ws_server.should_exit = True
        try:
            httpd.shutdown()
        except Exception:
            pass


def _start_embedded_web_bridge(
    host: str,
    ws_port: int,
    http_port: int,
    open_browser: bool = False,
    session_id: str | None = None,
    tracker_id: str = "webgazer",
):
    root = Path(__file__).resolve().parents[2]
    tools_root = root

    config = uvicorn.Config("eyetrk.web_bridge.server:app", host=host, port=ws_port, log_level="warning")
    ws_server = uvicorn.Server(config)
    ws_thread = threading.Thread(target=ws_server.run, daemon=True)

    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(*args, directory=str(tools_root), **kwargs)
    httpd = http.server.ThreadingHTTPServer((host, http_port), handler)
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)

    ws_thread.start()
    http_thread.start()

    # === главное изменение: в URL прокидываем session_id, tracker_id и ws_port ===
    qs = []
    if session_id:
        qs.append(f"session_id={session_id}")
    if tracker_id:
        qs.append(f"tracker_id={tracker_id}")
    if ws_port:
        qs.append(f"ws_port={ws_port}")
    query = ("?" + "&".join(qs)) if qs else ""

    url = f"http://localhost:{http_port}/tools/webgazer_bridge.html{query}"
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    def stop():
        ws_server.should_exit = True
        try:
            httpd.shutdown()
        except Exception:
            pass

    return url, stop

@app.command()
def validate(
    trackers: List[str] = typer.Option(None, "--trackers", "-t", help="Trackers to use"),
):
    """Placeholder validate command (adds second command to keep subcommand mode on)."""
    typer.echo("validate: not implemented yet")


@app.command("run-tasks")
def run_tasks(
    tasks: List[str] = typer.Option(
        ["fixation-grid", "step-saccades", "smooth-pursuit"],
        "--tasks",
        "-k",
        help="Benchmark tasks to run sequentially",
    ),
    trackers: List[str] = typer.Option(None, "--trackers", "-t", help="Trackers to use"),
    fullscreen: bool = typer.Option(False, help="Show stimuli fullscreen"),
    width_px: int = typer.Option(1280, help="Screen width in pixels for benchmark stimuli"),
    height_px: int = typer.Option(720, help="Screen height in pixels for benchmark stimuli"),
    pupil_host: str = typer.Option("127.0.0.1", help="Pupil Remote host"),
    pupil_req_port: int = typer.Option(50020, help="Pupil Remote REQ port"),
    session_id: str | None = typer.Option(None, help="Optional fixed session id"),
):
    tracker_names = _normalize_tracker_names(trackers)
    if not tracker_names:
        raise typer.BadParameter("No trackers selected.")
    try:
        task_objs = resolve_tasks(tasks)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    session_id = session_id or str(uuid.uuid4())
    logger = RunLogger()
    meta = SessionMeta(
        session_id=session_id,
        os="unknown",
        width_px=width_px,
        height_px=height_px,
        ppi=96.0,
        distance_cm=60.0,
        protocol="bench",
        dwell_ms=0,
    )
    logger.start_session(meta)

    mpiris_cfg = dict(MPIRIS_DEFAULT_CFG)
    mpiris_cfg["out_width"] = width_px
    mpiris_cfg["out_height"] = height_px

    adapters = _prepare_adapters(
        tracker_names,
        pupil_host=pupil_host,
        pupil_req_port=pupil_req_port,
        mpiris_cfg=mpiris_cfg,
    )
    orch = Orchestrator(adapters, logger, session_id, session_meta=meta)
    orch.start_streams()
    try:
        orch.run_tasks(task_objs, fullscreen=fullscreen)
    finally:
        orch.stop_streams()

    typer.echo(f"Task session created: {session_id}")


@app.command()
def report(
    runs: Path = typer.Option(Path("runs"), "--runs", help="Directory with recorded sessions"),
    session: str | None = typer.Option(None, "--session", help="Specific session id"),
    tracker: str | None = typer.Option(None, "--tracker", help="Filter tracker id"),
):
    sessions = _list_sessions(runs, session)
    if not sessions:
        if session:
            typer.echo(f"No session '{session}' found in {runs}.")
        else:
            typer.echo(f"No sessions found in {runs}.")
        raise typer.Exit(code=1)

    rows: List[TaskMetrics] = []
    for sess in sessions:
        rows.extend(compute_session_metrics(sess))

    if tracker:
        rows = [r for r in rows if r.tracker_id == tracker]

    if not rows:
        typer.echo("No benchmark samples with targets were found.")
        raise typer.Exit(code=1)

    _print_metrics(rows)


def main():
    app()


def _normalize_tracker_names(trackers: List[str] | None) -> List[str]:
    if not trackers:
        candidates = list(DEFAULT_TRACKERS)
    else:
        candidates = list(trackers)
    blacklist = {"calibrate", "validate", "run-tasks", "report"}
    return [t for t in candidates if t not in blacklist]


def _prepare_adapters(
    tracker_names: List[str],
    pupil_host: str,
    pupil_req_port: int,
    mpiris_cfg: dict | None = None,
    optimeyes_cfg: dict | None = None,
    openseeface_cfg: dict | None = None,
    load_models: bool = True,
) -> dict[str, TrackerAdapter]:
    adapters: dict[str, TrackerAdapter] = {}
    mpiris_cfg = dict(mpiris_cfg or MPIRIS_DEFAULT_CFG)
    optimeyes_cfg = dict(optimeyes_cfg or OPTIMEYES_DEFAULT_CFG)
    openseeface_cfg = dict(openseeface_cfg or OPENSEEFACE_DEFAULT_CFG)

    for name in tracker_names:
        adapter, cfg = _instantiate_adapter(
            name,
            pupil_host=pupil_host,
            pupil_req_port=pupil_req_port,
            mpiris_cfg=mpiris_cfg,
            optimeyes_cfg=optimeyes_cfg,
            openseeface_cfg=openseeface_cfg,
        )
        adapter.initialize(cfg)
        adapters[name] = adapter

        if load_models:
            # Для webgazer НЕ грузим внешнюю модель: используем его собственную калибровку
            if name == "webgazer":
                continue

            model, model_path = _load_external_model_for(name)
            if model_path is None:
                continue
            if model is None:
                _safe_echo(f"[eyetrack] Failed to load calibration model {model_path}", err=True)
                continue
            try:
                adapter.set_external_model(model)
                _safe_echo(f"[eyetrack] Loaded calibration model for {name} ({model_path.name})")
            except Exception:
                _safe_echo(f"[eyetrack] Failed to set external model for {name}", err=True)

    if not adapters:
        _safe_echo("[eyetrack] No trackers initialized.", err=True)

    # Зарегистрировать адаптеры для веб-моста (без изменений)
    for name in ("webgazer", "turkergaze"):
        if name in adapters:
            bridge_server.adapters_registry[name] = adapters[name]

    return adapters



def _instantiate_adapter(
    name: str,
    pupil_host: str,
    pupil_req_port: int,
    mpiris_cfg: dict | None = None,
    optimeyes_cfg: dict | None = None,
    openseeface_cfg: dict | None = None,
) -> Tuple[TrackerAdapter, dict]:
    if name == "webgazer":
        return WebGazerAdapter(), {}
    if name == "turkergaze":
        return TurkerGazeAdapter(), {}
    if name == "pupilcore":
        return PupilCoreAdapter(), {"host": pupil_host, "req_port": pupil_req_port}
    if name == "optimeyes":
        cfg = optimeyes_cfg or OPTIMEYES_DEFAULT_CFG
        return OptimeyesAdapter(), {
            "fps": cfg.get("fps", 60.0),
            "camera_index": cfg.get("camera_index", 0),
            "width": cfg.get("width", 1280),
            "height": cfg.get("height", 720),
            "flip_x": cfg.get("flip_x", False),
            "flip_y": cfg.get("flip_y", False),
            "gain_x": cfg.get("gain_x", 1.0),
            "gain_y": cfg.get("gain_y", 1.0),
            "ema_alpha": cfg.get("ema_alpha", 0.2),
            "max_jump_norm": cfg.get("max_jump_norm", 0.12),
            "min_eye_w": cfg.get("min_eye_w", 0.02),
            "min_eye_h": cfg.get("min_eye_h", 0.01),
            "blink_ratio": cfg.get("blink_ratio", 0.16),
            "min_conf": cfg.get("min_conf", 0.6),
            "yaw_gain": cfg.get("yaw_gain", 0.32),
            "pitch_gain": cfg.get("pitch_gain", 0.06),
            "pose_alpha": cfg.get("pose_alpha", 0.02),
            "auto_gain": cfg.get("auto_gain", False),
            "auto_gain_alpha": cfg.get("auto_gain_alpha", 0.07),
            "auto_gain_margin": cfg.get("auto_gain_margin", 0.1),
            "out_width": cfg.get("out_width", 1280),
            "out_height": cfg.get("out_height", 720),
        }
    if name == "openseeface":
        cfg = openseeface_cfg or OPENSEEFACE_DEFAULT_CFG
        return OpenSeeFaceAdapter(), {
            "cmd": cfg.get("cmd", []),
            "flip_x": cfg.get("flip_x", True),
            "flip_y": cfg.get("flip_y", False),
            "gain_x": cfg.get("gain_x", 1.0),
            "gain_y": cfg.get("gain_y", 1.0),
            "out_width": cfg.get("out_width", 1280),
            "out_height": cfg.get("out_height", 720),
            "min_conf": cfg.get("min_conf", 0.6),
        }
    if name == "mpiris":
        cfg = mpiris_cfg or MPIRIS_DEFAULT_CFG
        return MpirisAdapter(), {
            "fps": cfg.get("fps", 60.0),
            "camera_index": cfg.get("camera_index", 0),
            "width": cfg.get("width", 1280),
            "height": cfg.get("height", 720),
            "flip_x": cfg.get("flip_x", False),
            "flip_y": cfg.get("flip_y", False),
            "gain_x": cfg.get("gain_x", 1.0),
            "gain_y": cfg.get("gain_y", 1.0),
            "ema_alpha": cfg.get("ema_alpha", 0.2),
            "head_alpha": cfg.get("head_alpha", 0.05),
            "head_gain_x": cfg.get("head_gain_x", 1.0),
            "head_gain_y": cfg.get("head_gain_y", 0.7),
            "head_comp": cfg.get("head_comp", True),
            "min_conf": cfg.get("min_conf", 0.6),
            "max_jump_norm": cfg.get("max_jump_norm", 0.2),
            "min_eye_w": cfg.get("min_eye_w", 0.02),
            "min_eye_h": cfg.get("min_eye_h", 0.008),
            "blink_ratio": cfg.get("blink_ratio", 0.18),
            "auto_gain": cfg.get("auto_gain", False),
            "auto_gain_alpha": cfg.get("auto_gain_alpha", 0.07),
            "auto_gain_margin": cfg.get("auto_gain_margin", 0.1),
            "preview": cfg.get("preview", False),
            "out_width": cfg.get("out_width", 1280),
            "out_height": cfg.get("out_height", 720),
        }
    raise typer.BadParameter(f"Unknown tracker: {name}")


def _load_external_model_for(tracker_name: str) -> Tuple[CalibModel | None, Path | None]:
    candidates = [
        Path("models") / f"{tracker_name}.json",
        Path("models/calib_model.json"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return CalibModel(**data), path
        except Exception as exc:
            typer.echo(f"[eyetrack] Failed to parse {path}: {exc}", err=True)
            return None, path
    return None, None


def _finalize_calibration_outputs(
    session_id: str,
    tracker_names: List[str],
    *,
    width_px: int,
    height_px: int,
) -> None:
    exported = _export_session_models(session_id, tracker_names)
    if not exported:
        return
    _write_model_reports(session_id, exported, width_px=width_px, height_px=height_px)


def _export_session_models(session_id: str, tracker_names: List[str]) -> dict[str, List[Path]]:
    data_dir = Path("data/models")
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)

    exported: dict[str, List[Path]] = {}
    for name in tracker_names:
        src = data_dir / f"{session_id}_{name}.json"
        if not src.exists():
            continue
        dst = out_dir / f"{name}.json"
        shutil.copyfile(src, dst)
        aliases = [dst]
        if name == "mpiris":
            alias = out_dir / "calib_model.json"
            shutil.copyfile(src, alias)
            aliases.append(alias)
        exported[name] = aliases
        _safe_echo(f"[eyetrack] Saved {name} model to {dst}")
    return exported


def _write_model_reports(
    session_id: str,
    exported: dict[str, List[Path]],
    *,
    width_px: int,
    height_px: int,
) -> None:
    runs_dir = Path("runs") / session_id
    for name, paths in exported.items():
        csv_path = runs_dir / f"samples_{name}.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            _safe_echo(f"[eyetrack] Failed to read {csv_path}: {exc}", err=True)
            continue
        try:
            _warn_low_variance(df, tracker_name=name)
            fit_out = fit_dataframe(df, width=width_px, height=height_px, per_stim_median=False)
        except Exception as exc:
            _safe_echo(f"[eyetrack] Failed to build diagnostics for {name}: {exc}", err=True)
            continue

        report_path = paths[0].with_suffix(".report.txt")
        _write_report_file(report_path, csv_path, fit_out.diag, fit_out.per_stim)
        for alias_model in paths[1:]:
            alias_report = alias_model.with_suffix(".report.txt")
            shutil.copyfile(report_path, alias_report)
        _safe_echo(f"[eyetrack] Saved {name} report to {report_path}")


def _write_report_file(path: Path, csv_path: Path, diag: dict, per_stim) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"CSV: {csv_path}\n")
        f.write(json.dumps(diag, indent=2))
        f.write("\n")
        if per_stim is not None:
            f.write("\nPer-stim error (px):\n")
            f.write(per_stim.to_string(index=False))
            f.write("\n")


def _list_sessions(runs_dir: Path, session_id: str | None) -> List[Path]:
    runs_path = Path(runs_dir)
    if session_id:
        target = runs_path / session_id
        return [target] if target.exists() else []
    if not runs_path.exists():
        return []
    dirs = [p for p in runs_path.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs


def _print_metrics(rows: List[TaskMetrics]) -> None:
    rows = sorted(rows, key=lambda r: (r.session_id, r.tracker_id, r.task_name))
    current_session = None
    for row in rows:
        if row.session_id != current_session:
            if current_session is not None:
                typer.echo("")
            typer.echo(f"Session {row.session_id}:")
            typer.echo("  tracker    task               samples valid drop%  mae(px)  prec(px)")
            current_session = row.session_id
        drop_pct = f"{row.drop_rate * 100:5.1f}"
        mae_str = _fmt_float(row.mae_px, 9)
        prec_str = _fmt_float(row.precision_px, 10)
        typer.echo(
            f"  {row.tracker_id:<10}{row.task_name:<18}"
            f"{row.samples:>8}{row.valid_samples:>7}"
            f"{drop_pct:>7}{mae_str}{prec_str}"
        )


def _fmt_float(val: float | None, width: int = 8) -> str:
    if val is None:
        return " " * (width - 3) + "n/a"
    return f"{val:>{width}.1f}"


def _warn_low_variance(df: pd.DataFrame, tracker_name: str, threshold: float = 0.03) -> None:
    valid = df[df.get("validity", 1) == 0]
    if valid.empty:
        return
    std_x = valid["x_norm"].std(skipna=True)
    std_y = valid["y_norm"].std(skipna=True)
    if (std_x is not None and std_x < threshold) or (std_y is not None and std_y < threshold):
        _safe_echo(
            f"[eyetrack] Warning: {tracker_name} gaze samples have very low variance "
            f"(std x={std_x:.3f}, y={std_y:.3f}). Calibration may be unreliable.",
            err=True,
        )


def _safe_echo(message: str, *, err: bool = False) -> None:
    try:
        typer.echo(message, err=err)
    except OSError:
        stream = sys.stderr if err else sys.stdout
        try:
            stream.write(f"{message}\n")
            stream.flush()
        except Exception:
            pass


if __name__ == "__main__":
    main()
