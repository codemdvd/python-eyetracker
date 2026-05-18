# eyetrk/cli.py
import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import List, Tuple
from urllib.parse import parse_qs, urlparse

import pandas as pd
import typer
import uvicorn

from eyetrk.core.tracker import Tracker
from eyetrk.core.types import CalibModel
from eyetrk.web_bridge import server as bridge_server

from .adapters.gazerecorder import GazerecorderAdapter
from .adapters.mpiris import MpirisAdapter
from .adapters.optimeyes import OptimeyesAdapter
from .adapters.webgazer import WebGazerAdapter
from .bench.report import TaskMetrics, compute_session_metrics
from .bench.postprocess import BenchmarkAnchorRecalibrator, BenchmarkBiasCorrector
from .bench.tasks import resolve_tasks
from .calib.fitting import fit_dataframe
from .io.logger import RunLogger
from .io.replay import FrameLabeler, SessionTimelineLabeler, load_frame_labels, load_timeline, load_video_meta
from .io.schema import SessionMeta
from .orchestrator import Orchestrator

app = typer.Typer(help="Unified eye tracking CLI")


def _detect_screen_size() -> tuple[int, int]:
    """Return (width, height) of the primary display in logical (CSS) pixels."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        if w > 100 and h > 100:
            return w, h
    except Exception:
        pass
    return 1920, 1080


_SCREEN_W, _SCREEN_H = _detect_screen_size()


DEFAULT_TRACKERS = ["webgazer", "gazerecorder", "optimeyes", "mpiris"]
ROOT = Path(__file__).resolve().parents[2]
MPIRIS_DEFAULT_CFG = {
    "fps": 60.0,
    "camera_index": 0,
    "camera_source": "webcam",
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
    "head_gain_y": 0.3,
    "min_conf": 0.6,
    "max_jump_norm": 0.12,
    "min_eye_w": 0.02,
    "min_eye_h": 0.008,
    "blink_ratio": 0.16,
    "auto_gain": False,
    "auto_gain_alpha": 0.02,
    "auto_gain_margin": 0.05,
    "preview": False,
    "preview_mirror": False,
    "cam_unmirror": True,
    "framing_scale": 0.82,
    "out_width": _SCREEN_W,
    "out_height": _SCREEN_H,
    "camera_backend": "auto",
}
OPTIMEYES_DEFAULT_CFG = {
    "fps": 60.0,
    "camera_index": 0,
    "camera_source": "webcam",
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
    "yaw_gain": 0.0,
    "pitch_gain": 0.0,
    "head_gain_x": 0.0,
    "head_gain_y": 0.0,
    "pose_alpha": 0.02,
    "auto_gain": False,
    "auto_gain_alpha": 0.07,
    "auto_gain_margin": 0.1,
    "preview": False,
    "out_width": _SCREEN_W,
    "out_height": _SCREEN_H,
}
MODEL_QUALITY_GATES = {
    "mpiris": {
        "max_selection_cv_mae_px": 360.0,
        "max_mae_px": 200.0,
        "max_rmse_px": 220.0,
        "min_r2_y": 0.5,
    },
    "optimeyes": {
        "max_selection_cv_mae_px": 320.0,
        "max_mae_px": 200.0,
        "max_rmse_px": 240.0,
        "min_r2_y": 0.45,
    },
    "webgazer": {
        "max_selection_cv_mae_px": 500.0,
        "max_mae_px": 400.0,
        "max_rmse_px": 450.0,
        "min_r2_y": 0.25,
    },
    "gazerecorder": {
        # SDK gaze has inherent spread; after poly2 correction expect ~250-350px MAE.
        "max_selection_cv_mae_px": 700.0,
        "max_mae_px": 600.0,
        "max_rmse_px": 680.0,
        "min_r2_y": 0.10,
    },
}
TRANSFER_MODEL_QUALITY_GATES = {
    "mpiris": {
        "max_validation_task_mae_px": 320.0,
        "max_validation_task_rmse_px": 380.0,
        "max_abs_validation_task_bias_px": 120.0,
    },
    "optimeyes": {
        "max_validation_task_mae_px": 300.0,
        "max_validation_task_rmse_px": 340.0,
        "max_abs_validation_task_bias_px": 120.0,
    },
}


@app.command()
def calibrate(
    trackers: List[str] = typer.Option(
        None,
        "--trackers",
        "-t",
        help="Trackers to use (repeatable). E.g.: -t mpiris -t webgazer",
    ),
    fullscreen: bool = typer.Option(False, help="Show stimuli fullscreen"),
    width_px: int = typer.Option(None, help=f"Screen width in pixels (default: auto-detect, currently {_SCREEN_W})"),
    height_px: int = typer.Option(None, help=f"Screen height in pixels (default: auto-detect, currently {_SCREEN_H})"),
    dwell_ms: int = typer.Option(1500, help="Dwell time per calibration point (ms)"),
    session_id: str | None = typer.Option(None, help="Optional base session id"),
    start_bridge: bool = typer.Option(False, help="Auto-start embedded web bridge for web trackers"),
    bridge_ws_port: int = typer.Option(8000, help="WS port for embedded web bridge"),
    bridge_http_port: int = typer.Option(8001, help="HTTP port for embedded web bridge"),
    bridge_open_browser: bool = typer.Option(False, help="Open helper page when starting embedded bridge"),
    wait_ready: bool = typer.Option(False, help="Pause before showing stimuli (set up web trackers first)"),
    auto_start_ms: int | None = typer.Option(None, help="Auto-start after N ms (skip Space/Enter)"),
    record_video: bool = typer.Option(False, help="Record source camera video for native trackers"),
    camera_source: str = typer.Option("webcam", help="Camera source for native trackers: 'webcam' or 'daheng'"),
):
    """Run 9-point calibration for selected trackers.

    When multiple webcam trackers (mpiris, optimeyes) are requested, they are
    run sequentially (one per sub-session) because they cannot share the camera.
    Use --record-video to let the second tracker replay from the recorded video
    of the first pass, which is faster and produces more comparable data.
    """
    tracker_names = _normalize_tracker_names(trackers)
    if not tracker_names:
        raise typer.BadParameter("No trackers selected.")

    width_px = width_px or _SCREEN_W
    height_px = height_px or _SCREEN_H

    cam_tracker_set = {"mpiris", "optimeyes"}
    web_tracker_set = {"gazerecorder", "webgazer"}
    native_phase = [t for t in tracker_names if t in cam_tracker_set]
    web_phase = [t for t in tracker_names if t in web_tracker_set]

    base_session_id = session_id or str(uuid.uuid4())

    # Build phases — each native tracker needs exclusive camera access.
    phases: list[tuple[str, list[str], list[str]]] = []
    if native_phase:
        if len(native_phase) == 1:
            phases.append((f"native-{native_phase[0]}", native_phase, []))
        elif record_video:
            live = _preferred_live_native_tracker(native_phase)
            replay = [n for n in native_phase if n != live]
            typer.echo(
                "[eyetrack] Recorded native mode: one live pass will be used, "
                "remaining native trackers will be replayed from the recorded video."
            )
            phases.append((f"native-{live}", [live], replay))
        else:
            typer.echo(
                "[eyetrack] Sequential native mode: webcam trackers run one-by-one "
                "(use --record-video for a faster single-pass approach)."
            )
            for name in native_phase:
                phases.append((f"native-{name}", [name], []))
    if web_phase:
        phases.append(("web", web_phase, []))
    if not phases:
        phases.append(("all", tracker_names, []))

    phase_total = len(phases)
    mpiris_cfg = dict(MPIRIS_DEFAULT_CFG)
    mpiris_cfg["out_width"] = width_px
    mpiris_cfg["out_height"] = height_px
    mpiris_cfg["camera_source"] = camera_source

    opt_cfg = dict(OPTIMEYES_DEFAULT_CFG)
    opt_cfg["out_width"] = width_px
    opt_cfg["out_height"] = height_px
    opt_cfg["camera_source"] = camera_source

    # Start a shared web bridge once (persists across all phases).
    all_web_like = [t for t in tracker_names if t in web_tracker_set]
    stop_bridge = None
    if start_bridge and all_web_like:
        bridge_ws_port = _pick_free_port(bridge_ws_port, host="0.0.0.0")
        bridge_http_port = _pick_free_port(bridge_http_port, host="0.0.0.0")
        bridge_tracker_id = all_web_like[0]
        if "webgazer" in all_web_like:
            bridge_tracker_id = "webgazer"
        url, stop_bridge = _start_embedded_web_bridge(
            host="0.0.0.0",
            ws_port=bridge_ws_port,
            ws_host="127.0.0.1",
            http_port=bridge_http_port,
            open_browser=bridge_open_browser,
            session_id=base_session_id,
            screen_w=width_px,
            screen_h=height_px,
            tracker_id=bridge_tracker_id,
            auto_start=True,
        )
        typer.echo(f"[eyetrack] Embedded web bridge at {url}")
        for other in all_web_like:
            if other == bridge_tracker_id:
                continue
            other_url = _bridge_url(
                http_port=bridge_http_port,
                ws_port=bridge_ws_port,
                ws_host="127.0.0.1",
                tracker_id=other,
                session_id=base_session_id,
                screen_w=width_px,
                screen_h=height_px,
                auto_start=True,
            )
            typer.echo(f"[eyetrack] Open {other_url} for {other}")
            if bridge_open_browser:
                try:
                    _open_browser_url(other_url, fullscreen=True)
                except Exception:
                    pass
        if wait_ready:
            try:
                input("[eyetrack] Press Enter to start calibration once web tracker page is streaming...")
            except EOFError:
                pass

    shown_ready = False
    session_ids_created: list[str] = []
    try:
        for idx, (phase_name, phase_trackers, replay_trackers) in enumerate(phases, start=1):
            if phase_total > 1:
                typer.echo(f"[eyetrack] Phase {idx}/{phase_total}: {phase_name} ({', '.join(phase_trackers)})")

            phase_session_id = (
                base_session_id if phase_total == 1
                else f"{base_session_id}_{phase_name}_calib"
            )
            typer.echo(f"[eyetrack] Session id: {phase_session_id}")

            phase_mpiris_cfg = dict(mpiris_cfg)
            phase_opt_cfg = dict(opt_cfg)
            live_native = next((t for t in phase_trackers if t in cam_tracker_set), None)
            if record_video and live_native:
                video_path = str(Path("runs") / phase_session_id / "source_camera.mp4")
                if live_native == "mpiris":
                    phase_mpiris_cfg["record_video_path"] = video_path
                else:
                    phase_opt_cfg["record_video_path"] = video_path

            logger = RunLogger()
            meta = SessionMeta(
                session_id=phase_session_id,
                os="unknown",
                width_px=width_px,
                height_px=height_px,
                ppi=96.0,
                distance_cm=60.0,
                protocol="9pt",
                dwell_ms=dwell_ms,
            )
            logger.start_session(meta)

            adapters = _prepare_adapters(
                phase_trackers,
                mpiris_cfg=phase_mpiris_cfg,
                optimeyes_cfg=phase_opt_cfg,
                load_models=False,
            )
            has_cam = any(t in cam_tracker_set for t in phase_trackers)
            cam_idx = mpiris_cfg.get("camera_index", 0) if has_cam else None
            cam_unmirror = bool(mpiris_cfg.get("cam_unmirror", True))
            framing_scale = float(mpiris_cfg.get("framing_scale", 0.82))

            orch = Orchestrator(adapters, logger, phase_session_id, session_meta=meta)
            try:
                orch.calibrate_all(
                    dwell_ms=meta.dwell_ms,
                    gap_ms=300,
                    fullscreen=fullscreen,
                    camera_index=cam_idx,
                    mirror_preview=False,
                    cam_unmirror=cam_unmirror,
                    framing_scale=framing_scale,
                    manage_streams=True,
                    auto_start_ms=auto_start_ms,
                    show_ready=not shown_ready,
                    camera_source=camera_source,
                )
            finally:
                orch.stop_streams()
                logger.close()
            shown_ready = True

            internal_trackers = [
                name for name, adapter in adapters.items()
                if getattr(adapter, "uses_internal_calibration", False)
            ]
            # GazeRecorder runs the GazeFlow wizard but also collects our 9-point stim
            # samples — include it in the offline refit so a correction model is saved.
            external_trackers = [
                n for n in phase_trackers
                if n not in internal_trackers or n == "gazerecorder"
            ]
            _offline_refit_session(
                phase_session_id,
                external_trackers,
                width_px=width_px,
                height_px=height_px,
                per_stim_median=True,
            )
            _write_internal_calibration_reports(
                phase_session_id,
                phase_trackers,
                width_px=width_px,
                height_px=height_px,
            )
            typer.echo(f"[eyetrack] Calibration session created: {phase_session_id}")
            session_ids_created.append(phase_session_id)

            if replay_trackers:
                typer.echo(
                    f"[eyetrack] Replaying recorded sessions for: {', '.join(replay_trackers)}"
                )
                for replay_name in replay_trackers:
                    replay_calib_session = f"{base_session_id}_native-{replay_name}_calib"
                    try:
                        replay_session(
                            source_session=phase_session_id,
                            trackers=[replay_name],
                            session_id=replay_calib_session,
                        )
                    except Exception as exc:
                        _safe_echo(
                            f"[eyetrack] Replay skipped for {replay_name}: {exc}",
                            err=True,
                        )
                        continue
                    typer.echo(f"[eyetrack] Calibration replay created: {replay_calib_session}")
                    session_ids_created.append(replay_calib_session)

    finally:
        if stop_bridge:
            stop_bridge()

    if len(session_ids_created) > 1:
        typer.echo("[eyetrack] Created calibration sessions:")
        for s in session_ids_created:
            typer.echo(f"  {s}")
    elif session_ids_created:
        typer.echo(f"Calibration session created: {session_ids_created[0]}")


@app.command("quickstart")
def quickstart(
    trackers: List[str] = typer.Option(
        None,
        "--trackers",
        "-t",
        help="Trackers to use (repeatable). E.g.: -t mpiris -t webgazer",
    ),
    tasks: List[str] = typer.Option(
        ["fixation-grid", "step-saccades", "smooth-pursuit"],
        "--tasks",
        "-k",
        help="Benchmark tasks to run after calibration",
    ),
    fullscreen: bool = typer.Option(False, help="Show stimuli fullscreen"),
    width_px: int = typer.Option(None, help=f"Screen width in pixels (default: auto-detect, currently {_SCREEN_W})"),
    height_px: int = typer.Option(None, help=f"Screen height in pixels (default: auto-detect, currently {_SCREEN_H})"),
    dwell_ms: int = typer.Option(1500, help="Dwell time per calibration point (ms)"),
    session_id: str | None = typer.Option(None, help="Optional base session id used as a prefix for calibration and task sessions"),
    start_bridge: bool = typer.Option(False, help="Auto-start embedded web bridge for web trackers"),
    bridge_ws_port: int = typer.Option(8000, help="WS port for embedded web bridge"),
    bridge_http_port: int = typer.Option(8001, help="HTTP port for embedded web bridge"),
    bridge_open_browser: bool = typer.Option(False, help="Open helper page when starting embedded bridge"),
    wait_ready: bool = typer.Option(False, help="Pause before showing stimuli (set up web trackers first)"),
    auto_start_ms: int | None = typer.Option(None, help="Auto-start after N ms (skip Space/Enter)"),
    record_video: bool = typer.Option(False, help="Record source camera video for native trackers"),
    camera_source: str = typer.Option("webcam", help="Camera source for native trackers: 'webcam' or 'daheng'"),
    camera_backend: str = typer.Option(
        "auto",
        "--camera-backend",
        help=(
            "cv2 backend for native webcam trackers: 'auto' (DSHOW/ANY alternating), "
            "'dshow' (exclusive, Windows only), 'msmf' (Media Foundation — shares camera "
            "with browser via Windows Camera Frame Server, required for --all unified phase)."
        ),
    ),
    all_trackers: bool = typer.Option(
        False,
        "--all",
        "-a",
        help=(
            "Run all 4 trackers (mpiris, optimeyes, webgazer, gazerecorder) in a single unified "
            "calibration+tasks session. Implies --record-video --start-bridge --bridge-open-browser "
            "and sets camera backend to msmf so Python and browser can share the webcam."
        ),
    ),
):
    """One-command flow: calibration -> tasks -> accuracy report.

    All trackers are run with as much parallelism as possible:
    - Web trackers (webgazer, gazerecorder) share one calibration session and
      one task session -- both browser tabs stream simultaneously.
    - Native webcam trackers (mpiris, optimeyes) require exclusive camera access
      and run sequentially, or use --record-video for a single-pass approach.
    - Use --all to run all 4 trackers in one pass: msmf backend allows Python and
      browser to share the webcam simultaneously.
    """
    if all_trackers:
        if not trackers:
            trackers = ["mpiris", "optimeyes", "webgazer", "gazerecorder"]
        record_video = True
        start_bridge = True
        bridge_open_browser = True
        if camera_backend == "auto":
            camera_backend = "msmf"

    tracker_names = _normalize_tracker_names(trackers)
    if not tracker_names:
        raise typer.BadParameter("No trackers selected.")

    width_px = width_px or _SCREEN_W
    height_px = height_px or _SCREEN_H

    try:
        task_objs = resolve_tasks(tasks)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    web_tracker_set = {"gazerecorder", "webgazer"}
    web_phase = [t for t in tracker_names if t in web_tracker_set]
    native_phase = [t for t in tracker_names if t not in web_tracker_set]
    base_session_id = session_id or str(uuid.uuid4())

    # With MSMF backend, Python and browser share the camera via Windows Camera Frame
    # Server, so native and web trackers can run in one unified calibration+task phase.
    unified_possible = camera_backend == "msmf" and native_phase and web_phase

    phases: list[tuple[str, list[str], bool, list[str]]] = []

    if unified_possible:
        live_native = _preferred_live_native_tracker(native_phase) if len(native_phase) > 1 else native_phase[0]
        replay_native = [name for name in native_phase if name != live_native]
        if replay_native:
            typer.echo(
                "[eyetrack] Unified phase (msmf): all trackers calibrate together. "
                f"{live_native} runs live (recording video); "
                f"{', '.join(replay_native)} will replay from that video."
            )
        else:
            typer.echo("[eyetrack] Unified phase (msmf): all trackers calibrate together.")
        phases.append(("all", [live_native] + web_phase, True, replay_native))
    else:
        if native_phase:
            if len(native_phase) == 1:
                phases.append((f"native-{native_phase[0]}", list(native_phase), False, []))
            elif record_video:
                live_native = _preferred_live_native_tracker(native_phase)
                replay_native = [name for name in native_phase if name != live_native]
                typer.echo(
                    "[eyetrack] Recorded native mode: one live pass will be used for native trackers, "
                    "remaining native trackers will be replayed from the recorded video."
                )
                phases.append((f"native-{live_native}", [live_native], False, replay_native))
            else:
                typer.echo("[eyetrack] Sequential native mode: running webcam trackers one-by-one.")
                for name in native_phase:
                    phases.append((f"native-{name}", [name], False, []))

        if web_phase:
            phases.append(("web", web_phase, True, []))
        if not phases:
            phases.append(("all", tracker_names, True, []))

    # ---- configs ----
    mpiris_cfg = dict(MPIRIS_DEFAULT_CFG)
    mpiris_cfg["out_width"] = width_px
    mpiris_cfg["out_height"] = height_px
    mpiris_cfg["camera_source"] = camera_source
    mpiris_cfg["camera_backend"] = camera_backend

    opt_cfg = dict(OPTIMEYES_DEFAULT_CFG)
    opt_cfg["out_width"] = width_px
    opt_cfg["out_height"] = height_px
    opt_cfg["camera_source"] = camera_source

    phase_results: list[tuple[str, str, str]] = []
    shown_ready = False
    phase_total = len(phases)

    def _run_phase(
        phase_name: str,
        phase_trackers: list[str],
        allow_web_bridge: bool,
        replay_trackers: list[str],
        *,
        force_phase_suffix: bool = False,
    ) -> None:
        nonlocal shown_ready
        calib_session, task_session = _quickstart_session_ids(
            base_session_id=base_session_id,
            phase_name=phase_name,
            phase_total=phase_total,
            force_phase_suffix=force_phase_suffix,
        )

        typer.echo(f"[eyetrack] Phase '{phase_name}' trackers: {', '.join(phase_trackers)}")
        typer.echo(f"[eyetrack] Calibration session id: {calib_session}")
        typer.echo(f"[eyetrack] Task session id: {task_session}")

        # ---- web bridge (kept alive across both stages) ----
        web_like = [t for t in phase_trackers if t in web_tracker_set]
        stop_bridge = None
        if start_bridge and allow_web_bridge and web_like:
            ws_port = _pick_free_port(bridge_ws_port, host="0.0.0.0")
            http_port = _pick_free_port(bridge_http_port, host="0.0.0.0")
            bridge_tracker_id = web_like[0]
            if len(web_like) > 1 and "webgazer" in web_like:
                bridge_tracker_id = "webgazer"

            url, stopper = _start_embedded_web_bridge(
                host="0.0.0.0",
                ws_port=ws_port,
                ws_host="127.0.0.1",
                http_port=http_port,
                open_browser=bridge_open_browser,
                session_id=calib_session,
                screen_w=width_px,
                screen_h=height_px,
                tracker_id=bridge_tracker_id,
                auto_start=True,
                internal_calib=False,
            )
            stop_bridge = stopper
            typer.echo(f"[eyetrack] Embedded web bridge at {url} (session_id injected)")

            if len(web_like) > 1:
                for other in web_like:
                    if other == bridge_tracker_id:
                        continue
                    other_url = _bridge_url(
                        http_port=http_port,
                        ws_port=ws_port,
                        ws_host="127.0.0.1",
                        tracker_id=other,
                        session_id=calib_session,
                        screen_w=width_px,
                        screen_h=height_px,
                        auto_start=True,
                        internal_calib=False,
                    )
                    typer.echo(f"[eyetrack] Open {other_url} for {other}")
                    if bridge_open_browser:
                        try:
                            _open_browser_url(other_url, fullscreen=True)
                        except Exception:
                            pass
            if wait_ready:
                try:
                    input("[eyetrack] Press Enter to start calibration once web tracker page is ready...")
                except EOFError:
                    pass

        cam_trackers = {"mpiris", "optimeyes"}
        has_cam = any(t in cam_trackers for t in phase_trackers)
        cam_idx = mpiris_cfg.get("camera_index", 0) if has_cam else None
        mirror_preview = False  # force non-mirrored preview for mpiris
        cam_unmirror = bool(mpiris_cfg.get("cam_unmirror", True))
        framing_scale = float(mpiris_cfg.get("framing_scale", 0.82))

        calib_logger: RunLogger | None = None
        task_logger: RunLogger | None = None
        try:
            # --------------------
            # Calibration stage
            # --------------------
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
            phase_mpiris_cfg = dict(mpiris_cfg)
            phase_opt_cfg = dict(opt_cfg)
            calib_video_path: Path | None = None
            if record_video:
                video_path = str(Path("runs") / calib_session / "source_camera.mp4")
                calib_video_path = Path(video_path)
                typer.echo(f"[eyetrack] Calibration video target: {calib_video_path}")
                phase_mpiris_cfg["record_video_path"] = video_path
                phase_opt_cfg["record_video_path"] = video_path

            adapters = _prepare_adapters(
                phase_trackers,
                mpiris_cfg=phase_mpiris_cfg,
                optimeyes_cfg=phase_opt_cfg,
                load_models=False,
            )
            orch = Orchestrator(adapters, calib_logger, calib_session, session_meta=calib_meta)
            try:
                orch.calibrate_all(
                    dwell_ms=calib_meta.dwell_ms,
                    gap_ms=300,
                    fullscreen=fullscreen,
                    camera_index=cam_idx,
                    mirror_preview=mirror_preview,
                    cam_unmirror=cam_unmirror,
                    framing_scale=framing_scale,
                    manage_streams=True,
                    auto_start_ms=auto_start_ms,
                    show_ready=not shown_ready,
                    camera_source=camera_source,
                )
            finally:
                # Streams are managed inside calibrate_all when manage_streams=True,
                # but we still stop adapters in case of early failures.
                orch.stop_streams()
                calib_logger.close()
            shown_ready = True

            internal_trackers = [
                name for name, adapter in adapters.items()
                if getattr(adapter, "uses_internal_calibration", False)
            ]
            # GazeRecorder runs the wizard but also collects stim samples — include in refit.
            external_trackers = [
                name for name in phase_trackers
                if name not in internal_trackers or name == "gazerecorder"
            ]
            # force_save=True: tasks should always use THIS session's calibration,
            # not whatever was stored from a previous session.
            _offline_refit_session(
                calib_session,
                external_trackers,
                width_px=calib_meta.width_px,
                height_px=calib_meta.height_px,
                per_stim_median=True,
                force_save=True,
            )
            _write_internal_calibration_reports(
                calib_session,
                phase_trackers,
                width_px=calib_meta.width_px,
                height_px=calib_meta.height_px,
            )
            time.sleep(0.4)  # small delay to ensure files are flushed before tasks
            if record_video and calib_video_path is not None and not calib_video_path.exists():
                typer.echo(f"[eyetrack] Warning: calibration video was requested but not created: {calib_video_path}", err=True)
            typer.echo(f"[eyetrack] Calibration session created: {calib_session}")

            # --------------------
            # Tasks stage
            # --------------------
            # All trackers in this phase share one task session.
            # Web trackers run simultaneously (each in its own browser tab, separate CSV files).
            task_batches: list[tuple[str, list[str], str]] = [(phase_name, phase_trackers, task_session)]

            for task_phase_name, task_phase_trackers, task_session_id in task_batches:
                task_web_like = [t for t in task_phase_trackers if t in web_tracker_set]
                if start_bridge and allow_web_bridge and task_web_like:
                    for task_tracker_id in task_web_like:
                        task_url = _bridge_url(
                            http_port=http_port,
                            ws_port=ws_port,
                            ws_host="127.0.0.1",
                            tracker_id=task_tracker_id,
                            session_id=task_session_id,
                            screen_w=width_px,
                            screen_h=height_px,
                            auto_start=True,
                            internal_calib=False,
                        )
                        typer.echo(f"[eyetrack] Task page for {task_tracker_id}: {task_url}")
                        # The calibration tab is already connected via WebSocket and will
                        # receive bridge_start(phase=tasks) automatically — no new tab needed.
                        # Only open a new tab if the bridge was NOT already running for calib.
                        already_have_tab = stop_bridge is not None and task_tracker_id in [
                            t for t in phase_trackers if t in web_tracker_set
                        ]
                        if bridge_open_browser and not already_have_tab:
                            try:
                                _open_browser_url(task_url, fullscreen=True)
                            except Exception:
                                pass
                    time.sleep(1.0)

                phase_task_mpiris_cfg = dict(mpiris_cfg)
                phase_task_opt_cfg = dict(opt_cfg)
                task_video_path: Path | None = None
                if record_video:
                    video_path = str(Path("runs") / task_session_id / "source_camera.mp4")
                    task_video_path = Path(video_path)
                    typer.echo(f"[eyetrack] Task video target: {task_video_path}")
                    phase_task_mpiris_cfg["record_video_path"] = video_path
                    phase_task_opt_cfg["record_video_path"] = video_path
                adapters = _prepare_adapters(
                    task_phase_trackers,
                    mpiris_cfg=phase_task_mpiris_cfg,
                    optimeyes_cfg=phase_task_opt_cfg,
                    load_models=True,
                    prefer_transfer_model=False,
                )

                task_logger = RunLogger()
                task_meta = SessionMeta(
                    session_id=task_session_id,
                    os="unknown",
                    width_px=width_px,
                    height_px=height_px,
                    ppi=96.0,
                    distance_cm=60.0,
                    protocol="bench",
                    dwell_ms=0,
                )
                task_logger.start_session(task_meta)

                orch = Orchestrator(adapters, task_logger, task_session_id, session_meta=task_meta)
                orch.start_streams()
                try:
                    orch.run_tasks(
                        task_objs,
                        fullscreen=fullscreen,
                        camera_index=cam_idx,
                        mirror_preview=mirror_preview,
                        cam_unmirror=cam_unmirror,
                        framing_scale=framing_scale,
                        auto_start_ms=auto_start_ms,
                        show_ready=False,
                        show_head_prompt=False,
                        camera_source=camera_source,
                    )
                finally:
                    orch.stop_streams()
                    task_logger.close()

                if record_video and task_video_path is not None and not task_video_path.exists():
                    typer.echo(f"[eyetrack] Warning: task video was requested but not created: {task_video_path}", err=True)
                typer.echo(f"[eyetrack] Task session created: {task_session_id}")
                phase_results.append((task_phase_name, calib_session, task_session_id))

            if replay_trackers:
                typer.echo(
                    f"[eyetrack] Replaying recorded sessions for native trackers: {', '.join(replay_trackers)}"
                )
                for replay_name in replay_trackers:
                    replay_calib_session = f"{base_session_id}_{replay_name}_calib"
                    replay_task_session = f"{base_session_id}_{replay_name}_tasks"
                    try:
                        replay_session(
                            source_session=calib_session,
                            trackers=[replay_name],
                            session_id=replay_calib_session,
                        )
                        replay_session(
                            source_session=task_session,
                            trackers=[replay_name],
                            session_id=replay_task_session,
                        )
                    except Exception as exc:
                        _safe_echo(
                            f"[eyetrack] Replay skipped for {replay_name}: {exc}",
                            err=True,
                        )
                        continue
                    phase_results.append((f"replay-{replay_name}", replay_calib_session, replay_task_session))

        finally:
            if task_logger is not None:
                task_logger.close()
            if calib_logger is not None:
                calib_logger.close()
            if stop_bridge:
                stop_bridge()

    for idx, (phase_name, phase_trackers, allow_bridge, replay_trackers) in enumerate(phases, start=1):
        typer.echo(f"[eyetrack] Running phase {idx}/{len(phases)}")
        _run_phase(phase_name, phase_trackers, allow_bridge, replay_trackers)

    # ---- final report ----
    all_task_rows: list[TaskMetrics] = []
    for _pn, _cs, _ts in phase_results:
        sess_path = Path("runs") / _ts
        if sess_path.exists():
            all_task_rows.extend(compute_session_metrics(sess_path))
    all_task_rows = [r for r in all_task_rows if r.mae_px is None or r.mae_px < 5000]

    typer.echo("\n" + "=" * 60)
    typer.echo("BENCHMARK COMPLETE")
    typer.echo("=" * 60 + "\n")
    _print_model_quality()
    if all_task_rows:
        _print_accuracy_best(all_task_rows)
    else:
        typer.echo("No task accuracy data to display.")
    if phase_results:
        typer.echo("Sessions created:")
        for _pn, _cs, _ts in phase_results:
            typer.echo(f"  {_pn}: calib={_cs}  tasks={_ts}")

@app.command("web-bridge")
def web_bridge(
    host: str = typer.Option("0.0.0.0", help="Host for WS and HTTP servers"),
    ws_port: int = typer.Option(8000, help="Port for WebSocket bridge (uvicorn)"),
    http_port: int = typer.Option(8001, help="Port for static HTTP (serves tools/webgazer_bridge.html)"),
    open_browser: bool = typer.Option(True, help="Open bridge page in default browser"),
):
    """Start the WebSocket bridge for browser trackers and serve the helper page."""
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
            _open_browser_url(url, fullscreen=True)
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
    auto_start: bool | None = None,
    ws_host: str | None = None,
    screen_w: int | None = None,
    screen_h: int | None = None,
    internal_calib: bool = False,
):
    import http.server
    import threading
    import webbrowser
    from pathlib import Path

    import uvicorn

    tools_root = Path(__file__).resolve().parents[2]  # project root

    # --- WS server (FastAPI/uvicorn) ---
    config = uvicorn.Config(
        "eyetrk.web_bridge.server:app",
        host=host,
        port=ws_port,
        log_level="warning",
    )
    ws_server = uvicorn.Server(config)
    ws_thread = threading.Thread(target=ws_server.run, daemon=True)

    # --- HTTP static server (serves /tools/*) ---
    class _NoCacheHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            super().end_headers()

        def log_message(self, *args):
            pass  # suppress request logs

    handler = lambda *args, **kwargs: _NoCacheHandler(
        *args, directory=str(tools_root), **kwargs
    )
    httpd = http.server.ThreadingHTTPServer((host, http_port), handler)
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)

    ws_thread.start()
    http_thread.start()
    time.sleep(0.6)  # Wait for servers to bind before opening browser

    url = _bridge_url(
        http_port=http_port,
        ws_port=ws_port,
        tracker_id=tracker_id,
        session_id=session_id,
        auto_start=auto_start,
        ws_host=ws_host,
        screen_w=screen_w,
        screen_h=screen_h,
        internal_calib=internal_calib,
    )

    if open_browser:
        try:
            _open_browser_url(url, fullscreen=True)
        except Exception:
            pass

    def stop():
        ws_server.should_exit = True
        try:
            httpd.shutdown()
        except Exception:
            pass

    return url, stop


def _bridge_page_for(tracker_id: str) -> str:
    if tracker_id == "gazerecorder":
        return "gazerecorder_bridge.html"
    return "webgazer_bridge.html"


def _bridge_url(
    *,
    http_port: int,
    ws_port: int,
    tracker_id: str,
    session_id: str | None,
    auto_start: bool | None = None,
    ws_host: str | None = None,
    screen_w: int | None = None,
    screen_h: int | None = None,
    internal_calib: bool = False,
) -> str:
    page = _bridge_page_for(tracker_id)
    qs: list[str] = []
    if session_id:
        qs.append(f"session_id={session_id}")
    if tracker_id:
        qs.append(f"tracker_id={tracker_id}")
    if ws_port:
        qs.append(f"ws_port={ws_port}")
    if ws_host:
        qs.append(f"ws_host={ws_host}")
    if screen_w:
        qs.append(f"screen_w={screen_w}")
    if screen_h:
        qs.append(f"screen_h={screen_h}")
    if auto_start is not None:
        qs.append(f"auto_start={1 if auto_start else 0}")
    if internal_calib:
        qs.append("internal_calib=1")
    # Bust browser cache so the latest bridge HTML is always loaded fresh.
    qs.append(f"v={int(time.time())}")
    query = f"?{'&'.join(qs)}" if qs else ""
    return f"http://localhost:{http_port}/tools/{page}{query}"


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
    width_px: int = typer.Option(None, help=f"Screen width in pixels (default: auto-detect, currently {_SCREEN_W})"),
    height_px: int = typer.Option(None, help=f"Screen height in pixels (default: auto-detect, currently {_SCREEN_H})"),
    session_id: str | None = typer.Option(None, help="Optional fixed session id"),
    auto_start_ms: int | None = typer.Option(None, help="Auto-start after N ms (skip Space/Enter)"),
    start_bridge: bool = typer.Option(False, help="Auto-start embedded web bridge for web trackers"),
    bridge_ws_port: int = typer.Option(8000, help="WS port for embedded web bridge"),
    bridge_http_port: int = typer.Option(8001, help="HTTP port for embedded web bridge"),
    bridge_open_browser: bool = typer.Option(False, help="Open helper page when starting embedded bridge"),
    wait_ready: bool = typer.Option(False, help="Pause before showing tasks (set up web trackers first)"),
    record_video: bool = typer.Option(False, help="Record source camera video for native trackers"),
    camera_source: str = typer.Option("webcam", help="Camera source for native trackers: 'webcam' or 'daheng'"),
):
    tracker_names = _normalize_tracker_names(trackers)
    if not tracker_names:
        raise typer.BadParameter("No trackers selected.")

    width_px = width_px or _SCREEN_W
    height_px = height_px or _SCREEN_H

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
    try:
        web_like = [t for t in tracker_names if t in ("gazerecorder", "webgazer")]
        stop_bridge = None
        if start_bridge and web_like:
            bridge_ws_port = _pick_free_port(bridge_ws_port, host="0.0.0.0")
            bridge_http_port = _pick_free_port(bridge_http_port, host="0.0.0.0")
            bridge_tracker_id = web_like[0]
            if len(web_like) > 1 and "webgazer" in web_like:
                bridge_tracker_id = "webgazer"

            url, stopper = _start_embedded_web_bridge(
                host="0.0.0.0",
                ws_port=bridge_ws_port,
                ws_host="127.0.0.1",
                http_port=bridge_http_port,
                open_browser=bridge_open_browser,
                session_id=session_id,
                screen_w=width_px,
                screen_h=height_px,
                tracker_id=bridge_tracker_id,
                auto_start=True,
            )
            stop_bridge = stopper
            typer.echo(f"[eyetrack] Embedded web bridge at {url}")

            if len(web_like) > 1:
                for other in web_like:
                    if other == bridge_tracker_id:
                        continue
                    other_url = _bridge_url(
                        http_port=bridge_http_port,
                        ws_port=bridge_ws_port,
                        ws_host="127.0.0.1",
                        tracker_id=other,
                        session_id=session_id,
                        screen_w=width_px,
                        screen_h=height_px,
                        auto_start=True,
                    )
                    typer.echo(f"[eyetrack] Open {other_url} for {other}")
                    if bridge_open_browser:
                        try:
                            _open_browser_url(other_url, fullscreen=True)
                        except Exception:
                            pass

            if wait_ready:
                try:
                    input("[eyetrack] Press Enter to start tasks once web tracker page is streaming...")
                except EOFError:
                    pass

        mpiris_cfg = dict(MPIRIS_DEFAULT_CFG)
        mpiris_cfg["out_width"] = width_px
        mpiris_cfg["out_height"] = height_px
        mpiris_cfg["camera_source"] = camera_source
        if record_video:
            mpiris_cfg["record_video_path"] = str(Path("runs") / session_id / "source_camera.mp4")
        cam_idx = mpiris_cfg.get("camera_index", 0)
        mirror_preview = False  # force non-mirrored preview for mpiris
        cam_unmirror = bool(mpiris_cfg.get("cam_unmirror", True))
        framing_scale = float(mpiris_cfg.get("framing_scale", 0.82))
        opt_cfg = dict(OPTIMEYES_DEFAULT_CFG)
        opt_cfg["out_width"] = width_px
        opt_cfg["out_height"] = height_px
        opt_cfg["camera_source"] = camera_source
        if record_video:
            opt_cfg["record_video_path"] = str(Path("runs") / session_id / "source_camera.mp4")

        adapters = _prepare_adapters(
            tracker_names,
            mpiris_cfg=mpiris_cfg,
            optimeyes_cfg=opt_cfg,
            prefer_transfer_model=True,
        )
        orch = Orchestrator(adapters, logger, session_id, session_meta=meta)
        orch.start_streams()
        try:
            orch.run_tasks(
                task_objs,
                fullscreen=fullscreen,
                camera_index=cam_idx,
                mirror_preview=mirror_preview,
                cam_unmirror=cam_unmirror,
                framing_scale=framing_scale,
                auto_start_ms=auto_start_ms,
                camera_source=camera_source,
            )
        finally:
            orch.stop_streams()
    finally:
        try:
            if "stop_bridge" in locals() and stop_bridge:
                stop_bridge()
        except Exception:
            pass
        logger.close()

    typer.echo(f"Task session created: {session_id}")


@app.command("replay-session")
def replay_session(
    source_session: str = typer.Option(..., "--source-session", help="Source session id or path under runs/"),
    trackers: List[str] = typer.Option(None, "--trackers", "-t", help="Native trackers to replay from video"),
    session_id: str | None = typer.Option(None, help="Optional output session id"),
    mpiris_conservative: bool = typer.Option(False, help="Replay mpiris with conservative settings (no auto_gain, no head_comp)"),
):
    tracker_names = _normalize_tracker_names(trackers)
    if not tracker_names:
        raise typer.BadParameter("No trackers selected.")
    unsupported = [name for name in tracker_names if name not in ("mpiris", "optimeyes")]
    if unsupported:
        raise typer.BadParameter(f"Replay from video currently supports only mpiris/optimeyes, got: {', '.join(unsupported)}")

    source_dir = _resolve_session_dir(source_session)
    meta_path = source_dir / "session.json"
    timeline_path = source_dir / "timeline.jsonl"
    video_path = _find_source_video(source_dir)
    if not meta_path.exists():
        raise typer.BadParameter(f"Session meta not found: {meta_path}")
    if not timeline_path.exists():
        raise typer.BadParameter(f"Timeline not found: {timeline_path}")
    if video_path is None or not video_path.exists():
        raise typer.BadParameter(f"Recorded source video not found in {source_dir}")
    video_meta_path = Path(str(video_path) + ".json")

    with open(meta_path, "r", encoding="utf-8") as f:
        source_meta = SessionMeta(**json.load(f))
    events = load_timeline(timeline_path)
    video_meta = load_video_meta(video_meta_path)
    video_start_ms = int(video_meta.get("start_timestamp_ms", 0) or 0)
    if video_start_ms <= 0:
        raise typer.BadParameter(f"Video metadata missing start_timestamp_ms: {video_meta_path}")

    out_session_id = session_id or str(uuid.uuid4())
    logger = RunLogger()
    out_meta = source_meta.model_copy(update={"session_id": out_session_id})
    logger.start_session(out_meta)

    timeline_labelers = {name: SessionTimelineLabeler(events) for name in tracker_names}
    shared_frame_labels = _find_best_source_frame_labels(source_dir)
    frame_labelers: dict[str, FrameLabeler] = {}
    benchmark_correctors = {name: BenchmarkBiasCorrector() for name in tracker_names}
    benchmark_recalibrators = {name: BenchmarkAnchorRecalibrator() for name in tracker_names}
    for name in tracker_names:
        source_csv = source_dir / f"samples_{name}.csv"
        frame_labels = load_frame_labels(source_csv)
        if not frame_labels:
            frame_labels = shared_frame_labels
        if frame_labels:
            frame_labelers[name] = FrameLabeler(frame_labels)
    mpiris_cfg = dict(MPIRIS_DEFAULT_CFG)
    mpiris_cfg["out_width"] = out_meta.width_px
    mpiris_cfg["out_height"] = out_meta.height_px
    mpiris_cfg["preview"] = False
    mpiris_cfg["video_path"] = str(video_path)
    mpiris_cfg["video_start_timestamp_ms"] = video_start_ms
    source_unmirror_applied = bool(video_meta.get("cam_unmirror_applied", True))
    if source_unmirror_applied:
        mpiris_cfg["cam_unmirror"] = False
    if mpiris_conservative:
        mpiris_cfg["auto_gain"] = False
        mpiris_cfg["head_comp"] = False

    opt_cfg = dict(OPTIMEYES_DEFAULT_CFG)
    opt_cfg["out_width"] = out_meta.width_px
    opt_cfg["out_height"] = out_meta.height_px
    opt_cfg["preview"] = False
    opt_cfg["video_path"] = str(video_path)
    opt_cfg["video_start_timestamp_ms"] = video_start_ms

    load_models = out_meta.protocol != "9pt"
    adapters = _prepare_adapters(
        tracker_names,
        mpiris_cfg=mpiris_cfg,
        optimeyes_cfg=opt_cfg,
        load_models=load_models,
        prefer_transfer_model=load_models and out_meta.protocol != "9pt",
    )
    try:
        def on_sample(sample):
            frame_labeler = frame_labelers.get(sample.tracker_id)
            if frame_labeler is not None:
                sample = frame_labeler.apply(sample)
            else:
                tracker_labeler = timeline_labelers.get(sample.tracker_id)
                if tracker_labeler is not None:
                    sample = tracker_labeler.apply(sample)
            sample = benchmark_correctors[sample.tracker_id].apply(sample)
            sample = benchmark_recalibrators[sample.tracker_id].apply(sample)
            logger.write_sample(sample)

        for adapter in adapters.values():
            adapter.start_stream(on_sample, session_id=out_session_id)

        while True:
            threads = [
                getattr(adapter, "_thread", None)
                for adapter in adapters.values()
            ]
            alive = False
            for thread in threads:
                if thread is not None and thread.is_alive():
                    alive = True
                    break
            if not alive:
                break
            time.sleep(0.05)
    finally:
        for adapter in adapters.values():
            try:
                adapter.stop()
            except Exception:
                pass
        logger.close()

    if out_meta.protocol in ("9pt", "25pt"):
        _offline_refit_session(
            out_session_id,
            tracker_names,
            width_px=out_meta.width_px,
            height_px=out_meta.height_px,
            per_stim_median=True,
        )

    typer.echo(f"Replay session created: {out_session_id}")


@app.command("optimize-session-pair")
def optimize_session_pair(
    calib_session: str = typer.Option(..., "--calib-session", help="Calibration session id or path"),
    task_session: str = typer.Option(..., "--task-session", help="Task session id or path used for transfer validation"),
    trackers: List[str] = typer.Option(None, "--trackers", "-t", help="Trackers to optimize"),
    per_stim_median: bool = typer.Option(True, help="Use robust per-point aggregation for calibration fitting"),
):
    tracker_names = _normalize_tracker_names(trackers)
    if not tracker_names:
        raise typer.BadParameter("No trackers selected.")

    calib_dir = _resolve_session_dir(calib_session)
    task_dir = _resolve_session_dir(task_session)
    calib_meta_path = calib_dir / "session.json"
    if not calib_meta_path.exists():
        raise typer.BadParameter(f"Session meta not found: {calib_meta_path}")

    with open(calib_meta_path, "r", encoding="utf-8") as f:
        calib_meta = SessionMeta(**json.load(f))

    typer.echo(f"[eyetrack] Calibration session: {calib_dir.name}")
    typer.echo(f"[eyetrack] Validation task session: {task_dir.name}")
    for name in tracker_names:
        calib_csv = calib_dir / f"samples_{name}.csv"
        task_csv = task_dir / f"samples_{name}.csv"
        if not calib_csv.exists():
            _safe_echo(f"[eyetrack] Pair optimization skipped for {name}: missing calibration CSV {calib_csv}", err=True)
            continue
        if not task_csv.exists():
            _safe_echo(f"[eyetrack] Pair optimization skipped for {name}: missing task CSV {task_csv}", err=True)
            continue
        try:
            calib_df = pd.read_csv(calib_csv)
            task_df = pd.read_csv(task_csv)
            fit_out = fit_dataframe(
                calib_df,
                width=calib_meta.width_px,
                height=calib_meta.height_px,
                per_stim_median=per_stim_median,
                validation_df=task_df,
            )
        except Exception as exc:
            _safe_echo(f"[eyetrack] Pair optimization failed for {name}: {exc}", err=True)
            continue

        saved = _save_transfer_model(
            tracker_name=name,
            model_payload=fit_out.model.model_dump(),
            csv_path=calib_csv,
            fit_out=fit_out,
            source_label=f"paired refit from {calib_dir.name} + {task_dir.name}",
        )
        if not saved:
            continue
        diag = fit_out.diag
        task_mae = float(diag.get("validation_task_mae_px", float("nan")))
        task_rmse = float(diag.get("validation_task_rmse_px", float("nan")))
        typer.echo(
            f"[eyetrack] Saved transfer model for {name}: "
            f"validation_task_mae_px={task_mae:.1f}, validation_task_rmse_px={task_rmse:.1f}"
        )


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

    _print_model_quality(tracker_filter=tracker)

    if not rows:
        typer.echo("\nNo benchmark task samples with targets were found.")
        return

    _print_metrics(rows)


@app.command()
def accuracy(
    runs: Path = typer.Option(Path("runs"), "--runs", help="Directory with recorded sessions"),
    session: str | None = typer.Option(None, "--session", help="Specific session id (repeatable: use multiple --session flags)"),
    tracker: str | None = typer.Option(None, "--tracker", help="Filter by tracker id"),
    best: bool = typer.Option(False, "--best", help="Show only the best (lowest MAE) session per tracker"),
):
    """Compute accuracy, RMSE, precision and data-loss metrics for benchmark task sessions.

    Metrics are reported both in pixels and degrees of visual angle
    (using the viewing distance and PPI stored in session.json).
    Use --best to get a clean per-tracker comparison table.
    """
    from .bench.metrics import px_to_deg

    sessions = _list_sessions(runs, session)
    if not sessions:
        typer.echo(f"No sessions found in {runs}.")
        raise typer.Exit(code=1)

    rows: List[TaskMetrics] = []
    for sess in sessions:
        rows.extend(compute_session_metrics(sess))

    if tracker:
        rows = [r for r in rows if r.tracker_id == tracker]

    # Drop sessions with absurd MAE (> 5000px — broken/incompatible model artifact)
    rows = [r for r in rows if r.mae_px is None or r.mae_px < 5000]

    if not rows:
        typer.echo("No benchmark task samples with targets were found.")
        raise typer.Exit(code=1)

    if best:
        _print_accuracy_best(rows)
    else:
        _print_accuracy_full(rows)


def _print_accuracy_full(rows: List[TaskMetrics]) -> None:
    from .bench.metrics import px_to_deg

    rows = sorted(rows, key=lambda r: (r.tracker_id, r.session_id, r.task_name))
    current_key = None
    for row in rows:
        key = (row.tracker_id, row.session_id)
        if key != current_key:
            if current_key is not None:
                typer.echo("")
            deg_per_px = 1.0 / max(px_to_deg(1.0, ppi=row.ppi, distance_cm=row.distance_cm), 1e-9)
            typer.echo(
                f"{row.tracker_id}  [{row.session_id}]"
                f"  (dist={row.distance_cm:.0f}cm, ppi={row.ppi:.0f}, 1deg~{1/deg_per_px*1:.0f}px)"
            )
            typer.echo(
                f"  {'task':<18}{'n':>6}{'valid':>6}{'drop%':>7}"
                f"{'mae(px)':>9}{'mae(deg)':>9}{'rmse(px)':>10}{'prec(px)':>10}"
            )
            current_key = key

        mae_deg = px_to_deg(row.mae_px, ppi=row.ppi, distance_cm=row.distance_cm) if row.mae_px is not None else None
        typer.echo(
            f"  {row.task_name:<18}{row.samples:>6}{row.valid_samples:>6}"
            f"{row.drop_rate*100:>6.1f}%"
            f"{_fmt_float(row.mae_px, 9)}{_fmt_float(mae_deg, 8)}"
            f"{_fmt_float(row.rmse_px, 10)}{_fmt_float(row.precision_px, 10)}"
        )


def _print_accuracy_best(rows: List[TaskMetrics]) -> None:
    """Show the best (lowest overall MAE) session per tracker, with per-task breakdown."""
    from .bench.metrics import px_to_deg
    import math

    # Aggregate total MAE per (tracker, session) — weighted by valid sample count
    from collections import defaultdict
    agg: dict[tuple[str, str], dict] = defaultdict(lambda: {"mae_sum": 0.0, "n": 0, "rows": []})
    for row in rows:
        if row.mae_px is None or row.valid_samples == 0:
            continue
        key = (row.tracker_id, row.session_id)
        agg[key]["mae_sum"] += row.mae_px * row.valid_samples
        agg[key]["n"] += row.valid_samples
        agg[key]["rows"].append(row)

    # Pick best session per tracker
    best_per_tracker: dict[str, tuple[str, float]] = {}
    for (tr, sess), v in agg.items():
        if v["n"] == 0:
            continue
        overall_mae = v["mae_sum"] / v["n"]
        if tr not in best_per_tracker or overall_mae < best_per_tracker[tr][1]:
            best_per_tracker[tr] = (sess, overall_mae)

    if not best_per_tracker:
        typer.echo("No usable task sessions found.")
        return

    ordered_trackers = ["mpiris", "optimeyes", "webgazer", "gazerecorder"]
    ordered_trackers += [t for t in sorted(best_per_tracker) if t not in ordered_trackers]

    typer.echo("Best session per tracker (lowest weighted MAE across all tasks):")
    typer.echo("")

    for tr in ordered_trackers:
        if tr not in best_per_tracker:
            continue
        best_sess, _ = best_per_tracker[tr]
        key = (tr, best_sess)
        task_rows = sorted(agg[key]["rows"], key=lambda r: r.task_name)
        sample_row = task_rows[0]
        ppi = sample_row.ppi
        dist = sample_row.distance_cm
        deg_label = f"{px_to_deg(1.0, ppi=ppi, distance_cm=dist)*40:.1f}"  # 1° in px (40px reference)
        deg_px = 1.0 / px_to_deg(1.0, ppi=ppi, distance_cm=dist)  # px per degree

        typer.echo("-" * 72)
        typer.echo(
            f"  Tracker:  {tr}"
            f"   Session: {best_sess}"
        )
        typer.echo(
            f"  Viewing distance: {dist:.0f} cm   PPI: {ppi:.0f}   1 deg ~ {deg_px:.0f} px"
        )
        typer.echo(
            f"  {'Task':<20}{'Samples':>8}{'Valid':>7}{'Drop%':>7}"
            f"{'MAE px':>9}{'MAE deg':>9}{'RMSE px':>9}{'Prec px':>9}"
        )

        # Per-task rows
        all_preds: list[float] = []
        all_targets_x: list[float] = []
        all_n = 0
        all_valid = 0
        all_mae_sum = 0.0
        all_rmse_sq_sum = 0.0
        all_prec_sq_sum = 0.0

        for row in task_rows:
            mae_deg = px_to_deg(row.mae_px, ppi=ppi, distance_cm=dist) if row.mae_px is not None else None
            typer.echo(
                f"  {row.task_name:<20}{row.samples:>8}{row.valid_samples:>7}"
                f"{row.drop_rate*100:>6.1f}%"
                f"{_fmt_float(row.mae_px, 9)}{_fmt_float(mae_deg, 8)}"
                f"{_fmt_float(row.rmse_px, 9)}{_fmt_float(row.precision_px, 9)}"
            )
            if row.mae_px is not None and row.valid_samples > 0:
                all_n += row.samples
                all_valid += row.valid_samples
                all_mae_sum += row.mae_px * row.valid_samples
                if row.rmse_px is not None:
                    all_rmse_sq_sum += (row.rmse_px ** 2) * row.valid_samples

        # Overall weighted row
        if all_valid > 0:
            overall_mae = all_mae_sum / all_valid
            overall_mae_deg = px_to_deg(overall_mae, ppi=ppi, distance_cm=dist)
            overall_rmse = math.sqrt(all_rmse_sq_sum / all_valid) if all_rmse_sq_sum > 0 else None
            overall_drop = 1.0 - all_valid / all_n if all_n > 0 else 0.0
            typer.echo(
                f"  {'OVERALL':<20}{all_n:>8}{all_valid:>7}"
                f"{overall_drop*100:>6.1f}%"
                f"{_fmt_float(overall_mae, 9)}{_fmt_float(overall_mae_deg, 8)}"
                f"{_fmt_float(overall_rmse, 9)}"
            )
        typer.echo("")


def main():
    app()


def _normalize_tracker_names(trackers: List[str] | None) -> List[str]:
    if not trackers:
        candidates = list(DEFAULT_TRACKERS)
    else:
        candidates = list(trackers)
    blacklist = {"calibrate", "validate", "run-tasks", "report"}
    return [t for t in candidates if t not in blacklist]


def _preferred_live_native_tracker(trackers: List[str]) -> str:
    if "mpiris" in trackers:
        return "mpiris"
    return trackers[0]


def _prepare_adapters(
    tracker_names: List[str],
    mpiris_cfg: dict | None = None,
    optimeyes_cfg: dict | None = None,
    load_models: bool = True,
    prefer_transfer_model: bool = False,
    gazerecorder_cfg: dict | None = None,
) -> dict[str, Tracker]:
    adapters: dict[str, Tracker] = {}
    mpiris_cfg = dict(mpiris_cfg or MPIRIS_DEFAULT_CFG)
    optimeyes_cfg = dict(optimeyes_cfg or OPTIMEYES_DEFAULT_CFG)

    for name in tracker_names:
        adapter, cfg = _instantiate_adapter(
            name,
            mpiris_cfg=mpiris_cfg,
            optimeyes_cfg=optimeyes_cfg,
            gazerecorder_cfg=gazerecorder_cfg,
        )
        adapter.initialize(cfg)
        adapters[name] = adapter

        if load_models:
            model, model_path = _load_external_model_for(name, prefer_transfer=prefer_transfer_model)
            if model_path is None:
                continue
            if model is None:
                _safe_echo(f"[eyetrack] Failed to load model {model_path}", err=True)
                continue
            try:
                adapter.set_external_model(model)
                model_kind = "transfer" if model_path.name.endswith(".transfer.json") else "calibration"
                _safe_echo(f"[eyetrack] Loaded {model_kind} model for {name} ({model_path.name})")
            except Exception:
                _safe_echo(f"[eyetrack] Failed to set external model for {name}", err=True)

    if not adapters:
        _safe_echo("[eyetrack] No trackers initialized.", err=True)

    for name in ("webgazer", "gazerecorder"):
        if name in adapters:
            bridge_server.adapters_registry[name] = adapters[name]

    return adapters



def _instantiate_adapter(
    name: str,
    mpiris_cfg: dict | None = None,
    optimeyes_cfg: dict | None = None,
    gazerecorder_cfg: dict | None = None,
) -> Tuple[Tracker, dict]:
    if name == "webgazer":
        return WebGazerAdapter(), {}
    if name == "gazerecorder":
        size_cfg = mpiris_cfg or MPIRIS_DEFAULT_CFG
        gr_cfg = gazerecorder_cfg or {}
        return GazerecorderAdapter(), {
            "out_width": size_cfg.get("out_width", 1280),
            "out_height": size_cfg.get("out_height", 720),
            "use_internal_calibration": bool(gr_cfg.get("use_internal_calibration", False)),
        }
    if name == "optimeyes":
        cfg = optimeyes_cfg or OPTIMEYES_DEFAULT_CFG
        return OptimeyesAdapter(), {
            "fps": cfg.get("fps", 60.0),
            "camera_index": cfg.get("camera_index", 0),
            "camera_source": cfg.get("camera_source", "webcam"),
            "daheng_device_index": cfg.get("daheng_device_index", 1),
            "daheng_exposure_us": cfg.get("daheng_exposure_us", 5000.0),
            "daheng_gain_db": cfg.get("daheng_gain_db", 12.0),
            "video_path": cfg.get("video_path"),
            "video_start_timestamp_ms": cfg.get("video_start_timestamp_ms"),
            "record_video_path": cfg.get("record_video_path"),
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
            "yaw_gain": cfg.get("yaw_gain", 0.0),
            "pitch_gain": cfg.get("pitch_gain", 0.0),
            "head_gain_x": cfg.get("head_gain_x", 0.0),
            "head_gain_y": cfg.get("head_gain_y", 0.0),
            "pose_alpha": cfg.get("pose_alpha", 0.02),
            "auto_gain": cfg.get("auto_gain", False),
            "auto_gain_alpha": cfg.get("auto_gain_alpha", 0.07),
            "auto_gain_margin": cfg.get("auto_gain_margin", 0.1),
            "out_width": cfg.get("out_width", 1280),
            "out_height": cfg.get("out_height", 720),
        }
    if name == "mpiris":
        cfg = mpiris_cfg or MPIRIS_DEFAULT_CFG
        return MpirisAdapter(), {
            "fps": cfg.get("fps", 60.0),
            "camera_index": cfg.get("camera_index", 0),
            "camera_source": cfg.get("camera_source", "webcam"),
            "daheng_device_index": cfg.get("daheng_device_index", 1),
            "daheng_exposure_us": cfg.get("daheng_exposure_us", 5000.0),
            "daheng_gain_db": cfg.get("daheng_gain_db", 12.0),
            "video_path": cfg.get("video_path"),
            "video_start_timestamp_ms": cfg.get("video_start_timestamp_ms"),
            "record_video_path": cfg.get("record_video_path"),
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


def _load_external_model_for(
    tracker_name: str,
    *,
    prefer_transfer: bool = False,
) -> Tuple[CalibModel | None, Path | None]:
    candidates: list[Path] = []
    if prefer_transfer:
        candidates.append(Path("models") / f"{tracker_name}.transfer.json")
    candidates.append(Path("models") / f"{tracker_name}.json")
    if tracker_name == "mpiris":
        candidates.append(Path("models/calib_model.json"))
    for path in candidates:
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            report_diag = _load_report_diag(path.with_suffix(".report.txt"))
            if report_diag is not None:
                if path.name.endswith(".transfer.json"):
                    failures = _transfer_gate_failures(tracker_name, report_diag)
                else:
                    failures = _quality_gate_failures(tracker_name, report_diag)
                if failures:
                    kind = "transfer" if path.name.endswith(".transfer.json") else "calibration"
                    typer.echo(f"[eyetrack] Ignoring invalid {kind} model {path.name}: {'; '.join(failures)}", err=True)
                    continue
            return CalibModel(**data), path
        except Exception as exc:
            typer.echo(f"[eyetrack] Failed to parse {path}: {exc}", err=True)
            return None, path
    return None, None


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


def _write_rejected_report_file(
    path: Path,
    csv_path: Path,
    diag: dict,
    per_stim,
    reasons: List[str],
    *,
    source_label: str,
) -> None:
    _write_report_file(path, csv_path, diag, per_stim)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\nQuality gate: rejected\n")
        f.write(f"Source: {source_label}\n")
        for reason in reasons:
            f.write(f"- {reason}\n")


def _force_save_model(
    *,
    tracker_name: str,
    model_payload: dict,
    csv_path: Path,
    fit_out,
    source_label: str,
) -> List[Path]:
    """Save model unconditionally (used by quickstart so the just-calibrated model is always used)."""
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{tracker_name}.json"

    diag = fit_out.diag if fit_out is not None else {}
    per_stim = fit_out.per_stim if fit_out is not None else None

    failures = _quality_gate_failures(tracker_name, diag)
    if failures:
        _safe_echo(f"[eyetrack] {tracker_name} calibration failed quality gates: {'; '.join(failures)}", err=True)
        _safe_echo(f"[eyetrack] Tasks will use the previous stored model instead.", err=True)
        return []

    aliases = [model_path]
    if tracker_name == "mpiris":
        aliases.append(models_dir / "calib_model.json")

    for path in aliases:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(model_payload, f, ensure_ascii=False, indent=2)
        if fit_out is not None and csv_path.exists():
            report_path = path.with_suffix(".report.txt")
            _write_report_file(report_path, csv_path, diag, per_stim)

    _safe_echo(f"[eyetrack] Saved {tracker_name} model (forced) to {model_path}")
    return aliases


def _save_promoted_model(
    *,
    tracker_name: str,
    model_payload: dict,
    csv_path: Path,
    fit_out,
    source_label: str,
) -> List[Path]:
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{tracker_name}.json"

    diag = fit_out.diag if fit_out is not None else {}
    per_stim = fit_out.per_stim if fit_out is not None else None
    should_save, reasons = _should_promote_model(tracker_name, diag, model_path)
    if not should_save:
        reject_path = models_dir / f"{tracker_name}.rejected.report.txt"
        if fit_out is not None and csv_path.exists():
            _write_rejected_report_file(reject_path, csv_path, diag, per_stim, reasons, source_label=source_label)
            existing_diag = _load_report_diag(model_path.with_suffix(".report.txt"))
            if existing_diag is None or _quality_gate_failures(tracker_name, existing_diag):
                _write_rejected_report_file(
                    model_path.with_suffix(".report.txt"),
                    csv_path,
                    diag,
                    per_stim,
                    reasons,
                    source_label=source_label,
                )
        _safe_echo(f"[eyetrack] Skipped {tracker_name} model promotion: {'; '.join(reasons)}")
        return []

    aliases = [model_path]
    if tracker_name == "mpiris":
        aliases.append(models_dir / "calib_model.json")

    for path in aliases:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(model_payload, f, ensure_ascii=False, indent=2)
        if fit_out is not None and csv_path.exists():
            report_path = path.with_suffix(".report.txt")
            _write_report_file(report_path, csv_path, diag, per_stim)

    _safe_echo(f"[eyetrack] Saved {tracker_name} model to {model_path}")
    return aliases


def _save_transfer_model(
    *,
    tracker_name: str,
    model_payload: dict,
    csv_path: Path,
    fit_out,
    source_label: str,
) -> List[Path]:
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{tracker_name}.transfer.json"

    diag = fit_out.diag if fit_out is not None else {}
    per_stim = fit_out.per_stim if fit_out is not None else None
    should_save, reasons = _should_promote_transfer_model(tracker_name, diag, model_path)
    if not should_save:
        reject_path = models_dir / f"{tracker_name}.transfer.rejected.report.txt"
        if fit_out is not None and csv_path.exists():
            _write_rejected_report_file(reject_path, csv_path, diag, per_stim, reasons, source_label=source_label)
            existing_diag = _load_report_diag(model_path.with_suffix(".report.txt"))
            if existing_diag is None or _transfer_gate_failures(tracker_name, existing_diag):
                _write_rejected_report_file(
                    model_path.with_suffix(".report.txt"),
                    csv_path,
                    diag,
                    per_stim,
                    reasons,
                    source_label=source_label,
                )
        _safe_echo(f"[eyetrack] Skipped {tracker_name} transfer model promotion: {'; '.join(reasons)}")
        return []

    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model_payload, f, ensure_ascii=False, indent=2)
    if fit_out is not None and csv_path.exists():
        report_path = model_path.with_suffix(".report.txt")
        _write_report_file(report_path, csv_path, diag, per_stim)

    _safe_echo(f"[eyetrack] Saved {tracker_name} transfer model to {model_path}")
    return [model_path]


def _should_promote_model(tracker_name: str, diag: dict, model_path: Path) -> tuple[bool, List[str]]:
    reasons = _quality_gate_failures(tracker_name, diag)
    if reasons:
        return False, reasons

    report_path = model_path.with_suffix(".report.txt")
    existing_diag = _load_report_diag(report_path)
    if existing_diag is None:
        return True, []
    if _model_file_needs_refresh(model_path, diag):
        return True, []

    existing_failures = _quality_gate_failures(tracker_name, existing_diag)
    if existing_failures:
        return True, []

    if _model_rank(diag) < _model_rank(existing_diag):
        return True, []

    return False, [f"existing {tracker_name} model is better or equal"]


def _should_promote_transfer_model(tracker_name: str, diag: dict, model_path: Path) -> tuple[bool, List[str]]:
    reasons = _transfer_gate_failures(tracker_name, diag)
    if reasons:
        return False, reasons

    report_path = model_path.with_suffix(".report.txt")
    existing_diag = _load_report_diag(report_path)
    if existing_diag is None:
        return True, []
    if _model_file_needs_refresh(model_path, diag):
        return True, []

    existing_failures = _transfer_gate_failures(tracker_name, existing_diag)
    if existing_failures:
        return True, []

    if _transfer_model_rank(diag) < _transfer_model_rank(existing_diag):
        return True, []

    return False, [f"existing {tracker_name} transfer model is better or equal"]


def _quality_gate_failures(tracker_name: str, diag: dict) -> List[str]:
    gate = MODEL_QUALITY_GATES.get(tracker_name)
    if not gate or not diag:
        return []

    failures: List[str] = []
    cv_mae = float(diag.get("selection_cv_mae_px", float("inf")))
    mae = float(diag.get("mae_px", float("inf")))
    rmse = float(diag.get("rmse_px", float("inf")))
    r2_y = float(diag.get("r2_y", float("-inf")))

    if cv_mae > float(gate["max_selection_cv_mae_px"]):
        failures.append(f"selection_cv_mae_px={cv_mae:.1f} > {gate['max_selection_cv_mae_px']:.1f}")
    if mae > float(gate["max_mae_px"]):
        failures.append(f"mae_px={mae:.1f} > {gate['max_mae_px']:.1f}")
    if rmse > float(gate["max_rmse_px"]):
        failures.append(f"rmse_px={rmse:.1f} > {gate['max_rmse_px']:.1f}")
    if r2_y < float(gate["min_r2_y"]):
        failures.append(f"r2_y={r2_y:.3f} < {gate['min_r2_y']:.3f}")
    return failures


def _transfer_gate_failures(tracker_name: str, diag: dict) -> List[str]:
    gate = TRANSFER_MODEL_QUALITY_GATES.get(tracker_name)
    if not gate or not diag:
        return []

    failures: List[str] = []
    task_mae = float(diag.get("validation_task_mae_px", float("inf")))
    task_rmse = float(diag.get("validation_task_rmse_px", float("inf")))
    bias_x = abs(float(diag.get("validation_task_bias_x_px", float("inf"))))
    bias_y = abs(float(diag.get("validation_task_bias_y_px", float("inf"))))

    if task_mae > float(gate["max_validation_task_mae_px"]):
        failures.append(
            f"validation_task_mae_px={task_mae:.1f} > {gate['max_validation_task_mae_px']:.1f}"
        )
    if task_rmse > float(gate["max_validation_task_rmse_px"]):
        failures.append(
            f"validation_task_rmse_px={task_rmse:.1f} > {gate['max_validation_task_rmse_px']:.1f}"
        )
    max_bias = float(gate["max_abs_validation_task_bias_px"])
    if bias_x > max_bias:
        failures.append(f"|validation_task_bias_x_px|={bias_x:.1f} > {max_bias:.1f}")
    if bias_y > max_bias:
        failures.append(f"|validation_task_bias_y_px|={bias_y:.1f} > {max_bias:.1f}")
    return failures


def _model_rank(diag: dict) -> tuple[float, float, float, float]:
    return (
        float(diag.get("selection_cv_mae_px", float("inf"))),
        float(diag.get("mae_px", float("inf"))),
        float(diag.get("rmse_px", float("inf"))),
        -float(diag.get("r2_y", float("-inf"))),
    )


def _transfer_model_rank(diag: dict) -> tuple[float, float, float, float, float]:
    return (
        float(diag.get("validation_task_mae_px", float("inf"))),
        float(diag.get("validation_task_rmse_px", float("inf"))),
        abs(float(diag.get("validation_task_bias_y_px", float("inf")))),
        abs(float(diag.get("validation_task_bias_x_px", float("inf")))),
        float(diag.get("selection_cv_mae_px", float("inf"))),
    )


def _load_report_diag(report_path: Path) -> dict | None:
    if not report_path.exists():
        return None
    try:
        text = report_path.read_text(encoding="utf-8")
    except Exception:
        return None
    start = text.find("{")
    if start < 0:
        return None
    end = text.find("\n\nPer-stim")
    if end < 0:
        end = len(text)
    try:
        return json.loads(text[start:end].strip())
    except Exception:
        return None


def _load_report_session(report_path: Path) -> str | None:
    """Return the session id extracted from the 'CSV: runs/<id>/...' line in a report file."""
    if not report_path.exists():
        return None
    try:
        first_line = report_path.read_text(encoding="utf-8").splitlines()[0]
    except Exception:
        return None
    if not first_line.startswith("CSV:"):
        return None
    # e.g. "CSV: runs/abc123_calib/samples_mpiris.csv"
    parts = first_line.split("/")
    for i, p in enumerate(parts):
        if p == "runs" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _model_file_needs_refresh(model_path: Path, diag: dict) -> bool:
    if not model_path.exists():
        return True
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except Exception:
        return True
    params = payload.get("params") or {}
    input_features = params.get("input_features")
    expected_features = diag.get("input_features")
    if expected_features and input_features != expected_features:
        return True
    if "selected_degree" in diag and params.get("degree") != diag.get("selected_degree"):
        return True
    if "selection_cv_mae_px" in diag and "selection_cv_mae_px" not in params:
        return True
    return False


def _write_internal_calibration_reports(
    session_id: str,
    tracker_names: List[str],
    *,
    width_px: int,
    height_px: int,
) -> None:
    runs_dir = Path("runs") / session_id
    for name in tracker_names:
        if name not in ("webgazer",):
            continue
        csv_path = runs_dir / f"samples_{name}.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            _safe_echo(f"[eyetrack] Internal calib report skipped for {name}: failed to read {csv_path} ({exc})", err=True)
            continue
        try:
            if "stim_id" in df.columns:
                df = df[df["stim_id"].astype(str).str.startswith("calib_")]
            fit_out = fit_dataframe(df, width=width_px, height=height_px, per_stim_median=True)
        except Exception as exc:
            _safe_echo(f"[eyetrack] Internal calib report failed for {name}: {exc}", err=True)
            continue
        report_path = runs_dir / f"calib_{name}_report.txt"
        _write_report_file(report_path, csv_path, fit_out.diag, fit_out.per_stim)
        _safe_echo(f"[eyetrack] Saved {name} calibration report to {report_path}")
        try:
            saved_paths = _save_promoted_model(
                tracker_name=name,
                model_payload=fit_out.model.model_dump(),
                csv_path=csv_path,
                fit_out=fit_out,
                source_label=f"internal calibration correction from {session_id}",
            )
            if saved_paths:
                _safe_echo(f"[eyetrack] Saved {name} correction model to {saved_paths[0]}")
        except Exception as exc:
            _safe_echo(f"[eyetrack] Failed to save internal correction model for {name}: {exc}", err=True)


def _offline_refit_session(
    session_id: str,
    tracker_names: List[str],
    *,
    width_px: int,
    height_px: int,
    per_stim_median: bool = True,
    force_save: bool = False,
) -> None:
    """Re-fit models from recorded samples and overwrite models/<tracker>.json (and aliases).

    force_save=True skips the "is existing model better?" check and always promotes
    the freshly-fitted model. Use this in quickstart so the just-calibrated session
    is always used for the following tasks, regardless of stored model history.
    """
    runs_dir = Path("runs") / session_id
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    _web_trackers = {"webgazer", "gazerecorder"}
    for name in tracker_names:
        csv_path = runs_dir / f"samples_{name}.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            _safe_echo(f"[eyetrack] Offline refit skipped for {name}: failed to read {csv_path} ({exc})", err=True)
            continue
        # Web trackers have many noisy samples per calibration point — fitting on
        # raw samples (not per-point medians) uses all available data and gives
        # better regression than reducing to 8-9 noisy median points.
        use_per_stim_median = False if name in _web_trackers else per_stim_median
        try:
            if "stim_id" in df.columns:
                df = df[df["stim_id"].astype(str).str.startswith("calib_")]
            fit_out = fit_dataframe(df, width=width_px, height=height_px, per_stim_median=use_per_stim_median)
        except Exception as exc:
            _safe_echo(f"[eyetrack] Offline refit failed for {name}: {exc}", err=True)
            continue

        try:
            if force_save:
                saved_paths = _force_save_model(
                    tracker_name=name,
                    model_payload=fit_out.model.model_dump(),
                    csv_path=csv_path,
                    fit_out=fit_out,
                    source_label=f"quickstart refit from {session_id}",
                )
            else:
                saved_paths = _save_promoted_model(
                    tracker_name=name,
                    model_payload=fit_out.model.model_dump(),
                    csv_path=csv_path,
                    fit_out=fit_out,
                    source_label=f"offline refit from {session_id}",
                )
            if saved_paths:
                _safe_echo(f"[eyetrack] Offline refit saved for {name} ({saved_paths[0].name})")
        except Exception as exc:
            _safe_echo(f"[eyetrack] Failed to save offline refit for {name}: {exc}", err=True)


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


def _resolve_session_dir(session: str) -> Path:
    source = Path(session)
    if source.exists():
        return source
    candidate = Path("runs") / session
    if candidate.exists():
        return candidate
    raise typer.BadParameter(f"Session not found: {session}")


def _find_source_video(session_dir: Path) -> Path | None:
    preferred = session_dir / "source_camera.mp4"
    if preferred.exists():
        return preferred
    candidates = sorted(session_dir.glob("source*.mp4"))
    return candidates[0] if candidates else None


def _session_sample_count(session_id: str, tracker_name: str) -> int:
    csv_path = Path("runs") / session_id / f"samples_{tracker_name}.csv"
    if not csv_path.exists():
        return 0
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return 0
    return int(len(df))


def _print_model_quality(tracker_filter: str | None = None) -> None:
    """Print a summary table of the current calibration/transfer models in models/."""
    models_dir = Path("models")
    all_trackers = ["mpiris", "optimeyes", "webgazer", "gazerecorder"]
    if tracker_filter:
        all_trackers = [t for t in all_trackers if t == tracker_filter]

    rows = []
    for name in all_trackers:
        for suffix, kind in [(".transfer.json", "transfer"), (".json", "calib")]:
            model_path = models_dir / f"{name}{suffix}"
            report_path = model_path.with_suffix(".report.txt")
            if not model_path.exists():
                continue
            diag = _load_report_diag(report_path)
            if diag is None:
                continue
            session_hint = _load_report_session(report_path)
            rows.append((name, kind, diag, session_hint))
            break  # prefer transfer over calib

    if not rows:
        typer.echo("No calibration models found in models/.")
        return

    typer.echo("Calibration models:")
    typer.echo(
        f"  {'tracker':<13}{'kind':<10}"
        f"{'mae(px)':>9}{'cv_mae':>9}{'r2_x':>7}{'r2_y':>7}  features"
    )
    for name, kind, diag, session_hint in rows:
        if kind == "transfer":
            mae_val = diag.get("validation_task_mae_px")
        else:
            mae_val = diag.get("mae_px")
        mae = _fmt_float(mae_val, 9)
        cv_mae = _fmt_float(diag.get("selection_cv_mae_px"), 9)
        r2x = _fmt_float(diag.get("r2_x"), 7)
        r2y = _fmt_float(diag.get("r2_y"), 7)
        feats = ", ".join(diag.get("input_features") or [])
        typer.echo(f"  {name:<13}{kind:<10}{mae}{cv_mae}{r2x}{r2y}  {feats}")
        if session_hint:
            typer.echo(f"  {'':13}{'session':<10}  {session_hint}")
    typer.echo("")


def _print_metrics(rows: List[TaskMetrics]) -> None:
    rows = sorted(rows, key=lambda r: (r.session_id, r.tracker_id, r.task_name))
    current_session = None
    for row in rows:
        if row.session_id != current_session:
            if current_session is not None:
                typer.echo("")
            typer.echo(f"Session {row.session_id}:")
            typer.echo("  tracker    task               samples valid drop%  mae(px) rmse(px)  prec(px)")
            current_session = row.session_id
        drop_pct = f"{row.drop_rate * 100:5.1f}"
        mae_str = _fmt_float(row.mae_px, 8)
        rmse_str = _fmt_float(row.rmse_px, 9)
        prec_str = _fmt_float(row.precision_px, 10)
        typer.echo(
            f"  {row.tracker_id:<10}{row.task_name:<18}"
            f"{row.samples:>8}{row.valid_samples:>7}"
            f"{drop_pct:>7}{mae_str}{rmse_str}{prec_str}"
        )


def _fmt_float(val: float | None, width: int = 8) -> str:
    if val is None:
        return " " * (width - 3) + "n/a"
    return f"{val:>{width}.1f}"


def _warn_low_variance(df: pd.DataFrame, tracker_name: str, threshold: float = 0.03) -> None:
    if "validity" in df.columns:
        valid = df[df["validity"] == 0]
    else:
        valid = df
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



def _is_port_free(host: str, port: int) -> bool:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, port))
            return True
        finally:
            sock.close()
    except OSError:
        return False


def _pick_free_port(port: int, host: str = "0.0.0.0", max_tries: int = 10) -> int:
    for offset in range(max_tries + 1):
        candidate = port + offset
        if _is_port_free(host, candidate):
            if candidate != port:
                _safe_echo(f"[eyetrack] Port {port} busy, using {candidate} instead.")
            return candidate
    return port


def _find_browser_exe() -> str | None:
    roots = []
    for key in ("LOCALAPPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)"):
        val = os.environ.get(key)
        if val:
            roots.append(Path(val))
    candidates = [
        Path(r) / "Microsoft/Edge/Application/msedge.exe" for r in roots
    ] + [
        Path(r) / "Google/Chrome/Application/chrome.exe" for r in roots
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def _find_best_source_frame_labels(session_dir: Path) -> dict[int, dict]:
    candidates = [
        session_dir / "samples_mpiris.csv",
        session_dir / "samples_optimeyes.csv",
    ]
    candidates.extend(
        path for path in sorted(session_dir.glob("samples_*.csv"))
        if path not in candidates
    )
    for path in candidates:
        labels = load_frame_labels(path)
        if labels:
            return labels
    return {}


def _quickstart_session_ids(
    *,
    base_session_id: str | None,
    phase_name: str,
    phase_total: int,
    force_phase_suffix: bool = False,
) -> tuple[str, str]:
    if not base_session_id:
        return str(uuid.uuid4()), str(uuid.uuid4())

    phase_suffix = ""
    if phase_total > 1 or force_phase_suffix:
        phase_suffix = f"_{phase_name}"
    calib_session = f"{base_session_id}{phase_suffix}_calib"
    task_session = f"{base_session_id}{phase_suffix}_tasks"
    return calib_session, task_session


def _grant_camera_to_default_edge_profile(origin: str) -> None:
    """Write camera and fullscreen permissions for origin into the user's default Edge/Chrome profile.

    Only effective if the browser is not yet running (Chromium reads preferences
    on profile load). Safe to call even when the browser is already running --
    the permission will take effect on the next browser launch.
    Granting fullscreen allows the bridge page to call requestFullscreen()
    programmatically without a user gesture.
    """
    local_app = os.environ.get("LOCALAPPDATA", "")
    candidates = []
    if local_app:
        candidates.append(Path(local_app) / "Microsoft" / "Edge" / "User Data" / "Default" / "Preferences")
        candidates.append(Path(local_app) / "Google" / "Chrome" / "User Data" / "Default" / "Preferences")
    key = f"{origin},*"
    for prefs_path in candidates:
        if not prefs_path.parent.exists():
            continue
        try:
            prefs = json.loads(prefs_path.read_text(encoding="utf-8")) if prefs_path.exists() else {}
        except Exception:
            prefs = {}
        exceptions = (
            prefs
            .setdefault("profile", {})
            .setdefault("content_settings", {})
            .setdefault("exceptions", {})
        )
        exceptions.setdefault("media_stream_camera", {})[key] = {"setting": 1}
        exceptions.setdefault("fullscreen", {})[key] = {"setting": 1}
        try:
            prefs_path.write_text(json.dumps(prefs), encoding="utf-8")
        except Exception:
            pass


def _find_chromium_exe() -> str | None:
    """Return path to Chrome or Edge executable on Windows, or None."""
    if sys.platform != "win32":
        return None
    candidates = []
    local_app = os.environ.get("LOCALAPPDATA", "")
    prog_files = os.environ.get("PROGRAMFILES", r"C:\Program Files")
    prog_files_x86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
    if local_app:
        candidates += [
            Path(local_app) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
            Path(local_app) / "Google" / "Chrome" / "Application" / "chrome.exe",
        ]
    candidates += [
        Path(prog_files_x86) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
        Path(prog_files) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
        Path(prog_files) / "Google" / "Chrome" / "Application" / "chrome.exe",
        Path(prog_files_x86) / "Google" / "Chrome" / "Application" / "chrome.exe",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _grant_camera_to_profile_dir(origin: str, profile_dir: Path) -> None:
    """Write camera + fullscreen permissions for origin into a Chrome user-data-dir."""
    prefs_path = profile_dir / "Default" / "Preferences"
    try:
        prefs_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    try:
        prefs = json.loads(prefs_path.read_text(encoding="utf-8")) if prefs_path.exists() else {}
    except Exception:
        prefs = {}
    key = f"{origin},*"
    exceptions = (
        prefs
        .setdefault("profile", {})
        .setdefault("content_settings", {})
        .setdefault("exceptions", {})
    )
    exceptions.setdefault("media_stream_camera", {})[key] = {"setting": 1}
    exceptions.setdefault("fullscreen", {})[key] = {"setting": 1}
    try:
        prefs_path.write_text(json.dumps(prefs), encoding="utf-8")
    except Exception:
        pass


def _open_browser_url(url: str, fullscreen: bool = False) -> None:
    """Open url in a browser app window (--app mode).

    For GazeRecorder (tracker_id=gazerecorder in the URL), a fresh per-session
    Chrome profile directory is used so that GazeRecorder's cloud server has no
    cached calibration model and always shows its own calibration wizard.

    For all other trackers, the DEFAULT profile is used so multiple tracker windows
    can share the same Chrome process and thus share the camera device.
    """
    parsed = None
    try:
        parsed = urlparse(url)
    except Exception:
        pass

    origin = f"{parsed.scheme}://{parsed.netloc}" if parsed else ""

    # Detect GazeRecorder: use a wiped fresh profile to force GR wizard.
    tracker_id_qs = ""
    try:
        tracker_id_qs = parse_qs(parsed.query).get("tracker_id", [""])[0] if parsed else ""
    except Exception:
        pass
    use_fresh_profile = tracker_id_qs == "gazerecorder"

    profile_dir: Path | None = None
    if use_fresh_profile:
        profile_dir = _browser_profile_dir_for_url(url)
        try:
            if profile_dir.exists():
                shutil.rmtree(profile_dir, ignore_errors=True)
            profile_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            profile_dir = None
        if profile_dir and origin:
            _grant_camera_to_profile_dir(origin, profile_dir)
    else:
        try:
            if origin:
                _grant_camera_to_default_edge_profile(origin)
        except Exception:
            pass

    if fullscreen:
        exe = _find_chromium_exe()
        if exe:
            cmd = [
                exe,
                f"--app={url}",
                "--start-maximized",
                "--no-first-run",
                "--disable-default-browser-check",
            ]
            if profile_dir:
                cmd.append(f"--user-data-dir={profile_dir}")
            try:
                subprocess.Popen(cmd)
                return
            except Exception:
                pass
    try:
        webbrowser.open(url)
    except Exception:
        pass


def _browser_profile_dir_for_url(url: str) -> Path:
    try:
        parsed = urlparse(url)
        tracker_id = parse_qs(parsed.query).get("tracker_id", ["browser"])[0]
    except Exception:
        tracker_id = "browser"
    tracker_id = "".join(ch for ch in tracker_id if ch.isalnum() or ch in ("-", "_")) or "browser"
    return ROOT / f".eyetrk-browser-profile-{tracker_id}"


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



