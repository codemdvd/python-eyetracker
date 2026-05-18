import asyncio
import math
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect
from pydantic import BaseModel
import sys
import queue

adapters_registry: Dict[str, Any] = {}
recv_counts: Dict[str, int] = {}

# Outgoing messages: python -> browser (per-tracker queues).
send_queues: Dict[str, "queue.Queue[dict]"] = {}
send_counts: Dict[str, int] = {}
SEND_QUEUE_MAXSIZE = 1000


@asynccontextmanager
async def lifespan(app: FastAPI):
    if sys.platform == "win32":
        loop = asyncio.get_event_loop()

        def _suppress_conn_reset(loop, context):
            exc = context.get("exception")
            if isinstance(exc, (ConnectionResetError, BrokenPipeError)):
                return
            loop.default_exception_handler(context)

        loop.set_exception_handler(_suppress_conn_reset)
    yield


app = FastAPI(lifespan=lifespan)


class WGSample(BaseModel):
    tracker_id: str
    session_id: str
    timestamp_ms: int
    frame_id: int | None = None
    head_x: float | None = None
    head_y: float | None = None
    head_z: float | None = None
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    x_norm: float | None = None
    y_norm: float | None = None
    confidence: float | None = None
    validity: int = 0
    stim_id: str | None = None
    event: str | None = None
    task_name: str | None = None
    target_x_px: float | None = None
    target_y_px: float | None = None
    inner_h: int | None = None
    inner_w: int | None = None
    fullscreen: int | None = None
    gr_gaze_x: float | None = None
    gr_gaze_y: float | None = None
    gr_doc_x: float | None = None
    gr_doc_y: float | None = None


@app.get("/health")
def health():
    return {"ok": True}


_PASSTHROUGH_EVENTS = {"internal_calibration", "sdk_calibrated", "sdk_load_failed"}


def _sanitize_browser_sample(sample: WGSample) -> WGSample | None:
    if sample.x_norm is None or sample.y_norm is None:
        if sample.event in _PASSTHROUGH_EVENTS:
            return sample
        sample.validity = max(1, int(sample.validity))
        if sample.confidence is None:
            sample.confidence = 0.0
        return sample

    if not math.isfinite(sample.x_norm) or not math.isfinite(sample.y_norm):
        return None

    if sample.confidence is not None and not math.isfinite(sample.confidence):
        sample.confidence = None

    # Drop clearly corrupted points, and mark mild out-of-bounds samples invalid.
    if sample.x_norm < -0.25 or sample.x_norm > 1.25 or sample.y_norm < -0.25 or sample.y_norm > 1.25:
        return None

    in_bounds = 0.0 <= sample.x_norm <= 1.0 and 0.0 <= sample.y_norm <= 1.0
    if not in_bounds:
        sample.validity = max(1, int(sample.validity))
        sample.x_norm = min(1.0, max(0.0, sample.x_norm))
        sample.y_norm = min(1.0, max(0.0, sample.y_norm))
        sample.confidence = min(float(sample.confidence or 0.25), 0.25)
    elif sample.confidence is None:
        sample.confidence = 0.75

    return sample


def _get_send_queue(tracker_id: str) -> "queue.Queue[dict]":
    q = send_queues.get(tracker_id)
    if q is None:
        q = queue.Queue(maxsize=SEND_QUEUE_MAXSIZE)
        send_queues[tracker_id] = q
        send_counts.setdefault(tracker_id, 0)
    return q


def push_event(tracker_id: str, payload: dict) -> None:
    q = _get_send_queue(tracker_id)
    try:
        q.put_nowait(payload)
    except queue.Full:
        # Drop oldest to keep the most recent stim events.
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(payload)
        except queue.Full:
            return

    send_counts[tracker_id] = send_counts.get(tracker_id, 0) + 1
    cnt = send_counts[tracker_id]
    if cnt <= 5 or cnt % 100 == 0:
        print(f"[web-bridge] send {tracker_id}: {cnt}", file=sys.stderr, flush=True)


@app.websocket("/ws/{tracker_id}")
async def ws_endpoint(ws: WebSocket, tracker_id: str):
    print(f"[web-bridge] WS connect: {tracker_id}", file=sys.stderr, flush=True)
    recv_counts[tracker_id] = 0
    _get_send_queue(tracker_id)
    await ws.accept()

    async def sender():
        q = send_queues[tracker_id]
        try:
            while True:
                try:
                    msg = q.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue
                await ws.send_json(msg)
        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            return

    sender_task = asyncio.create_task(sender())

    try:
        while True:
            data = await ws.receive_json()
            s = WGSample(**data)
            s = _sanitize_browser_sample(s)
            if s is None:
                continue

            ad = adapters_registry.get(tracker_id)
            if ad is not None:
                from eyetrk.core.types import Sample
                ad.emit(
                    Sample(
                        session_id=s.session_id,
                        tracker_id=s.tracker_id,
                        timestamp_ms=s.timestamp_ms,
                        frame_id=s.frame_id,
                        head_x=s.head_x,
                        head_y=s.head_y,
                        head_z=s.head_z,
                        yaw=s.yaw,
                        pitch=s.pitch,
                        roll=s.roll,
                        x_norm=s.x_norm,
                        y_norm=s.y_norm,
                        confidence=s.confidence,
                        validity=s.validity,
                        stim_id=s.stim_id,
                        event=s.event,
                        task_name=s.task_name,
                        target_x_px=s.target_x_px,
                        target_y_px=s.target_y_px,
                        inner_h=s.inner_h,
                        inner_w=s.inner_w,
                        fullscreen=s.fullscreen,
                        gr_gaze_x=s.gr_gaze_x,
                        gr_gaze_y=s.gr_gaze_y,
                        gr_doc_x=s.gr_doc_x,
                        gr_doc_y=s.gr_doc_y,
                    )
                )

            recv_counts[tracker_id] += 1
            cnt = recv_counts[tracker_id]
            if cnt <= 5 or cnt % 100 == 0:
                print(f"[web-bridge] recv {tracker_id}: {cnt}", file=sys.stderr, flush=True)

    except (WebSocketDisconnect, ConnectionResetError, BrokenPipeError):
        print(f"[web-bridge] WS disconnect: {tracker_id}", file=sys.stderr, flush=True)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[web-bridge] WS error for {tracker_id}: {e}", file=sys.stderr, flush=True)
    finally:
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        # Clear pending outgoing messages to avoid replay on reconnect.
        q = send_queues.get(tracker_id)
        if q is not None:
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
