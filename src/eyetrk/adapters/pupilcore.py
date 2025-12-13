from __future__ import annotations
import threading
import time
import json
from typing import Callable, Optional

import zmq
import msgpack  # Pupil PUB uses msgpack

from ..core.tracker import Tracker
from ..core.types import Sample, TrackerInfo


class PupilCoreAdapter(Tracker):
    """
    ZMQ-based adapter for Pupil Labs Core (Pupil Capture / Service with Pupil Remote plugin).
    - REQ socket connects to tcp://<host>:<req_port> (default 50020)
      to query SUB_PORT and send control commands later (calibration etc.).
    - SUB socket subscribes to 'gaze' topics on tcp://<host>:<pub_port>.
    """

    def __init__(self):
        self.ctx: Optional[zmq.Context] = None
        self.req = None
        self.sub = None
        self._cb: Optional[Callable[[Sample], None]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._tracker_version = "unknown"
        self._session_id = "NA"

        self._host = "127.0.0.1"
        self._req_port = 50020
        self._pub_port = None  # will be discovered by SUB_PORT

    def initialize(self, config: dict) -> TrackerInfo:
        # read config
        self._host = config.get("host", "127.0.0.1")
        self._req_port = int(config.get("req_port", 50020))
        # context
        self.ctx = zmq.Context.instance()
        # REQ
        self.req = self.ctx.socket(zmq.REQ)
        self.req.setsockopt(zmq.LINGER, 0)
        self.req.connect(f"tcp://{self._host}:{self._req_port}")

        # ask for SUB_PORT (returns ASCII port as bytes)
        self.req.send_string("SUB_PORT")
        reply = self.req.recv_string()
        try:
            self._pub_port = int(reply)
        except ValueError:
            # some versions reply with "SUB_PORT <port>"
            parts = reply.strip().split()
            self._pub_port = int(parts[-1])

        # SUB
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.LINGER, 0)
        self.sub.connect(f"tcp://{self._host}:{self._pub_port}")
        # subscribe to gaze topics (gaze., gaze) – both for safety across versions
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "gaze")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "gaze.")

        # get version (best effort)
        try:
            self.req.send_string("v")
            self._tracker_version = self.req.recv_string()
        except Exception:
            pass

        return TrackerInfo(name="pupilcore", version=self._tracker_version, reported_fps=None)

    def start_stream(self, callback: Callable[[Sample], None], session_id: str | None = None) -> None:
        self._cb = callback
        if session_id is not None:
            self._session_id = session_id
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.sub:
            self.sub.close(0)
            self.sub = None
        if self.req:
            self.req.close(0)
            self.req = None
        # do not term ctx (may be shared)
        self._cb = None

    # map our event names to Pupil Remote later (calibration commands)
    def on_event(self, event: str, payload: dict | None = None) -> None:
        # NOTE: for now we just ignore; later we’ll send:
        # - 'R CALIBRATION_START'
        # - 'R CALIBRATION_POINT_START x y'
        # - 'R CALIBRATION_POINT_END'
        # - 'R CALIBRATION_STOP'
        return None

    # ------------ internals ------------

    def _poll_loop(self):
        """Receive gaze messages from Pupil PUB socket and forward as Sample."""
        while not self._stop.is_set() and self.sub is not None:
            try:
                # topic, payload (msgpack)
                topic = self.sub.recv_string(flags=0)
                payload = self.sub.recv(flags=0)
                msg = self._unpack(payload)

                # Compatible keys across versions:
                # - 'timestamp' (float seconds)
                # - 'norm_pos': [x,y] normalized to [0..1], origin top-left (Pupil world)
                # - 'confidence' or 'confidence_interval' (we use confidence if present)
                # Some topics: 'gaze', 'gaze.2d.0x', 'gaze.3d.'
                ts = msg.get("timestamp")
                norm = msg.get("norm_pos")
                conf = msg.get("confidence", msg.get("confidence_interval"))

                if ts is None or norm is None:
                    continue

                # KEEP top-left origin to match our unified format (same as browser canvas)
                x_norm = float(norm[0])
                y_norm = float(norm[1])

                s = Sample(
                    session_id=self._session_id,
                    tracker_id="pupilcore",
                    timestamp_ms=int(float(ts) * 1000.0),
                    x_norm=x_norm,
                    y_norm=y_norm,
                    confidence=float(conf) if conf is not None else None,
                    validity=0 if (conf is None or float(conf) >= 0.5) else 1,
                    stim_id=None,
                    event="stream",
                )
                if self._cb:
                    self._cb(s)

            except zmq.Again:
                time.sleep(0.001)
            except Exception:
                # swallow unexpected payloads, keep streaming
                time.sleep(0.005)

    @staticmethod
    def _unpack(payload: bytes) -> dict:
        # Pupil PUB is msgpack; try that first, then fallback to JSON if needed
        try:
            return msgpack.unpackb(payload, raw=False)
        except Exception:
            try:
                return json.loads(payload.decode("utf-8", errors="ignore"))
            except Exception:
                return {}
