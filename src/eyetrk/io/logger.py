import csv
import json
import threading
from pathlib import Path

from .schema import SessionMeta
from ..core.types import SAMPLE_CSV_FIELDS, Sample, sample_to_csv_row


class RunLogger:
    def __init__(self, base: str = "runs"):
        self.base = Path(base)
        self.base.mkdir(parents=True, exist_ok=True)
        self.session_dir: Path | None = None
        self._lock = threading.Lock()
        self._files: dict[str, object] = {}
        self._writers: dict[str, csv.DictWriter] = {}
        self._timeline_handle: object | None = None

    def start_session(self, meta: SessionMeta):
        self.close()
        self.session_dir = self.base / meta.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        (self.session_dir / "session.json").write_text(
            meta.model_dump_json(indent=2),
            encoding="utf-8"
        )

    def write_sample(self, sample: Sample):
        if self.session_dir is None:
            raise RuntimeError("Session not started. Call start_session(meta) first.")
        row = sample_to_csv_row(sample)
        tracker_id = sample.tracker_id

        with self._lock:
            writer = self._writers.get(tracker_id)
            handle = self._files.get(tracker_id)
            if writer is None or handle is None:
                csv_path = self.session_dir / f"samples_{tracker_id}.csv"
                f = open(csv_path, "a", newline="", encoding="utf-8")
                writer = csv.DictWriter(f, fieldnames=SAMPLE_CSV_FIELDS, extrasaction="ignore")
                if f.tell() == 0:
                    writer.writeheader()
                self._files[tracker_id] = f
                self._writers[tracker_id] = writer
                handle = f

            writer.writerow(row)
            handle.flush()

    def write_event(self, event_type: str, payload: dict) -> None:
        if self.session_dir is None:
            raise RuntimeError("Session not started. Call start_session(meta) first.")
        record = {"event": str(event_type), **dict(payload)}

        with self._lock:
            handle = self._timeline_handle
            if handle is None:
                timeline_path = self.session_dir / "timeline.jsonl"
                handle = open(timeline_path, "a", encoding="utf-8")
                self._timeline_handle = handle
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    def close(self) -> None:
        with self._lock:
            files = list(self._files.values())
            timeline_handle = self._timeline_handle
            self._files.clear()
            self._writers.clear()
            self._timeline_handle = None
            self.session_dir = None

        for handle in files:
            try:
                handle.close()
            except Exception:
                pass
        if timeline_handle is not None:
            try:
                timeline_handle.close()
            except Exception:
                pass
