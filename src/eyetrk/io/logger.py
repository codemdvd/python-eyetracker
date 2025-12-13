from pathlib import Path
import pandas as pd
from .schema import SessionMeta
from ..core.types import Sample


class RunLogger:
    def __init__(self, base: str = "runs"):
        self.base = Path(base)
        self.base.mkdir(parents=True, exist_ok=True)
        self.buffers = {}
        self.session_dir: Path | None = None

    def start_session(self, meta: SessionMeta):
        self.session_dir = self.base / meta.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        (self.session_dir / "session.json").write_text(
            meta.model_dump_json(indent=2),
            encoding="utf-8"
        )

    def write_sample(self, sample: Sample):
        if self.session_dir is None:
            raise RuntimeError("Session not started. Call start_session(meta) first.")
        csvf = self.session_dir / f"samples_{sample.tracker_id}.csv"
        line = sample.model_dump()
        df = pd.DataFrame([line])
        header = not csvf.exists()
        df.to_csv(csvf, mode="a", header=header, index=False, encoding="utf-8")
