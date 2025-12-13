from pathlib import Path


def session_path(base: str, session_id: str) -> Path:
    p = Path(base) / session_id
    p.mkdir(parents=True, exist_ok=True)
    return p