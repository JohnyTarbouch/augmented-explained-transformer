from __future__ import annotations

from pathlib import Path


def with_run_id(base_dir: str | Path, run_id: str | None) -> Path:
    base = Path(base_dir)
    return base / run_id if run_id else base
