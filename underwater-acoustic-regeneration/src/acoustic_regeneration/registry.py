"""Append-only run records and a fixed wall-clock deadline."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_record(path: Path, record: dict) -> None:
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = dict(record, recorded_utc=datetime.now(timezone.utc).isoformat())
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def remaining_seconds(deadline_utc: str) -> float:
    deadline = datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        raise ValueError("Budget deadline must include a timezone")
    return (deadline - datetime.now(timezone.utc)).total_seconds()
