"""Download the exact official split volumes, with size and SHA-256 records."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import gdown

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ("allclass_5hours.z01", "1IaDNuCOtg9cl7TfCXyEfj-uizIwCTX34", 10737418240),
    ("allclass_5hours.zip", "1gBOsHTfSKmAhXKODFK2twz5AGtQT3Od1", 7059560361),
]


def main() -> None:
    raw = ROOT / "data" / "raw" / "uniqueship_v1"
    raw.mkdir(parents=True, exist_ok=True)
    (ROOT / "manifests").mkdir(exist_ok=True)
    needed = sum(max(0, size - (raw / name).stat().st_size) if (raw / name).exists() else size
                 for name, _, size in FILES)
    if shutil.disk_usage(raw).free < needed + 5 * 1024 ** 3:
        raise RuntimeError("Insufficient space for split archives plus 5 GiB reserve")
    records = []
    for name, file_id, expected_bytes in FILES:
        path = raw / name
        start = time.monotonic()
        print(datetime.now(timezone.utc).isoformat(), "downloading", name, flush=True)
        if not path.exists() or path.stat().st_size != expected_bytes:
            outcome = gdown.download(id=file_id, output=str(path), quiet=True, use_cookies=False, resume=True)
            if outcome is None:
                raise RuntimeError("Official download failed: " + name)
        if path.stat().st_size != expected_bytes:
            raise RuntimeError("Release size mismatch: " + name)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        records.append({"name": name, "drive_id": file_id, "bytes": expected_bytes,
                        "sha256": digest.hexdigest(), "sha256_status": "locally computed; no publisher digest provided",
                        "elapsed_seconds": time.monotonic() - start})
        (ROOT / "manifests" / "uniqueship_archives.json").write_text(json.dumps(records, indent=2) + "\n")
        print("verified", name, expected_bytes, digest.hexdigest(), flush=True)


if __name__ == "__main__":
    main()
