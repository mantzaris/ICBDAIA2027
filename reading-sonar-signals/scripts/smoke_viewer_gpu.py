#!/usr/bin/env python3
"""Exercise the optional localhost viewer on an explicitly allocated CUDA slot.

The two exploratory requests write only into the selected viewer's reinspection
directory. No trained model, benchmark report, or annotation is modified.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", type=Path, default=Path("viewer_development"))
    parser.add_argument("--output", type=Path, default=Path("reports/viewer_http_cuda_smoke.json"))
    args = parser.parse_args()
    viewer = (ROOT / args.viewer).resolve()
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle = json.loads((viewer / "viewer_bundle.json").read_text())
    case = bundle["cases"][0]
    start, end = case["time_s"]
    if end - start < 8 or case["record"]["sample_rate"] < 4200:
        raise ValueError("Smoke fixture requires at least eight seconds and the low DEMON carrier")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    report = {"started_at": datetime.now(timezone.utc).isoformat(),
              "scene_id": case["scene_id"], "scope": "exploratory_only_excluded_from_benchmark",
              "bundle_sha256": hashlib.sha256((viewer / "viewer_bundle.json").read_bytes()).hexdigest(),
              "checks": [], "status": "running"}

    def request(path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        message = Request(base + path, data=data,
                          headers={"Content-Type": "application/json", "Origin": base})
        try:
            with urlopen(message, timeout=45) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            return error.code, json.loads(error.read())

    with output.with_suffix(".server.log").open("w") as log:
        process = subprocess.Popen([sys.executable, str(ROOT / "scripts/serve_viewer_gpu.py"),
                                    "--viewer", str(viewer), "--port", str(port), "--max-calls-per-case", "2"],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 45
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Local CUDA server exited during initialization; see server log")
                try:
                    status, available = request("/api/status")
                    if status == 200:
                        break
                except (URLError, ConnectionError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Local CUDA server did not become ready")
                time.sleep(.2)
            assert available["cuda"] and not available["benchmark_mutation"]
            report["service"] = available
            report["checks"].append("localhost CUDA status")
            invalid = {"name": "inspect_region", "time_s": [start - 1, start + 2],
                       "window_s": 1., "frequency_hz": [100., 600.]}
            status, response = request("/api/reinspect", {"scene_id": case["scene_id"], "action": invalid})
            assert status == 400, response
            report["checks"].append("out-of-scene request rejected")
            results = []
            for action in ({"name": "inspect_region", "time_s": [start, start + 8],
                            "window_s": 1., "frequency_hz": [100., 600.]},
                           {"name": "inspect_modulation", "time_s": [start, start + 8],
                            "window_s": 4., "carrier_band_hz": [500, 2000]}):
                status, response = request("/api/reinspect", {"scene_id": case["scene_id"], "action": action})
                assert status == 200 and response.get("numeric_device") == "cuda", response
                assert response.get("view") and response["view"]["frequency_domain"] in {"acoustic", "modulation"}, response
                for raster in response["view"]["rasters"].values():
                    path = viewer / raster["url"]
                    assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == raster["sha256"]
                results.append(response)
                report["checks"].append(action["name"] + " CUDA measurement and replay raster")
            status, response = request("/api/reinspect", {"scene_id": case["scene_id"], "action": action})
            assert status == 429, response
            report["checks"].append("bounded call allowance enforced")
            report.update(status="passed", results=results)
        except Exception as error:
            report.update(status="failed", error=str(error))
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
            report["completed_at"] = datetime.now(timezone.utc).isoformat()
            report["server_stopped"] = process.poll() is not None
            output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "checks": report["checks"],
                      "server_stopped": report["server_stopped"], "output": str(output)}))


if __name__ == "__main__":
    main()
