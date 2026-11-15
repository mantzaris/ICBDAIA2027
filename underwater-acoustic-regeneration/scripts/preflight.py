"""Report configuration and hardware; never launch a job or provision a resource."""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    from acoustic_regeneration.models.diffwave import ModelConfig, WaveformModel
    authorization = json.loads((ROOT / "configs" / "run_authorization.json").read_text())
    model = WaveformModel(ModelConfig())
    output = {
        "python": platform.python_version(), "project": str(ROOT),
        "available_storage_bytes": shutil.disk_usage(ROOT).free,
        "max_run_hours": authorization["maximum_run_hours"],
        "stop_pod_when_finished": authorization["stop_pod_when_finished"],
        "final_model_parameters": model.parameter_count,
        "receptive_field_samples": model.config.receptive_field_samples,
        "receptive_field_seconds_at_8k": model.config.receptive_field_samples / 8000,
    }
    if shutil.which("nvidia-smi"):
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=10)
        output["gpu"] = result.stdout.strip()
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
