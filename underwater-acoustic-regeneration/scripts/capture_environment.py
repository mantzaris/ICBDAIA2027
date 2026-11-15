"""Record resolved package versions and hardware without environment secrets."""
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = sorted({distribution.metadata["Name"] for distribution in metadata.distributions()})
    packages = {name: metadata.version(name) for name in names}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=10)
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "torch": torch.__version__, "torch_cuda": torch.version.cuda,
              "gpu": gpu.stdout.strip(), "packages": packages,
              "container_image_digest": None,
              "container_image_digest_status": "not exposed by the supplied SSH connection; not inferred"}
    directory = ROOT / "provenance"
    directory.mkdir(exist_ok=True)
    (directory / "gpu_environment.json").write_text(json.dumps(report, indent=2) + "\n")
    (ROOT / "requirements.gpu.lock").write_text("# Resolved installed environment, including inherited pod packages.\n" +
                                               "\n".join(name + "==" + version for name, version in packages.items()) + "\n")
    print("Recorded", len(packages), "resolved package versions")


if __name__ == "__main__":
    main()
