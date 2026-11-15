"""Retrieve pinned public transfer model/code, with bounded size and hash checks."""
from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
REVISION = "28b0e92683d58d0ab12a73e675a4da387b00caa5"
WEIGHTS_REVISION = "9d7358c7a6b157d925ee2255d122ab5ec93771fb"
WEIGHTS_SHA256 = "25436b182f906fbcfe19b5147937f7fb4226542230381fcecb77114bdba60d87"


def retrieve(url, path, maximum_bytes, expected_hash=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        if shutil.disk_usage(ROOT).free < maximum_bytes + 2 * 1024 ** 3:
            raise RuntimeError("Preserve 2 GiB free storage")
        part = path.with_suffix(path.suffix + ".part")
        response = requests.get(url, timeout=(15, 60), stream=True)
        response.raise_for_status()
        size = 0
        with part.open("wb") as output:
            for block in response.iter_content(1024 ** 2):
                size += len(block)
                if size > maximum_bytes:
                    raise ValueError("Download exceeded its declared bound")
                output.write(block)
        part.replace(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    if expected_hash and digest.hexdigest() != expected_hash:
        raise ValueError("Published checkpoint hash mismatch")
    return {"url": url, "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(), "published_sha256_verified": expected_hash is not None}


def main():
    directory = ROOT / "provenance/downloads/transfer"
    archive = directory / "audio-inpainting-diffusion.tar.gz"
    records = [retrieve("https://codeload.github.com/eloimoliner/audio-inpainting-diffusion/tar.gz/" + REVISION,
                        archive, 16 * 1024 ** 2)]
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise ValueError("Unsafe source archive entry")
        bundle.extractall(directory, filter="data")
    weights = ROOT / "checkpoints/transfer/22k_8s-850000.pt"
    records.append(retrieve("https://huggingface.co/Eloimoliner/audio-inpainting-diffusion/resolve/" + WEIGHTS_REVISION + "/22k_8s-850000.pt",
                            weights, 712985958, WEIGHTS_SHA256))
    report = {"source_commit": REVISION, "weights_commit": WEIGHTS_REVISION, "license": "MIT code and model card",
              "network_config": "conf/network/paper_1912_unet_cqt_oct_noattention_adaln.yaml",
              "config_evidence": "pinned testing_shortgaps.sh n=54", "audio_len": 184184, "sample_rate": 22050,
              "artifacts": records, "performance_tested": False}
    (ROOT / "provenance/transfer_artifacts.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
