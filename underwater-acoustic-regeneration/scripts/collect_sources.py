"""Retrieve bounded public research sources and record hashes; never run macros.

No audio archive or model checkpoint is downloaded by this preparation script.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "conference_cfp.html": "https://easychair.org/cfp/ICBDAIA27",
    "conference_home.html": "https://icbdaia.com/",
    "uniqueship.html": "https://uniqueshipdata.org/",
    "uniqueship_folder.html": "https://drive.google.com/drive/folders/1HxV8AGh8ckZS3JehAU_gY2fuCnscX-eV",
    "uniqueship_paper.pdf": "https://arxiv.org/pdf/2609.13659",
    "vliz_record.json": "https://zenodo.org/api/records/17233667",
    "diffwave.pdf": "https://arxiv.org/pdf/2009.09761",
    "repaint.pdf": "https://openaccess.thecvf.com/content/CVPR2022/papers/Lugmayr_RePaint_Inpainting_Using_Denoising_Diffusion_Probabilistic_Models_CVPR_2022_paper.pdf",
    "csdi.pdf": "https://arxiv.org/pdf/2107.03502",
    "learn_then_test.pdf": "https://people.eecs.berkeley.edu/~angelopoulos/publications/downloads/ltt.pdf",
    "inpainting_uncertainty.pdf": "https://www.isca-archive.org/interspeech_2025/cohen25_interspeech.pdf",
    "runpod_storage.html": "https://docs.runpod.io/pods/storage/types",
    "runpod_ssh.html": "https://docs.runpod.io/pods/configuration/use-ssh",
    "conference_template.docm": "https://drive.google.com/uc?export=download&id=1X67baiOAmfXYsMWcHocoHNuiSS75y24w",
    "closest_work_metadata.json": "https://api.crossref.org/works/10.1109/ICIGP68997.2026.11619767",
}


def retrieve(name: str, url: str) -> dict:
    destination = ROOT / "provenance" / "downloads" / name
    record = {"name": name, "url": url, "accessed_utc": datetime.now(timezone.utc).isoformat()}
    try:
        response = requests.get(url, timeout=(10, 30), stream=True)
        response.raise_for_status()
        chunks = []
        total = 0
        for chunk in response.iter_content(65536):
            total += len(chunk)
            if total > 16 * 1024 * 1024:
                raise ValueError("Source exceeds 16 MiB preparation cap")
            chunks.append(chunk)
        payload = b"".join(chunks)
        if name.endswith(".pdf") and not payload.startswith(b"%PDF"):
            raise ValueError("Response is not a PDF")
        if name.endswith(".docm") and not payload.startswith(b"PK"):
            raise ValueError("Response is not an Office ZIP container")
        if name.endswith(".json"):
            json.loads(payload)
        destination.write_bytes(payload)
        record.update(status="retrieved", bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                      local_path=str(destination.relative_to(ROOT)), content_type=response.headers.get("Content-Type"))
    except (requests.RequestException, ValueError) as error:
        record.update(status="unavailable", error=str(error))
    return record


def main() -> None:
    from concurrent.futures import ThreadPoolExecutor
    (ROOT / "provenance" / "downloads").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(lambda item: retrieve(*item), SOURCES.items()))
    path = ROOT / "provenance" / "source_retrieval.json"
    path.write_text(json.dumps(records, indent=2) + "\n")
    for record in records:
        print(record["name"], record["status"], record.get("bytes", record.get("error")))


if __name__ == "__main__":
    main()
