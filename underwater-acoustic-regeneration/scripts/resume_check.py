"""Verify bitwise CUDA continuation on real G sources before longer runs."""
from __future__ import annotations

import copy
import gzip
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from acoustic_regeneration.checkpoint import load_checkpoint, save_checkpoint
from acoustic_regeneration.models.diffwave import ModelConfig, WaveformModel
from acoustic_regeneration.registry import file_hash
from acoustic_regeneration.restoration.diffusion import cosine_schedule
from train import BatchSampler, compute_loss, make_batch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)
    manifest = ROOT / "manifests/pilot_sources.jsonl.gz"
    with gzip.open(manifest, "rt") as stream:
        rows = [row for row in map(json.loads, stream) if row["partition"] == "G"][:16]
    audio = [np.load(ROOT / row["processed_path"], allow_pickle=False) for row in rows]
    floor = max(float(np.quantile([np.std(wave) for wave in audio], 0.01)) * 0.001, 1e-12)
    model = WaveformModel(ModelConfig(32, 12, 12)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    sampler = BatchSampler(len(rows), 17)
    schedule = cosine_schedule().cuda()
    metadata = {"git_commit": os.environ["RESEARCH_GIT_COMMIT"], "config_hash": "cuda-resume-check-v1",
                "manifest_hash": file_hash(manifest)}

    def update():
        indices = sampler.next(4)
        batch = make_batch(rows, audio, indices, sampler.rng, floor)
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model, *batch, schedule)
        loss.backward()
        optimizer.step()
        return indices, batch[0].clone(), loss.item(), copy.deepcopy(model.state_dict())

    update()
    path = ROOT / "checkpoints/resume_check/state.pt"
    save_checkpoint(path, model, optimizer, update=1, sampler_state=sampler.state(), metadata=metadata, ema=model.state_dict())
    expected = update()
    payload = load_checkpoint(path, model, optimizer, expected_metadata=metadata)
    sampler.restore(payload["sampler"])
    actual = update()
    result = {"indices_identical": expected[0] == actual[0], "batch_identical": torch.equal(expected[1], actual[1]),
              "loss_identical": expected[2] == actual[2], "parameters_identical": all(torch.equal(expected[3][k], actual[3][k]) for k in expected[3]),
              "expected_loss": expected[2], "resumed_loss": actual[2], **metadata,
              "kind": "software_verification_on_G; not restoration performance"}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/cuda_resume_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not all(result[key] for key in ("indices_identical", "batch_identical", "loss_identical", "parameters_identical")):
        raise RuntimeError("CUDA checkpoint resumption mismatch")


if __name__ == "__main__":
    main()
