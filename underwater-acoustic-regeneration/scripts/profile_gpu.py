"""Measured CUDA microbenchmark using synthetic tensors, not research results."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from acoustic_regeneration.models.diffwave import ModelConfig, WaveformModel
from acoustic_regeneration.restoration.diffusion import cosine_schedule, masked_noise_loss, noisy_training_input

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    torch.set_num_threads(4)
    torch.manual_seed(17)
    results = []
    for label, config in [("smoke", ModelConfig(32, 12, 12)), ("final", ModelConfig())]:
        for batch in (1, 2, 4):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = WaveformModel(config).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
            reference = torch.randn(batch, 32768, device="cuda")
            mask = torch.ones_like(reference, dtype=torch.bool)
            mask[:, 14000:16000] = False
            schedule = cosine_schedule().cuda()
            durations = []
            try:
                for iteration in range(7):
                    noise = torch.randn_like(reference)
                    steps = torch.randint(0, 1000, (batch,), device="cuda")
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    state, observed = noisy_training_input(reference, mask, steps, noise, schedule)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        predicted = model(state, steps, observed, mask)
                    loss = masked_noise_loss(predicted, noise, mask)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    torch.cuda.synchronize()
                    if iteration >= 2:
                        durations.append(time.perf_counter() - start)
                record = {"model": label, "batch": batch, "parameters": model.parameter_count,
                          "median_forward_backward_step_seconds": sorted(durations)[len(durations) // 2],
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                          "finite_loss": bool(torch.isfinite(loss)), "last_loss": loss.item()}
            except torch.cuda.OutOfMemoryError:
                record = {"model": label, "batch": batch, "status": "out_of_memory"}
            results.append(record)
            print(json.dumps(record), flush=True)
            del model, optimizer, reference, mask
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "gpu_microbenchmark.json").write_text(json.dumps({
        "kind": "software_microbenchmark_on_synthetic_random_tensors",
        "research_evidence": False, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "runs": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
