"""G-only numerical audit of near-terminal epsilon prediction and DDIM states."""
from __future__ import annotations

import argparse
import gzip
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from acoustic_regeneration.data.observation import Observation, normalize_observation
from acoustic_regeneration.data.preprocessing import crop_source, draw_mask
from acoustic_regeneration.evaluation.metrics import MetricFloors, gap_metrics
from acoustic_regeneration.registry import file_hash
from acoustic_regeneration.restoration.diffusion import cosine_schedule, noisy_training_input
from acoustic_regeneration.restoration.learned import load_ema_model

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", default="reports/sampler_numerics.json")
    parser.add_argument("--sampler-config", type=Path)
    args = parser.parse_args()
    sampler_config = json.loads(args.sampler_config.read_text()) if args.sampler_config else {}
    clean_clip = sampler_config.get("clean_prediction_clip_abs")
    initial_timestep = sampler_config.get("initial_timestep", 999)
    torch.set_num_threads(2)
    model, config, _ = load_ema_model(args.checkpoint)
    with gzip.open(ROOT / "manifests/pilot_sources.jsonl.gz", "rt") as stream:
        rows = [row for row in map(json.loads, stream) if row["partition"] == "G"][:4]
    schedule = cosine_schedule().cuda()
    results = []
    for row in rows:
        reference = crop_source(np.load(ROOT / row["processed_path"], allow_pickle=False))
        mask = draw_mask(len(reference), np.random.default_rng(20261115))
        observation, transform = normalize_observation(Observation(reference, mask), config["rms_floor"])
        target = transform.normalize_reference_for_scoring(reference)
        x = torch.tensor(target, dtype=torch.float32, device="cuda")[None]
        m = torch.tensor(mask, device="cuda")[None]
        values = torch.tensor(observation.values.copy(), dtype=torch.float32, device="cuda")[None]
        noise = torch.randn(x.shape, generator=torch.Generator(device="cuda").manual_seed(17), device="cuda")
        for precision in ("fp32", "bf16"):
            context = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
            local = {"source_id": row["original_file_id"], "partition": "G", "precision": precision, "teacher_forced": []}
            for time in (999, 980, 900, 750, 500, 250, 50, 0):
                t = torch.tensor([time], device="cuda")
                state, observed = noisy_training_input(x, m, t, noise, schedule)
                with context():
                    predicted = model(state, t, observed, m)
                clean = (state - (1 - schedule[time]).sqrt() * predicted.float()) / schedule[time].sqrt()
                exact = (state - (1 - schedule[time]).sqrt() * noise) / schedule[time].sqrt()
                local["teacher_forced"].append({"time": time, "alpha_bar": float(schedule[time]),
                    "epsilon_mse": float((predicted.float()[~m] - noise[~m]).square().mean()),
                    "x0_mse": float((clean[~m] - x[~m]).square().mean()),
                    "exact_epsilon_x0_mse": float((exact[~m] - x[~m]).square().mean())})
            state = torch.where(m, values, noise)
            times = np.linspace(initial_timestep, 0, 32, dtype=int).tolist()
            trajectory = []
            for index, time in enumerate(times):
                with context():
                    prediction = model(state, torch.tensor([time], device="cuda"), values, m)
                clean = (state - (1 - schedule[time]).sqrt() * prediction.float()) / schedule[time].sqrt()
                if clean_clip is not None:
                    clean = clean.clamp(-clean_clip, clean_clip)
                    prediction = (state - schedule[time].sqrt() * clean) / (1 - schedule[time]).sqrt()
                following = times[index + 1] if index + 1 < len(times) else -1
                state = clean if following < 0 else schedule[following].sqrt() * clean + (1 - schedule[following]).sqrt() * prediction
                state = torch.where(m, values, state)
                trajectory.append({"time": time, "predicted_x0_rms": float(clean[~m].square().mean().sqrt()),
                                   "next_state_gap_rms": float(state[~m].square().mean().sqrt())})
            output = observation.project(state[0].cpu().numpy())
            local.update(trajectory=trajectory, metrics=gap_metrics(output, target, observation.gaps, MetricFloors(config["rms_floor"] / transform.scale)))
            results.append(local)
    report = {"checkpoint_sha256": file_hash(args.checkpoint), "source_partition": "G", "cases": results,
              "clean_prediction_clip_abs": clean_clip,
              "initial_timestep": initial_timestep,
              "purpose": "Implementation/numerical diagnosis only; no V, C, or test outcomes used"}
    (ROOT / args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps([{key: row[key] for key in ("source_id", "precision", "metrics")} for row in results], indent=2))


if __name__ == "__main__":
    main()
