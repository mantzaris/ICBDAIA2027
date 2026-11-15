"""Bounded pilot trainer with source hashes, EMA, exact RNG state and atomic saves."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from acoustic_regeneration.checkpoint import capture_rng, load_checkpoint, restore_rng, save_checkpoint
from acoustic_regeneration.data.observation import Observation, normalize_observation
from acoustic_regeneration.data.preprocessing import crop_source, draw_mask
from acoustic_regeneration.models.diffwave import ModelConfig, WaveformModel
from acoustic_regeneration.models.losses import deterministic_loss
from acoustic_regeneration.registry import append_record, file_hash, remaining_seconds
from acoustic_regeneration.restoration.diffusion import cosine_schedule, masked_noise_loss, noisy_training_input

ROOT = Path(__file__).resolve().parents[1]


class BatchSampler:
    def __init__(self, count: int, seed: int):
        self.rng = np.random.default_rng(seed)
        self.count = count
        self.permutation = self.rng.permutation(count)
        self.position = 0

    def next(self, count: int) -> list:
        indices = []
        for _ in range(count):
            if self.position == self.count:
                self.permutation = self.rng.permutation(self.count)
                self.position = 0
            indices.append(int(self.permutation[self.position]))
            self.position += 1
        return indices

    def state(self) -> dict:
        return {"rng": self.rng.bit_generator.state, "count": self.count,
                "permutation": self.permutation, "position": self.position}

    def restore(self, state: dict) -> None:
        if state["count"] != self.count:
            raise ValueError("Sampler source count changed")
        self.rng.bit_generator.state = state["rng"]
        self.permutation = state["permutation"]
        self.position = state["position"]


def make_batch(rows: list, audio: list, indices: list, rng, floor: float, fixed: bool = False):
    references, masks, floors = [], [], []
    for index in indices:
        waveform = audio[index]
        if fixed:
            sample_seed = int(rows[index]["original_file_id"][:8], 16)
            local_rng = np.random.default_rng(sample_seed)
            start = 0
        else:
            local_rng = rng
            start = int(rng.integers(0, len(waveform) - 32768 + 1))
        target = crop_source(waveform, start)
        mask = draw_mask(len(target), local_rng, two_gap_probability=0.0 if fixed else 0.2)
        observation, transform = normalize_observation(Observation(target, mask), floor)
        references.append(transform.normalize_reference_for_scoring(target).astype(np.float32))
        masks.append(mask)
        floors.append(floor / transform.scale)
    return (torch.tensor(np.stack(references), device="cuda"),
            torch.tensor(np.stack(masks), device="cuda"), torch.tensor(floors, dtype=torch.float32, device="cuda"))


def compute_loss(model, reference, mask, floors, schedule):
    if model.config.mode == "deterministic":
        observed = torch.where(mask, reference, torch.zeros_like(reference))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(observed, torch.zeros(len(reference), dtype=torch.long, device="cuda"), observed, mask)
        return deterministic_loss(output, reference, mask, floors)
    time_steps = torch.randint(0, len(schedule), (len(reference),), device="cuda")
    noise = torch.randn_like(reference)
    if model.config.mode == "unconditional":
        alpha = schedule[time_steps].unsqueeze(-1)
        state = alpha.sqrt() * reference + (1 - alpha).sqrt() * noise
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(state, time_steps)
        return (output.float() - noise).square().mean()
    state, observed = noisy_training_input(reference, mask, time_steps, noise, schedule)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(state, time_steps, observed, mask)
    return masked_noise_loss(output, noise, mask)


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["conditional", "deterministic", "unconditional"], default="conditional")
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-every", type=int, default=2000)
    parser.add_argument("--validation-cases", type=int, default=32)
    parser.add_argument("--overfit", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size % args.microbatch or args.updates < 1:
        raise ValueError("Effective batch must be divisible by microbatch")
    authorization = json.loads((ROOT / "configs/run_authorization.json").read_text())
    deadline = authorization["stop_jobs_by_utc"]
    if remaining_seconds(deadline) < 300:
        raise RuntimeError("Insufficient authorized time to start training")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    manifest = ROOT / "manifests/pilot_sources.jsonl.gz"
    with gzip.open(manifest, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    training = [row for row in rows if row["partition"] == "G"]
    validation = [row for row in rows if row["partition"] == "V"][:args.validation_cases]
    if not training or not validation:
        raise ValueError("Independent G and V sources are required")
    if set(row["group_id"] for row in training) & set(row["group_id"] for row in validation):
        raise ValueError("Training/validation group overlap")
    if args.overfit:
        training = training[:args.overfit]
    audio = [np.load(ROOT / row["processed_path"], allow_pickle=False) for row in training]
    valid_audio = [np.load(ROOT / row["processed_path"], allow_pickle=False) for row in validation]
    rms_values = [float(np.std(waveform.astype(np.float64))) for waveform in audio]
    floor = max(float(np.quantile(rms_values, 0.01)) * 0.001, 1e-12)
    model_config = ModelConfig(args.width, args.layers, 12, 1000, args.kind)
    config = {key: value for key, value in vars(args).items() if key != "resume"}
    config.update(model=asdict(model_config), rms_floor=floor,
                  rms_floor_definition="max(1e-12, 0.001 * 1st percentile of G source centered RMS)",
                  phase="development_only_pilot", validation_metric="EMA loss on fixed V masks/noise")
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    metadata = {"git_commit": args.git_commit, "config_hash": config_hash, "manifest_hash": file_hash(manifest)}
    run_directory = ROOT / "checkpoints" / args.run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    model = WaveformModel(model_config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    ema = copy.deepcopy(model.state_dict())
    sampler = BatchSampler(len(audio), args.seed)
    schedule = cosine_schedule().cuda()
    update, best_loss, validations_without_gain = 0, float("inf"), 0
    elapsed_before_resume = 0.0
    if args.resume:
        checkpoint = load_checkpoint(run_directory / "latest.pt", model, optimizer, expected_metadata=metadata)
        state = checkpoint["sampler"]
        sampler.restore(state["batch_sampler"])
        best_loss = state["best_loss"]
        validations_without_gain = state["validations_without_gain"]
        elapsed_before_resume = state["elapsed_seconds"]
        ema, update = checkpoint["ema"], checkpoint["update"]
        ema = {key: value.to("cuda") for key, value in ema.items()}
    interrupted = []
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda signum, frame: interrupted.append(signum))
    start = time.monotonic()
    last_checkpoint = start
    last_gpu_log = 0.0
    torch.cuda.reset_peak_memory_stats()
    registry = ROOT / "results/run_registry.jsonl"
    history = ROOT / "logs" / (args.run_id + ".jsonl")
    append_record(registry, dict(run_id=args.run_id, status="running", command=sys.argv,
                                config=config, **metadata, seed=args.seed, source_groups=len({row['group_id'] for row in training}),
                                source_files=len(training), hardware=torch.cuda.get_device_name(), output_path=str(run_directory.relative_to(ROOT))))

    def save(name):
        state = {"batch_sampler": sampler.state(), "best_loss": best_loss,
                 "validations_without_gain": validations_without_gain,
                 "elapsed_seconds": elapsed_before_resume + time.monotonic() - start}
        save_checkpoint(run_directory / name, model, optimizer, update=update, sampler_state=state,
                        metadata=metadata, ema=ema)

    @torch.no_grad()
    def validate():
        rng = capture_rng()
        original = copy.deepcopy(model.state_dict())
        model.load_state_dict(ema)
        model.eval()
        losses = []
        torch.manual_seed(20261115)
        try:
            for index in range(0, len(validation), args.microbatch):
                indices = list(range(index, min(index + args.microbatch, len(validation))))
                batch = make_batch(validation, valid_audio, indices, None, floor, fixed=True)
                losses.append((float(compute_loss(model, *batch, schedule)), len(indices)))
        finally:
            model.load_state_dict(original)
            model.train()
            restore_rng(rng)
        return sum(value * count for value, count in losses) / sum(count for _, count in losses)

    status = "running"
    try:
        while update < args.updates:
            if interrupted or remaining_seconds(deadline) < 120:
                status = "interrupted" if interrupted else "budget_stopped"
                break
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(args.batch_size // args.microbatch):
                indices = sampler.next(args.microbatch)
                batch = make_batch(training, audio, indices, sampler.rng, floor, fixed=bool(args.overfit))
                loss = compute_loss(model, *batch, schedule)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                (loss / (args.batch_size // args.microbatch)).backward()
                losses.append(float(loss.detach()))
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for key, value in model.state_dict().items():
                    ema[key].mul_(0.999).add_(value, alpha=0.001)
            update += 1
            if update == 1 or update % 50 == 0:
                record = {"run_id": args.run_id, "update": update, "training_loss": float(np.mean(losses)),
                          "gradient_norm": float(gradient_norm), "elapsed_seconds": elapsed_before_resume + time.monotonic() - start,
                          "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                          "remaining_authorized_seconds": remaining_seconds(deadline)}
                append_record(history, record)
                print(json.dumps(record), flush=True)
            if time.monotonic() - last_gpu_log >= 60:
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10)
                append_record(ROOT / "logs/gpu_utilization.jsonl", {"run_id": args.run_id, "update": update, "gpu_util_memory_power": gpu.stdout.strip()})
                last_gpu_log = time.monotonic()
            if update % args.validation_every == 0 or update == args.updates:
                valid_loss = validate()
                append_record(history, {"run_id": args.run_id, "update": update, "validation_loss": valid_loss, "validation_partition": "V"})
                if valid_loss < best_loss:
                    best_loss, validations_without_gain = valid_loss, 0
                    save("best.pt")
                else:
                    validations_without_gain += 1
                if validations_without_gain >= 5:
                    status = "early_stopped"
                    break
            if update % 2000 == 0 or time.monotonic() - last_checkpoint >= 900:
                save("latest.pt")
                last_checkpoint = time.monotonic()
        if status == "running":
            status = "completed" if update == args.updates else "incomplete"
    except Exception as error:
        status = "failed"
        append_record(history, {"run_id": args.run_id, "error": repr(error), "update": update})
        raise
    finally:
        save("latest.pt")
        append_record(registry, {"run_id": args.run_id, "status": status, "updates": update, **metadata,
                                "elapsed_seconds": elapsed_before_resume + time.monotonic() - start,
                                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                                "training_forward_batches": update * args.batch_size // args.microbatch,
                                "training_backward_batches": update * args.batch_size // args.microbatch,
                                "model_hash": file_hash(run_directory / "latest.pt"),
                                "output_path": str(run_directory.relative_to(ROOT))})


if __name__ == "__main__":
    main()
