"""Compatibility check for the published, hash-verified Moliner checkpoint."""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import typing
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from scipy.signal import resample_poly
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode

ROOT = Path(__file__).resolve().parents[1]
REVISION = "28b0e92683d58d0ab12a73e675a4da387b00caa5"


def load_published_checkpoint(path):
    # The published archive also stores OmegaConf settings. Restrict loading to
    # tensors, primitive containers, and these explicitly inspected data classes.
    allowed = [DictConfig, ListConfig, Metadata, ContainerMetadata, AnyNode,
               defaultdict, typing.Any, dict, list, int]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--git-commit", default="unrecorded")
    args = parser.parse_args()
    path = ROOT / "checkpoints/transfer/22k_8s-850000.pt"
    from acoustic_regeneration.registry import append_record, file_hash, remaining_seconds
    if file_hash(path) != "25436b182f906fbcfe19b5147937f7fb4226542230381fcecb77114bdba60d87":
        raise ValueError("Published checkpoint hash mismatch")
    payload = load_published_checkpoint(path)
    details = {"iteration": payload["it"], "keys": list(payload), "ema_tensor_elements": sum(x.numel() for x in payload["ema"].values()),
               "ema_first_keys": list(payload["ema"])[:10],
               "published_configuration": OmegaConf.to_container(payload["args"], resolve=False)}
    (ROOT / "provenance/transfer_checkpoint_inspection.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps({key: value for key, value in details.items() if key != "published_configuration"}, indent=2))
    if args.inspect_only:
        return
    start = time.monotonic()
    code = ROOT / "provenance/downloads/transfer" / ("audio-inpainting-diffusion-" + REVISION)
    sys.path.insert(0, str(code))
    from networks.unet_cqt_oct_with_projattention_adaLN_2 import Unet_CQT_oct_with_attention
    from diff_params.edm import EDM
    from testing.edm_sampler_inpainting import Sampler
    from acoustic_regeneration.data.observation import Observation, normalize_observation
    from acoustic_regeneration.data.preprocessing import crop_source
    from acoustic_regeneration.evaluation.metrics import MetricFloors, gap_metrics
    from acoustic_regeneration.restoration.classical import bidirectional_ar
    torch.set_num_threads(4)
    configuration = OmegaConf.create(details["published_configuration"])
    configuration.tester = OmegaConf.load(code / "conf/tester/inpainting_tester.yaml")
    assert configuration.exp.sample_rate == 22050 and configuration.exp.audio_len == 184184
    assert configuration.tester.T == 70 and configuration.tester.order == 2
    torch.manual_seed(20261115)
    network = Unet_CQT_oct_with_attention(configuration, torch.device("cuda")).cuda().eval().requires_grad_(False)
    network.load_state_dict(payload["ema"], strict=True)
    del payload
    sampler = Sampler(network, EDM(configuration), configuration)
    with gzip.open(ROOT / "manifests/pilot_sources.jsonl.gz", "rt") as stream:
        rows = [row for row in map(json.loads, stream) if row["partition"] == "V"][:8]
    floor = json.loads((ROOT / "checkpoints/pilot_conditional_seed17/config.json").read_text())["rms_floor"]
    authorization = json.loads((ROOT / "configs/run_authorization.json").read_text())
    run_id = "moliner_zero_shot_8V"
    output = ROOT / "results/transfer"
    output.mkdir(exist_ok=True)
    case_manifest = [{"source_id": row["original_file_id"], "group_id": row["group_id"],
                      "gap_ms": (50, 100, 250, 500)[index % 4], "sampler_seed": 20261115 + index}
                     for index, row in enumerate(rows)]
    case_path = output / "case_manifest.json"
    case_path.write_text(json.dumps(case_manifest, indent=2) + "\n")
    append_record(ROOT / "results/run_registry.jsonl", {"run_id": run_id, "status": "running",
                  "git_commit": args.git_commit, "checkpoint_sha256": file_hash(path), "source_commit": REVISION,
                  "manifest_sha256": file_hash(case_path), "stage": "V_only_zero_shot_compatibility",
                  "configuration": OmegaConf.to_container(configuration.tester, resolve=False)})
    calls = 0
    def count(module, inputs):
        nonlocal calls
        calls += len(inputs[0])
    handle = network.register_forward_pre_hook(count)
    torch.cuda.reset_peak_memory_stats()
    results = []
    status = "running"
    try:
        for index, (row, case) in enumerate(zip(rows, case_manifest)):
            if remaining_seconds(authorization["stop_jobs_by_utc"]) < 300 or time.monotonic() - start > 2400:
                status = "budget_stopped"
                break
            reference = crop_source(np.load(ROOT / row["processed_path"], allow_pickle=False))
            gap_samples = case["gap_ms"] * 8
            gap_start = (len(reference) - gap_samples) // 2
            gap_end = gap_start + gap_samples
            mask = np.ones(len(reference), dtype=bool)
            mask[gap_start:gap_end] = False
            original = Observation(reference, mask)
            observation, transform = normalize_observation(original, floor)
            # Resample ONLY already-masked values. Expand the invalid interval by
            # 12 input samples, conservatively exceeding scipy's default 10-sample
            # polyphase-filter half support. Hidden reference values cannot enter.
            upsampled = resample_poly(observation.values, 441, 160)
            positions = np.arange(len(upsampled)) * 160 / 441
            known = (positions < gap_start - 12) | (positions >= gap_end + 12)
            sigma_data = float(configuration.tester.diff_params.sigma_data)
            length = int(configuration.exp.audio_len)
            offset = (length - len(upsampled)) // 2
            values = np.zeros(length, dtype=np.float32)
            known_padded = np.zeros(length, dtype=np.float32)
            values[offset:offset + len(upsampled)] = np.where(known, upsampled * sigma_data, 0)
            known_padded[offset:offset + len(upsampled)] = known
            y = torch.tensor(values, device="cuda")[None]
            m = torch.tensor(known_padded, device="cuda")[None]
            torch.manual_seed(case["sampler_seed"])
            torch.cuda.synchronize()
            before, initial_calls = time.perf_counter(), calls
            generated = sampler.predict_inpainting(y, m)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - before
            raw = generated[0, offset:offset + len(upsampled)].cpu().numpy().astype(float) / sigma_data
            downsampled = resample_poly(raw, 160, 441)[:len(reference)]
            estimate = observation.project(downsampled)
            target = transform.normalize_reference_for_scoring(reference)
            metrics = gap_metrics(estimate, target, observation.gaps, MetricFloors(floor / transform.scale))
            baseline = bidirectional_ar(observation, 128)
            baseline_metrics = gap_metrics(baseline, target, observation.gaps, MetricFloors(floor / transform.scale))
            record = dict(case, metrics=metrics, diagnostic_ar128_metrics=baseline_metrics,
                          denoiser_forward_evaluations=calls - initial_calls,
                          backward_evaluations=calls - initial_calls, pipeline_seconds=seconds,
                          padding_samples_excluded_from_guidance=int((known_padded == 0).sum()) - int((~known).sum()),
                          before_projection_observation_drift=float(np.max(np.abs(downsampled[mask] - observation.values[mask]))))
            append_record(output / "cases.jsonl", record)
            results.append(record)
            print("transfer case", len(results), "/ 8", seconds, "seconds", flush=True)
        if status == "running":
            status = "completed"
    except Exception as error:
        status = "failed"
        append_record(output / "failures.jsonl", {"error": repr(error), "completed_cases": len(results)})
        raise
    finally:
        handle.remove()
        summary = {"status": status, "completed_cases": len(results), "planned_cases": 8,
                   "source_partition": "V", "model_parameters": sum(p.numel() for p in network.parameters()),
                   "peak_vram_bytes": torch.cuda.max_memory_allocated(), "elapsed_seconds": time.monotonic() - start,
                   "mean_Q": float(np.mean([row["metrics"]["Q"] for row in results])) if results else None,
                   "mean_gap_SC": float(np.mean([row["metrics"]["gap_spectral_convergence"] for row in results])) if results else None,
                   "adaptation": "none; observation-only RMS scaled to published sigma_data=0.063",
                   "input_rate_adapter": "zeroed 8k observation -> 22050 Hz; mask expanded 12 input samples; unknown padding excluded from guidance; inverse resampling then exact observation projection",
                   "claim_limit": "Compatibility/domain diagnostic on 8 V groups, not a reproduction score or evidence of superiority; AR128 is descriptive, not the selected B",
                   "git_commit": args.git_commit}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        append_record(ROOT / "results/run_registry.jsonl", dict(summary, run_id=run_id, output_path="results/transfer/summary.json"))
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        from acoustic_regeneration.registry import append_record
        append_record(ROOT / "results/run_registry.jsonl", {"run_id": "moliner_zero_shot_8V", "status": "failed",
                      "stage": "startup_or_execution", "error": repr(error), "command": sys.argv})
        raise
