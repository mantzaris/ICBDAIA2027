#!/usr/bin/env python3
"""CUDA low-frequency near-duplicate screening across frozen recording splits.

Spectral similarity is a candidate generator, never proof of duplicate audio.
Even high 400-Hz waveform correlation flags provenance review rather than silently
changing a split. The bounded check cannot exclude duplicates with larger shifts,
time warps, non-overlapping excerpts, or similarities outside its retained band.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import soundfile as sf
import torch
from torch.nn import functional as F
from sonar.signal import require_cuda, resample, stft_psd


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def shifted_normalized_correlation(first: torch.Tensor, second: torch.Tensor,
                                   max_lag: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairwise rows, overlap-local means/variances, FFT cross-products on CUDA.

    Positive lag means first[lag:] matches second[:-lag]. Returns signed maximum
    absolute normalized correlation and its lag in samples for every row.
    """
    require_cuda(first); require_cuda(second)
    if first.ndim == 1:
        first, second = first[None], second[None]
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Equal [pair,sample] tensors required")
    n = first.shape[-1]
    if max_lag < 0 or max_lag >= n:
        raise ValueError("Lag must leave a nonempty overlap")
    fft_size = 1 << (2 * n - 2).bit_length()
    full = torch.fft.irfft(torch.fft.rfft(first, fft_size) * torch.fft.rfft(second, fft_size).conj(), fft_size)
    lags = torch.arange(-max_lag, max_lag + 1, device=first.device)
    cross = full[:, lags % fft_size]
    a_start, b_start = lags.clamp_min(0), (-lags).clamp_min(0)
    count = n - lags.abs()
    def sums(signal):
        cumulative = F.pad(signal.cumsum(dim=-1), (1, 0))
        squared = F.pad(signal.square().cumsum(dim=-1), (1, 0))
        return cumulative, squared
    a_sum, a_squared = sums(first)
    b_sum, b_squared = sums(second)
    sa = a_sum[:, a_start + count] - a_sum[:, a_start]
    sb = b_sum[:, b_start + count] - b_sum[:, b_start]
    qa = a_squared[:, a_start + count] - a_squared[:, a_start]
    qb = b_squared[:, b_start + count] - b_squared[:, b_start]
    numerator = cross - sa * sb / count
    variance_a = (qa - sa.square() / count).clamp_min(0)
    variance_b = (qb - sb.square() / count).clamp_min(0)
    denominator = (variance_a * variance_b).sqrt()
    correlation = torch.where(denominator > 1e-15, numerator / denominator.clamp_min(1e-15), 0.).clamp(-1, 1)
    best = correlation.abs().argmax(dim=1)
    return correlation.gather(1, best[:, None]).flatten(), lags[best]


def self_test():
    device = require_cuda()
    generator = torch.Generator(device=device).manual_seed(4821)
    x = torch.randn(12000, device=device, generator=generator)
    shifted = torch.cat((torch.zeros(37, device=device), x[:-37])) * 2.3
    correlation, lag = shifted_normalized_correlation(shifted, x, 400)
    assert correlation.item() > .99999 and lag.item() == 37, (correlation.item(), lag.item())
    independent = torch.randn(12000, device=device, generator=generator)
    unrelated, _ = shifted_normalized_correlation(x, independent, 400)
    assert unrelated.abs().item() < .1
    silence, _ = shifted_normalized_correlation(torch.zeros_like(x), torch.zeros_like(x), 400)
    assert silence.item() == 0
    return dict(status="passed", shifted_scale_invariant_correlation=correlation.item(),
                recovered_lag_samples=lag.item(), independent_absolute_correlation=unrelated.abs().item(),
                silence_correlation=silence.item(), device=str(device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/real_scenes.jsonl")
    parser.add_argument("--output", default="reports/source_similarity_audit.json")
    parser.add_argument("--spectral-candidate-threshold", type=float, default=.995)
    parser.add_argument("--aligned-candidate-threshold", type=float, default=.95)
    parser.add_argument("--waveform-review-threshold", type=float, default=.98)
    parser.add_argument("--max-shift-seconds", type=float, default=1.)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2)); return
    device = require_cuda()
    manifest_path = Path(args.manifest)
    manifest_bytes = manifest_path.read_bytes()
    rows = [json.loads(line) for line in manifest_bytes.decode().splitlines() if line.strip()]
    if not rows or any(row["stratum"] != "real" for row in rows):
        raise ValueError("This audit requires a nonempty real-scene manifest")
    start = time.perf_counter()
    signals, fingerprints = [], []
    hashes, source_records = [], []
    decoding_seconds = 0.
    cuda_begin, cuda_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    cuda_begin.record()
    with torch.inference_mode():
        for index, row in enumerate(rows):
            path = Path(row["path"])
            if not path.is_absolute():
                path = ROOT / path
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if row.get("sha256") and digest != row["sha256"]:
                raise RuntimeError(f"Source file hash changed: {row['scene_id']}")
            decode_start = time.perf_counter()
            audio, sr = sf.read(path, dtype="float32", always_2d=True)
            decoding_seconds += time.perf_counter() - decode_start
            waveform = torch.from_numpy(audio).to(device=device).mean(dim=1)
            low = resample(waveform, sr, 400.)
            signals.append(low)
            view = stft_psd(low, 400., window_s=1., hop_s=.5).crop(frequency_hz=(2., 180.))
            fingerprint = 10 * torch.log10(view.psd.mean(dim=1).clamp_min(1e-20))
            fingerprint = fingerprint - fingerprint.mean()
            fingerprints.append(fingerprint / fingerprint.norm().clamp_min(1e-12))
            hashes.append(digest)
            source_records.append({k: row.get(k) for k in
                                   ("scene_id", "group_id", "domain", "split", "source_id", "source_start_seconds")})
            if (index + 1) % 30 == 0:
                print(json.dumps(dict(phase="source_similarity_features", complete=index + 1, total=len(rows))), flush=True)
        common_samples = min(signal.numel() for signal in signals)
        aligned = torch.stack([signal[:common_samples] for signal in signals])
        aligned = aligned - aligned.mean(dim=1, keepdim=True)
        degenerate_indices = torch.where(aligned.norm(dim=1) <= 1e-12)[0].tolist()
        normalized = aligned / aligned.norm(dim=1, keepdim=True).clamp_min(1e-12)
        waveform_cos = normalized @ normalized.T
        spectral = torch.stack(fingerprints)
        spectral_cos = spectral @ spectral.T
        split_names = sorted({row["split"] for row in rows})
        split_indices = torch.tensor([split_names.index(row["split"]) for row in rows], device=device)
        cross_split = split_indices[:, None] != split_indices[None, :]
        cross_split = torch.triu(cross_split, diagonal=1)
        candidate_mask = cross_split & ((spectral_cos >= args.spectral_candidate_threshold) |
                                        (waveform_cos.abs() >= args.aligned_candidate_threshold))
        candidates = torch.nonzero(candidate_mask)
        candidate_records = []
        max_lag = round(400 * args.max_shift_seconds)
        for batch in candidates.split(64):
            if batch.numel() == 0:
                continue
            correlation, lag = shifted_normalized_correlation(aligned[batch[:, 0]], aligned[batch[:, 1]], max_lag)
            for position, (a, b) in enumerate(batch.tolist()):
                candidate_records.append(dict(
                    scene_ids=[rows[a]["scene_id"], rows[b]["scene_id"]],
                    parent_groups=[rows[a]["group_id"], rows[b]["group_id"]],
                    splits=[rows[a]["split"], rows[b]["split"]],
                    domains=[rows[a]["domain"], rows[b]["domain"]],
                    file_hash_equal=hashes[a] == hashes[b],
                    spectral_shape_cosine=spectral_cos[a, b].item(),
                    aligned_waveform_cosine=waveform_cos[a, b].item(),
                    best_shifted_waveform_correlation=correlation[position].item(),
                    best_lag_seconds=lag[position].item() / 400,
                    strong_low_frequency_waveform_match=bool(correlation[position].abs() >= args.waveform_review_threshold),
                    interpretation="Candidate for provenance review, not an automatically confirmed duplicate"))
        cuda_end.record(); cuda_end.synchronize()
    exact_cross_split = []
    for a in range(len(rows)):
        for b in range(a + 1, len(rows)):
            if rows[a]["split"] != rows[b]["split"] and hashes[a] == hashes[b]:
                exact_cross_split.append([rows[a]["scene_id"], rows[b]["scene_id"]])
    report = dict(status="complete", manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                  source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  signal_code_sha256=hashlib.sha256((ROOT / "src/sonar/signal.py").read_bytes()).hexdigest(),
                  scenes=len(rows), cross_split_pairs_screened=int(cross_split.sum().item()),
                  waveform_sample_rate_hz=400, retained_band_hz=[0., 180.],
                  common_aligned_duration_seconds=common_samples / 400,
                  insufficient_low_frequency_variance_scene_ids=[rows[i]["scene_id"] for i in degenerate_indices],
                  spectral_fingerprint="mean-centered log mean PSD, 2–180 Hz in 1 Hz bins; L2 normalized",
                  thresholds={k: getattr(args, k) for k in ("spectral_candidate_threshold", "aligned_candidate_threshold",
                                                           "waveform_review_threshold", "max_shift_seconds")},
                  exact_cross_split_file_hash_matches=exact_cross_split,
                  candidates=candidate_records,
                  provenance_review_required=any(row["strong_low_frequency_waveform_match"] for row in candidate_records),
                  automatic_split_changes=False,
                  limitations=["Spectral similarity alone does not establish duplication",
                               "High low-frequency waveform correlation still requires provenance/full-band review",
                               "Bounded lag screening cannot exclude larger offsets, time warps or non-overlapping excerpts",
                               "No independence claim beyond the audited metadata and retained audio band"],
                  wall_seconds=time.perf_counter() - start, cpu_decoding_seconds=decoding_seconds,
                  cuda_event_elapsed_seconds=cuda_begin.elapsed_time(cuda_end) / 1000,
                  cuda_timing_note="elapsed span includes host decoding/launch gaps; not occupancy-derived active time",
                  sources=source_records, numerical_device=str(device), self_test=self_test())
    save_json(args.output, report)
    print(json.dumps({k: report[k] for k in ("status", "scenes", "cross_split_pairs_screened", "provenance_review_required")}), flush=True)


if __name__ == "__main__":
    main()
