"""Reproducible CUDA mechanism controls, never simulated vessel identities."""
from __future__ import annotations

from typing import Any
import math
import hashlib
import torch
from .signal import require_cuda, bandpass

KINDS = ("tones", "chirp", "harmonics", "missing_fundamental", "crossing",
         "occlusion", "transient", "am_noise", "beating", "mixed", "noise", "silence")
SPLIT_OFFSETS = {"train": 100_000, "validation": 200_000, "val": 200_000,
                 "test": 300_000, "pilot": 400_000}


def generate_scene(seed: int, split: str = "train", kind: str | None = None,
                   sample_rate: int = 16000, duration_s: float = 30.,
                   snr_db: float = 10., return_components: bool = False):
    """Return (waveform, truth), or (waveform, truth, CUDA component waveforms).

    Exact supports and trajectories are emitted only in the separate truth object.
    A split-specific seed namespace prevents noise realization leakage. Test chirps
    use descending slopes, a parameter combination absent from training chirps.
    Repeated deterministic seeds are repeated inputs, not independent replicates.
    """
    device = require_cuda()
    if split not in SPLIT_OFFSETS:
        raise ValueError("Unknown split")
    kind = kind or KINDS[seed % len(KINDS)]
    if kind not in KINDS:
        raise ValueError(f"Unknown synthetic kind: {kind}")
    if sample_rate < 12000 or duration_s < 5:
        raise ValueError("Mechanism suite requires >=12 kHz and >=5 seconds")
    generator_seed = SPLIT_OFFSETS[split] + int(seed)
    generator = torch.Generator(device=device).manual_seed(generator_seed)
    n = round(sample_rate * duration_s)
    t = torch.arange(n, device=device, dtype=torch.float32) / sample_rate
    # All parameter sampling and waveform arithmetic execute on CUDA.
    random = torch.rand(8, generator=generator, device=device)
    base = (110 + 250 * random[0]).item()
    start, end = .1 * duration_s, .9 * duration_s
    waveforms: dict[str, torch.Tensor] = {}
    events: list[dict] = []

    def envelope(begin, finish):
        attack = ((t - begin) / .1).clamp(0, 1)
        release = ((finish - t) / .1).clamp(0, 1)
        return torch.minimum(attack, release)

    def tone(f0, slope=0., begin=start, finish=end, amplitude=.6, component_id=None,
             mechanism="oscillator"):
        cid = component_id or f"s{len(waveforms)}"
        phase = (2 * math.pi * torch.rand((), generator=generator, device=device)).item()
        local_t = t - begin
        signal = amplitude * envelope(begin, finish) * torch.cos(
            2 * math.pi * (f0 * local_t + .5 * slope * local_t.square()) + phase)
        waveforms[cid] = signal
        points_t = torch.linspace(begin, finish, 49, device=device)
        points_f = f0 + slope * (points_t - begin)
        events.append(dict(id=f"gold:{len(events)}", component_waveform_id=cid,
                           type="chirp" if slope else "tone", frequency_domain="acoustic",
                           time_s=[begin, finish], frequency_hz=[points_f.min().item(), points_f.max().item()],
                           center_frequency_hz=points_f.median().item(),
                           frequency_track=torch.stack((points_t, points_f), dim=1).tolist(),
                           slope_hz_per_s=slope, frequency_uncertainty_hz=0.,
                           mechanism=mechanism, source_identity="not_defined"))

    def am_noise(begin=start, finish=end, modulation=12.):
        noise = torch.randn(n, generator=generator, device=device)
        noise = bandpass(noise, sample_rate, (2500., min(4500., sample_rate / 2 - 200)))
        noise = noise / noise.square().mean().sqrt().clamp_min(1e-10)
        cid = f"s{len(waveforms)}"
        waveforms[cid] = .5 * envelope(begin, finish) * (1 + .8 * torch.cos(2 * math.pi * modulation * t)) * noise
        events.append(dict(id=f"gold:{len(events)}", component_waveform_id=cid,
                           type="modulation", frequency_domain="modulation", time_s=[begin, finish],
                           frequency_hz=[modulation, modulation], center_frequency_hz=modulation,
                           frequency_track=[[begin, modulation], [finish, modulation]],
                           carrier_band_hz=[2000., 5000.], mechanism="amplitude_modulated_noise",
                           source_identity="not_defined", frequency_uncertainty_hz=0.))

    def transient(begin=None, duration=.5):
        begin = .55 * duration_s if begin is None else begin
        cid = f"s{len(waveforms)}"
        burst = bandpass(torch.randn(n, generator=generator, device=device), sample_rate, (500., 1500.))
        burst = burst / burst.square().mean().sqrt().clamp_min(1e-10)
        waveforms[cid] = 1.5 * envelope(begin, begin + duration) * burst
        events.append(dict(id=f"gold:{len(events)}", component_waveform_id=cid,
                           type="transient" if duration < 1 else "broadband", frequency_domain="acoustic",
                           time_s=[begin, begin + duration], frequency_hz=[500., 1500.],
                           mechanism="band_limited_noise_burst", source_identity="not_defined"))

    if kind == "tones":
        tone(base); tone(base + 95, begin=.25 * duration_s, finish=.75 * duration_s)
    elif kind == "chirp":
        slope = (-1 if split == "test" else 1) * (2 + 2 * random[1].item())
        tone(base + 100, slope)
    elif kind in ("harmonics", "missing_fundamental"):
        for harmonic in ([1, 2, 3] if kind == "harmonics" else [2, 3, 4]):
            tone(base * harmonic, amplitude=.7 / harmonic, mechanism="harmonic_oscillator")
    elif kind == "crossing":
        delta = 80.
        tone(base, 2 * delta / (end - start))
        tone(base + 2 * delta, -2 * delta / (end - start))
    elif kind == "occlusion":
        tone(base, begin=start, finish=.43 * duration_s)
        tone(base, begin=.57 * duration_s, finish=end)
        tone(base + 55, begin=.35 * duration_s, finish=.65 * duration_s)
    elif kind == "transient":
        tone(base); transient()
    elif kind == "am_noise":
        am_noise(modulation=(9. if split == "test" else 12.) + 2 * random[2].item())
    elif kind == "beating":
        delta = 12.
        tone(1000., amplitude=.5, mechanism="interfering_oscillator")
        tone(1000. + delta, amplitude=.5, mechanism="interfering_oscillator")
        events.append(dict(id=f"gold:{len(events)}", component_waveform_id=None,
                           type="modulation", frequency_domain="modulation", time_s=[start, end],
                           frequency_hz=[delta, delta], center_frequency_hz=delta,
                           frequency_track=[[start, delta], [end, delta]],
                           carrier_band_hz=[500., 2000.], mechanism="interference_beating",
                           source_identity="not_defined", frequency_uncertainty_hz=0.))
        # abs(cos(pi*delta*t)) has a Fourier series at every n*delta. These
        # deterministic envelope harmonics are known observable structure, not
        # hallucinations and not additional physical sources. Only the fundamental
        # is a required endpoint; higher harmonics up to the plotted 200 Hz are
        # ignored known structure for precision/recall (no reward, no penalty).
        for harmonic in range(2, int(200 // delta) + 1):
            frequency = harmonic * delta
            events.append(dict(id=f"gold:{len(events)}", component_waveform_id=None,
                               type="modulation", frequency_domain="modulation", time_s=[start, end],
                               frequency_hz=[frequency, frequency], center_frequency_hz=frequency,
                               frequency_track=[[start, frequency], [end, frequency]],
                               carrier_band_hz=[500., 2000.], mechanism="interference_envelope_harmonic",
                               source_identity="not_defined", frequency_uncertainty_hz=0.,
                               evaluation_required=False, relative_amplitude_to_fundamental=3 / (4 * harmonic ** 2 - 1)))
    elif kind == "mixed":
        tone(base); tone(base + 90, 2.); transient(); am_noise(modulation=15.)
    background = torch.randn(n, generator=generator, device=device)
    if waveforms:
        clean = torch.stack(list(waveforms.values())).sum(dim=0)
        background *= clean.square().mean().sqrt() / background.square().mean().sqrt() * 10 ** (-snr_db / 20)
    else:
        clean = torch.zeros(n, device=device)
        background *= .1 if kind == "noise" else 0.
    waveform = clean + background
    scale = torch.maximum(waveform.abs().max(), torch.tensor(1., device=device)) / .95
    waveform /= scale
    waveforms = {cid: signal / scale for cid, signal in waveforms.items()}
    waveforms["background"] = background / scale
    opaque_id = hashlib.sha256(f"sonar-scene-v1:{split}:{seed}:{kind}".encode()).hexdigest()[:16]
    truth = dict(scene_id=f"scene-{opaque_id}", stratum="synthetic", split=split,
                 parent_group=f"synthetic-generator-{generator_seed}", generator_seed=generator_seed,
                 kind=kind, sample_rate_hz=sample_rate, duration_s=duration_s, snr_db=snr_db,
                 global_scale=scale.item(), events=events, device=str(device),
                 mechanism_label_is_not_vessel_identity=True,
                 annotation_origin="exact_declared_simulator_parameters",
                 generator_version="1.0", real_world_physical_validation=False)
    truth["modulation_exhaustive"] = kind in ("am_noise", "beating", "mixed", "noise", "silence")
    truth["modulation_evaluation_scope"] = (
        "Required declared modulation and known optional beat harmonics; noise/silence negative controls"
        if truth["modulation_exhaustive"] else
        "Envelope consequences of acoustic superposition are not exhaustively enumerated; exclude modulation predictions from scoring")
    if return_components:
        return waveform, truth, waveforms
    return waveform, truth


def remove_component(waveform: torch.Tensor, components: dict[str, torch.Tensor], component_id: str):
    """Keep background and other components fixed; recompute all derived views."""
    require_cuda(waveform)
    if component_id == "background" or component_id not in components:
        raise ValueError("Choose a known simulated signal component")
    return waveform - components[component_id]


def real_mixture(first: torch.Tensor, second: torch.Tensor, sample_rate: int,
                 first_parent: dict, second_parent: dict, offset_s: float = 0.,
                 ratio_db: float = 0.) -> tuple[torch.Tensor, dict]:
    """Controlled real excerpts, not clean-source or natural-overlap ground truth."""
    require_cuda(first); require_cuda(second)
    if first_parent["split"] != second_parent["split"]:
        raise ValueError("Mixture parents must share one preassigned split")
    offset = round(offset_s * sample_rate)
    if offset < 0:
        raise ValueError("Offset must be nonnegative")
    result = torch.zeros(max(first.numel(), offset + second.numel()), device=first.device)
    rms_a = first.square().mean().sqrt().clamp_min(1e-10)
    rms_b = second.square().mean().sqrt().clamp_min(1e-10)
    gain_b = rms_a / rms_b * 10 ** (-ratio_db / 20)
    result[:first.numel()] += first
    result[offset:offset + second.numel()] += gain_b * second
    scale = torch.maximum(result.abs().max() / .95, torch.tensor(1., device=result.device))
    result /= scale
    return result, dict(stratum="semi_synthetic", split=first_parent["split"],
                        parents=[first_parent, second_parent], offset_s=offset_s,
                        nominal_excerpt_ratio_db=ratio_db, gain_second=gain_b.item(),
                        global_scale=scale.item(),
                        parent_support_s=[[0, first.numel() / sample_rate],
                                          [offset_s, offset_s + second.numel() / sample_rate]],
                        annotation_origin="known_excerpt_timing_only",
                        individual_physical_source_truth_available=False)
