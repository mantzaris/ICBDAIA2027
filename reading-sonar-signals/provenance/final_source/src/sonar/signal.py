"""CUDA acoustic measurements; tensors never fall back to a CPU implementation.

The DSP is deliberately independent of display normalization and language models.
Waveforms are mono digital amplitudes: no absolute pressure calibration is implied.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import torch
import torch.nn.functional as F


def require_cuda(value: torch.Tensor | None = None) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU experimental fallback is disabled")
    if value is not None and value.device.type != "cuda":
        raise RuntimeError("Experimental input must already be on CUDA")
    return value.device if value is not None else torch.device("cuda:0")


@dataclass
class SpectralView:
    psd: torch.Tensor                 # [frequency, time], digital-amplitude^2 / Hz
    frequencies_hz: torch.Tensor
    times_s: torch.Tensor             # absolute original-recording frame centers
    metadata: dict[str, Any]

    def __post_init__(self):
        for value in (self.psd, self.frequencies_hz, self.times_s):
            require_cuda(value)

    def log_power(self) -> torch.Tensor:
        return 10 * torch.log10(self.psd.clamp_min(1e-20))

    def crop(self, time_s=None, frequency_hz=None) -> "SpectralView":
        fi = torch.ones_like(self.frequencies_hz, dtype=torch.bool)
        ti = torch.ones_like(self.times_s, dtype=torch.bool)
        if time_s is not None:
            ti &= (self.times_s >= time_s[0]) & (self.times_s <= time_s[1])
        if frequency_hz is not None:
            fi &= (self.frequencies_hz >= frequency_hz[0]) & (self.frequencies_hz <= frequency_hz[1])
        if not fi.any() or not ti.any():
            raise ValueError("Requested region has no supported spectral bins")
        return SpectralView(self.psd[fi][:, ti], self.frequencies_hz[fi], self.times_s[ti],
                            dict(self.metadata, crop_time_s=time_s, crop_frequency_hz=frequency_hz))


def stft_psd(x: torch.Tensor, sample_rate: float, window_s: float = 1.,
             hop_s: float | None = None, offset_s: float = 0.,
             domain: str = "acoustic", **metadata) -> SpectralView:
    """Hann, center=False, one-sided density; no implicit zero-padding.

    Frame centers use (N-1)/2, the exact arithmetic mean sample coordinate.
    Summing density*df equals the window-weighted mean-square amplitude.
    """
    require_cuda(x)
    if x.ndim != 1 or not torch.isfinite(x).all():
        raise ValueError("Expected a finite mono waveform")
    n = round(sample_rate * window_s)
    hop = round(sample_rate * (hop_s if hop_s is not None else window_s / 4))
    if n < 8 or hop < 1 or x.numel() < n:
        raise ValueError("Insufficient duration for requested analysis window")
    x = x.float()
    window = torch.hann_window(n, periodic=True, device=x.device, dtype=x.dtype)
    spectrum = torch.stft(x, n_fft=n, hop_length=hop, win_length=n,
                          window=window, center=False, return_complex=True)
    power = spectrum.abs().square() / (sample_rate * window.square().sum())
    weights = torch.full((n // 2 + 1,), 2., device=x.device)
    weights[0] = 1.
    if n % 2 == 0:
        weights[-1] = 1.
    power *= weights[:, None]
    frequencies = torch.fft.rfftfreq(n, d=1 / sample_rate, device=x.device)
    times = offset_s + (torch.arange(power.shape[1], device=x.device) * hop + (n - 1) / 2) / sample_rate
    meta = dict(sample_rate_hz=sample_rate, window_samples=n, hop_samples=hop,
                window_s=n / sample_rate, hop_s=hop / sample_rate,
                bin_spacing_hz=sample_rate / n, window="periodic_Hann", center=False,
                frequency_domain=domain, psd_units="digital_amplitude_squared_per_Hz",
                time_origin_s=offset_s, source_duration_s=x.numel() / sample_rate,
                device=str(x.device), **metadata)
    return SpectralView(power, frequencies, times, meta)


def _fir_lowpass(cutoff_hz: float, sample_rate: float, taps: int,
                 device: torch.device) -> torch.Tensor:
    if not 0 < cutoff_hz < sample_rate / 2:
        raise ValueError("Lowpass cutoff must be inside Nyquist")
    taps = int(taps) | 1
    n = torch.arange(taps, device=device) - (taps - 1) / 2
    kernel = 2 * cutoff_hz / sample_rate * torch.sinc(2 * cutoff_hz / sample_rate * n)
    kernel *= torch.hann_window(taps, periodic=False, device=device)
    return kernel / kernel.sum()


def fir_filter(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Zero-phase odd FIR with reflect extension, FFT linear convolution.

    Reflect extension is an explicit boundary assumption, not new observed audio.
    DEMON excludes its filter guard region from reported frame centers/support.
    """
    require_cuda(x); require_cuda(kernel)
    half = kernel.numel() // 2
    if x.numel() <= half:
        raise ValueError("Insufficient samples for FIR reflection padding")
    padded = F.pad(x[None, None], (half, half), mode="reflect").flatten()
    size = padded.numel() + kernel.numel() - 1
    fft_size = 1 << (size - 1).bit_length()
    conv = torch.fft.irfft(torch.fft.rfft(padded, fft_size) * torch.fft.rfft(kernel, fft_size), fft_size)
    # Valid convolution discards both delay and reflected samples.
    return conv[kernel.numel() - 1:kernel.numel() - 1 + x.numel()]


def bandpass(x: torch.Tensor, sample_rate: float, band_hz: tuple[float, float],
             taps: int = 513) -> torch.Tensor:
    require_cuda(x)
    lo, hi = map(float, band_hz)
    if lo <= 0 or hi >= sample_rate / 2 or hi <= lo:
        raise ValueError("Carrier band must lie strictly inside waveform Nyquist")
    kernel = _fir_lowpass(hi, sample_rate, taps, x.device) - _fir_lowpass(lo, sample_rate, taps, x.device)
    return fir_filter(x, kernel)


def resample(x: torch.Tensor, sample_rate: float, target_rate: float) -> torch.Tensor:
    """GPU FIR antialiasing followed by GPU sample-coordinate interpolation.

    This is a declared FIR/linear interpolator, not ideal Fourier resampling.
    Integer downsampling uses exact original sample locations after filtering.
    """
    require_cuda(x)
    if sample_rate == target_rate:
        return x
    if target_rate <= 0:
        raise ValueError("Target sample rate must be positive")
    signal = x
    if target_rate < sample_rate:
        taps = max(129, int(32 * sample_rate / target_rate) | 1)
        signal = fir_filter(x, _fir_lowpass(.45 * target_rate, sample_rate, taps, x.device))
    length = math.floor(x.numel() * target_rate / sample_rate)
    positions = torch.arange(length, device=x.device, dtype=torch.float64) * sample_rate / target_rate
    left = positions.floor().long().clamp(max=x.numel() - 1)
    right = (left + 1).clamp(max=x.numel() - 1)
    fraction = (positions - left).float()
    return signal[left] * (1 - fraction) + signal[right] * fraction


def analytic_envelope(x: torch.Tensor) -> torch.Tensor:
    require_cuda(x)
    n = x.numel()
    h = torch.zeros(n, device=x.device)
    h[0] = 1
    h[1:(n + 1) // 2] = 2
    if n % 2 == 0:
        h[n // 2] = 1
    return torch.fft.ifft(torch.fft.fft(x) * h).abs()


def lofar(x: torch.Tensor, sample_rate: float, window_s: float = 1.,
          hop_s: float | None = None, offset_s: float = 0.,
          frequency_hz: tuple[float, float] = (0., 1800.)) -> SpectralView:
    target = min(sample_rate, 4000.)
    low = resample(x, sample_rate, target)
    view = stft_psd(low, target, window_s, hop_s, offset_s,
                    source_sample_rate_hz=sample_rate, resampler="Hann_sinc_FIR_linear")
    # 1.8 kHz upper limit respects the antialias filter transition at 4 kHz.
    return view.crop(frequency_hz=(frequency_hz[0], min(frequency_hz[1], .45 * target)))


def demon(x: torch.Tensor, sample_rate: float, carrier_band: tuple[float, float],
          window_s: float = 4., hop_s: float = .5, offset_s: float = 0.,
          envelope_rate: float = 500., interval_s: tuple[float, float] | None = None,
          modulation_hz: tuple[float, float] = (.5, 200.)) -> SpectralView:
    """Compute from the full padded waveform, then select valid time support.

    A band requires at least a 100 Hz margin below Nyquist. At 16 kHz the
    requested 5--8 kHz band is therefore unavailable, rather than zero-filled.
    """
    require_cuda(x)
    lo, hi = map(float, carrier_band)
    if lo < 100 or hi > sample_rate / 2 - 100 or hi - lo < 200:
        raise ValueError("Unavailable carrier bandwidth or filter transition margin")
    taps = 513
    filtered = bandpass(x, sample_rate, (lo, hi), taps)
    # Pad the analytic transform as well; discard its artificial endpoints.
    guard_samples = max(taps, round(.25 * sample_rate))
    if x.numel() <= 2 * guard_samples:
        raise ValueError("Insufficient duration for DEMON filter guard")
    padded = F.pad(filtered[None, None], (guard_samples, guard_samples), mode="reflect").flatten()
    envelope = analytic_envelope(padded)[guard_samples:-guard_samples]
    envelope = resample(envelope, sample_rate, envelope_rate)
    guard = math.ceil(guard_samples / sample_rate * envelope_rate)
    envelope = envelope[guard:-guard]
    envelope = envelope - envelope.mean()
    start = offset_s + guard / envelope_rate
    if interval_s is not None:
        lower = max(0, math.ceil((interval_s[0] - start) * envelope_rate))
        upper = min(envelope.numel(), math.floor((interval_s[1] - start) * envelope_rate))
        if upper <= lower:
            raise ValueError("Requested interval has no valid envelope support")
        envelope = envelope[lower:upper]
        start += lower / envelope_rate
    view = stft_psd(envelope, envelope_rate, window_s, hop_s, start,
                    domain="modulation", carrier_band_hz=[lo, hi],
                    source_sample_rate_hz=sample_rate, carrier_fir_taps=taps,
                    envelope_resampler="Hann_sinc_FIR_linear", edge_guard_s=guard / envelope_rate,
                    calibrated=False, mechanism="periodic_envelope_not_source_identity")
    return view.crop(frequency_hz=modulation_hz)


def spectral_peaks(view: SpectralView, max_peaks: int = 12,
                   min_contrast_db: float = 8., min_spacing_hz: float = 2.) -> list[dict]:
    power = view.psd.mean(dim=1)
    df = view.metadata["bin_spacing_hz"]
    width = max(3, 2 * math.ceil(min_spacing_hz / df) + 1)
    maxima = F.max_pool1d(power[None, None], width, stride=1, padding=width // 2).flatten()
    background = power.median().clamp_min(1e-20)
    contrast = 10 * torch.log10(power.clamp_min(1e-20) / background)
    mask = (power >= maxima) & (contrast >= min_contrast_db) & (power > 1e-18)
    scores = torch.where(mask, power, torch.full_like(power, -1))
    indices = torch.topk(scores, min(max_peaks, power.numel())).indices
    indices = indices[scores[indices] > 0]
    values = torch.stack((view.frequencies_hz[indices], contrast[indices], power[indices]), dim=1).tolist()
    return [dict(frequency_hz=f, contrast_db=c, psd=p, bin_spacing_hz=df,
                 uncertainty_hz=df, frequency_domain=view.metadata["frequency_domain"],
                 carrier_band_hz=view.metadata.get("carrier_band_hz")) for f, c, p in values]


def band_power(view: SpectralView, frequency_hz=None, time_s=None) -> dict:
    chosen = view.crop(time_s=time_s, frequency_hz=frequency_hz) if (time_s or frequency_hz) else view
    power = chosen.psd.sum(dim=0).mean() * view.metadata["bin_spacing_hz"]
    return dict(mean_square=power.item(), level_db=(10 * power.clamp_min(1e-20).log10()).item(),
                frequency_hz=[chosen.frequencies_hz[0].item(), chosen.frequencies_hz[-1].item()],
                frame_center_time_s=[chosen.times_s[0].item(), chosen.times_s[-1].item()],
                units="digital_amplitude_squared", calibrated=False)


def harmonic_fit(frequencies_hz: list[float], tolerance_hz: float = 2.,
                  max_harmonic: int = 12) -> dict:
    device = require_cuda()
    if len(frequencies_hz) < 3:
        return dict(supported=False, reason="At least three measured peaks are required", alternatives=[])
    freqs = torch.tensor(frequencies_hz, device=device)
    candidates = (freqs[:, None] / torch.arange(1, max_harmonic + 1, device=device)).flatten()
    indices = torch.round(freqs[None] / candidates[:, None]).clamp(1, max_harmonic)
    residuals = (freqs[None] - indices * candidates[:, None]).abs()
    compatible = residuals <= tolerance_hz
    counts = compatible.sum(dim=1)
    # Duplicate harmonic indices do not count as independent support.
    unique = (indices[:, :, None] == indices[:, None, :]).sum(dim=-1).eq(1)
    valid = (compatible & unique).sum(dim=1) >= 3
    score = residuals.mean(dim=1) + (~valid) * 1e6
    order = score.argsort()[:5]
    alternatives = []
    for idx in order.tolist():
        if not valid[idx].item():
            continue
        alternatives.append(dict(fundamental_hz=candidates[idx].item(),
                                 harmonic_indices=indices[idx].int().tolist(),
                                 residuals_hz=residuals[idx].tolist(),
                                 normalized_residual=(residuals[idx] / tolerance_hz).tolist()))
    return dict(supported=bool(alternatives), alternatives=alternatives,
                interpretation="compatible_harmonic_spacing_not_unique_source_identity",
                tolerance_hz=tolerance_hz)


def propose_components(view: SpectralView, min_contrast_db: float = 12.,
                        max_tracks: int = 24, min_duration_s: float = 1.) -> list[dict]:
    """Measurement proposals, not independently validated truth or probabilities.

    Link local spectral maxima using one-to-one nearest predicted frequencies.
    Numerical linking and fitted trajectories remain on CUDA. Adjacent crossing
    tracks can remain unresolved; the output never infers source identity.
    """
    power = view.psd
    if power.shape[1] < 2 or not power.max() > 1e-18:
        return []
    df = float(view.metadata["bin_spacing_hz"])
    hop = float(view.metadata["hop_s"])
    window = float(view.metadata["window_s"])
    smooth = F.avg_pool2d(power[None, None], (1, 3), stride=1, padding=(0, 1))[0, 0]
    global_background = smooth.median(dim=0, keepdim=True).values
    # Nearby broadband energy is not tonal evidence. Exclude the central five
    # bins from a 41-bin local estimate so a Hann main lobe does not suppress
    # its own contrast, while a noise burst cannot create dozens of "tones".
    outer = F.avg_pool2d(smooth[None, None], (41, 1), stride=1, padding=(20, 0))[0, 0] * 41
    inner = F.avg_pool2d(smooth[None, None], (5, 1), stride=1, padding=(2, 0))[0, 0] * 5
    local_background = (outer - inner).clamp_min(0) / 36
    background = torch.maximum(global_background, local_background).clamp_min(1e-20)
    contrast = 10 * torch.log10(smooth.clamp_min(1e-20) / background)
    local_max = F.max_pool2d(smooth[None, None], (5, 1), stride=1, padding=(2, 0))[0, 0]
    valid = (smooth == local_max) & (contrast >= min_contrast_db) & (smooth > 1e-18)
    valid &= view.frequencies_hz[:, None] > 1
    k = min(16, power.shape[0])
    scores, peaks = torch.topk(torch.where(valid, smooth, -torch.ones_like(smooth)), k, dim=0)
    peak_freq = view.frequencies_hz[peaks]
    # More slots than displayed tracks allows short-lived noise candidates to die.
    slots = max(128, max_tracks * 8)
    trajectories = torch.full((slots, power.shape[1]), float("nan"), device=power.device)
    strengths = torch.full_like(trajectories, float("nan"))
    last = torch.full((slots,), -1000, device=power.device, dtype=torch.long)
    last_freq = torch.zeros(slots, device=power.device)
    velocity = torch.zeros(slots, device=power.device)
    used = 0
    max_gap = max(1, round(.75 / hop))
    for frame in range(power.shape[1]):
        active = (frame - last <= max_gap) & (last >= 0)
        predicted = last_freq + velocity * (frame - last) * hop
        distances = (predicted[:, None] - peak_freq[:, frame][None]).abs()
        distances[~active] = float("inf")
        distances[:, scores[:, frame] <= 0] = float("inf")
        assigned = torch.zeros(k, device=power.device, dtype=torch.bool)
        for _ in range(k):
            best = distances.argmin()
            track, peak = torch.div(best, k, rounding_mode="floor"), best % k
            if not distances[track, peak] <= max(3 * df, 10. * hop):
                break
            dt = (frame - last[track]) * hop
            velocity[track] = .5 * velocity[track] + .5 * (peak_freq[peak, frame] - last_freq[track]) / dt
            trajectories[track, frame] = peak_freq[peak, frame]
            strengths[track, frame] = contrast[peaks[peak, frame], frame]
            last_freq[track] = peak_freq[peak, frame]
            last[track] = frame
            assigned[peak] = True
            distances[track, :] = float("inf")
            distances[:, peak] = float("inf")
        for peak in range(k):
            if used >= slots:
                break
            if not assigned[peak] and scores[peak, frame] > 0:
                trajectories[used, frame] = peak_freq[peak, frame]
                strengths[used, frame] = contrast[peaks[peak, frame], frame]
                last_freq[used] = peak_freq[peak, frame]
                last[used] = frame
                used += 1
    observed = torch.isfinite(trajectories)
    counts = observed.sum(dim=1)
    keep = torch.where(counts * hop >= min_duration_s)[0]
    if keep.numel() == 0:
        return []
    quality = torch.nanmean(strengths[keep], dim=1) * counts[keep].sqrt()
    keep = keep[quality.argsort(descending=True)[:max_tracks]]
    events = []
    domain = view.metadata["frequency_domain"]
    for index in keep.tolist():
        mask = observed[index]
        t, f = view.times_s[mask], trajectories[index, mask]
        tc = t - t.mean()
        slope = ((f - f.mean()) * tc).sum() / tc.square().sum().clamp_min(1e-12)
        residual = (f - (f.mean() + slope * tc)).square().mean().sqrt()
        is_chirp = slope.abs() * (t[-1] - t[0]) > 3 * df
        kind = "modulation" if domain == "modulation" else ("chirp" if is_chirp else "tone")
        start = torch.maximum(t[0] - window / 2,
                              torch.as_tensor(view.metadata["time_origin_s"], device=t.device))
        stop = torch.minimum(t[-1] + window / 2, torch.as_tensor(
            view.metadata["time_origin_s"] + view.metadata["source_duration_s"], device=t.device))
        events.append(dict(type=kind, frequency_domain=domain, time_s=[start.item(), stop.item()],
                           frequency_hz=[f.min().item(), f.max().item()],
                           frequency_track=torch.stack((t, f), dim=1).tolist(),
                           center_frequency_hz=f.median().item(), slope_hz_per_s=slope.item(),
                           fit_residual_hz=residual.item(), contrast_db=torch.nanmean(strengths[index]).item(),
                           bin_spacing_hz=df, boundary_uncertainty_s=window / 2,
                           frequency_uncertainty_hz=max(df, residual.item()),
                           carrier_band_hz=view.metadata.get("carrier_band_hz"),
                           confidence_kind="uncalibrated_measurement_contrast", source_identity="unknown"))
    return events


def broadband_events(view: SpectralView, min_increase_db: float = 6.) -> list[dict]:
    """Detect broadband received-energy increases, without assigning a source."""
    if view.metadata["frequency_domain"] != "acoustic" or view.psd.shape[1] < 3:
        return []
    # Suppress narrow lines with the frequency median before temporal comparison.
    energy = view.psd.median(dim=0).values
    baseline = energy.median().clamp_min(1e-20)
    active = (10 * torch.log10(energy.clamp_min(1e-20) / baseline) > min_increase_db) & (energy > 1e-18)
    changes = torch.diff(F.pad(active.int(), (1, 1)))
    starts = torch.where(changes == 1)[0]
    ends = torch.where(changes == -1)[0] - 1
    result = []
    for first, last in zip(starts.tolist(), ends.tolist()):
        window = view.metadata["window_s"]
        interval = [max(view.metadata["time_origin_s"], view.times_s[first].item() - window / 2),
                    min(view.metadata["time_origin_s"] + view.metadata["source_duration_s"],
                        view.times_s[last].item() + window / 2)]
        result.append(dict(type="transient" if interval[1] - interval[0] < 1 else "broadband",
                           frequency_domain="acoustic", time_s=interval,
                           frequency_hz=[view.frequencies_hz[0].item(), view.frequencies_hz[-1].item()],
                           bin_spacing_hz=view.metadata["bin_spacing_hz"], boundary_uncertainty_s=window / 2,
                           contrast_db=(10 * torch.log10(energy[first:last + 1].mean() / baseline)).item(),
                           source_identity="unknown"))
    return result
