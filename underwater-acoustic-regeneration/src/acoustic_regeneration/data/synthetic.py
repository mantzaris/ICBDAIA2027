"""Controlled signals for identifiability and shift diagnostics, never training."""
from __future__ import annotations

import numpy as np


def signal_cases(seed: int = 20261115):
    rng = np.random.default_rng(seed)
    time = np.arange(32768) / 8000
    frequencies = rng.uniform(30, 1500, int(rng.integers(1, 7)))
    phases = rng.uniform(0, 2 * np.pi, len(frequencies))
    amplitudes = rng.uniform(0.2, 1.0, len(frequencies))
    tone = sum(amplitude * np.sin(2 * np.pi * frequency * time + phase)
               for amplitude, frequency, phase in zip(amplitudes, frequencies, phases))
    tone /= max(np.std(tone), 1e-12)
    modulation_hz = rng.uniform(2, 20)
    tone *= 1 + 0.2 * np.sin(2 * np.pi * modulation_hz * time)
    beta = rng.uniform(0, 2)
    noise_spectrum = np.fft.rfft(rng.normal(size=len(time)))
    noise_spectrum /= np.maximum(np.fft.rfftfreq(len(time), 1 / 8000), 1) ** (beta / 2)
    background = np.fft.irfft(noise_spectrum, n=len(time))
    background /= max(np.std(background), 1e-12)
    snr_db = int(rng.choice([-5, 0, 10, 20]))
    background *= 10 ** (-snr_db / 20)
    stationary = tone + background
    start, end = 14384, 18384
    mask = np.ones(len(time), dtype=bool)
    mask[start:end] = False
    chirp = np.sin(2 * np.pi * (frequencies[0] * time + 0.05 * frequencies[0] * time ** 2)) + background
    onset = stationary.copy()
    onset[:start + 1000] *= 0.08
    offset = stationary.copy()
    offset[start + 2000:] *= 0.08
    event = np.zeros(len(time))
    local_time = np.arange(end - start) / 8000
    event[start:end] = 2.5 * np.hanning(end - start) * np.sin(2 * np.pi * rng.uniform(400, 1500) * local_time)
    # Shared short multipath filter; add the hidden event afterwards so its
    # support remains wholly inside the erased interval by construction.
    delays = rng.integers(8, 161, size=int(rng.integers(2, 5)))
    gains = rng.uniform(0.05, 0.35, size=len(delays))
    def echo(values):
        output = values.copy()
        for delay, gain in zip(delays, gains):
            output[delay:] += gain * values[:-delay]
        return output
    signals = {"stationary": echo(stationary), "chirp": echo(chirp), "onset": echo(onset),
               "offset": echo(offset), "broadband": echo(background)}
    signals["hidden_event_absent"] = signals["stationary"]
    signals["hidden_event_present"] = signals["stationary"] + event
    return {key: (value, mask.copy()) for key, value in signals.items()}, event


def stress_masks(length: int = 32768):
    output = {}
    for duration in (750, 1000):
        width = duration * 8
        start = (length - width) // 2
        mask = np.ones(length, dtype=bool)
        mask[start:start + width] = False
        output[str(duration) + "ms"] = mask
    packets = np.ones(length, dtype=bool)
    for start in (6400, 12800, 19200, 25600):
        packets[start:start + 400] = False
    output["four_50ms_packets"] = packets
    short_context = np.ones(length, dtype=bool)
    short_context[100:2100] = False
    output["insufficient_context"] = short_context
    return output
