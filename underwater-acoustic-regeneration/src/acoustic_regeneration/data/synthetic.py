"""Controlled signals for identifiability and shift diagnostics, never training."""
from __future__ import annotations

import numpy as np


def signal_cases(seed: int = 20261115):
    rng = np.random.default_rng(seed)
    time = np.arange(32768) / 8000
    background = 0.08 * rng.normal(size=len(time))
    stationary = (np.sin(2 * np.pi * 90 * time + 0.3) +
                  0.4 * np.sin(2 * np.pi * 180 * time + 1.1) + background)
    start, end = 14384, 18384
    mask = np.ones(len(time), dtype=bool)
    mask[start:end] = False
    chirp = np.sin(2 * np.pi * (80 * time + 30 * time ** 2)) + background
    onset = stationary.copy()
    onset[:start + 1000] *= 0.08
    offset = stationary.copy()
    offset[start + 2000:] *= 0.08
    event = np.zeros(len(time))
    local_time = np.arange(end - start) / 8000
    event[start:end] = 2.5 * np.hanning(end - start) * np.sin(2 * np.pi * 700 * local_time)
    return {"stationary": (stationary, mask), "chirp": (chirp, mask), "onset": (onset, mask),
            "offset": (offset, mask), "broadband": (background, mask),
            "hidden_event_absent": (stationary, mask), "hidden_event_present": (stationary + event, mask)}, event


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
