"""CPU analysis tools with an observed-only input boundary and virtual display state."""
from __future__ import annotations

import hashlib

import numpy as np


class AcousticTools:
    WINDOWS = (32, 128)

    def __init__(self, waveform, observed, sample_rate=8000):
        waveform = np.asarray(waveform, dtype=np.float64)
        observed = np.asarray(observed, dtype=bool)
        if waveform.ndim != 1 or waveform.shape != observed.shape or sample_rate != 8000:
            raise ValueError('Expected matching mono arrays at 8000 Hz')
        if not np.isfinite(waveform[observed]).all() or not observed.any():
            raise ValueError('Finite observed audio is required')
        # Erase unknown coordinates before any computation, even if a caller passes references.
        self.audio = np.where(observed, waveform, 0.0)
        self.observed = observed.copy()
        self.sample_rate = sample_rate
        self.window_ms = 32
        self.analyzed = set()
        self.initial_sha256 = self.audio_hash()

    def audio_hash(self):
        return hashlib.sha256(self.audio.astype('<f8').tobytes()).hexdigest()

    def inspect_recording(self):
        changes = np.diff(np.r_[False, ~self.observed, False].astype(int))
        gaps = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
        return {'clip': 'clip_001', 'sample_rate_hz': self.sample_rate,
                'duration_seconds': len(self.audio) / self.sample_rate,
                'missing_intervals_seconds': [[a / self.sample_rate, b / self.sample_rate] for a, b in gaps],
                'observed_samples': int(self.observed.sum()), 'current_window_ms': self.window_ms,
                'allowed_window_ms': list(self.WINDOWS), 'hop_rule': 'window / 8',
                'source_identity_evidence_available': False, 'physical_hardware_connected': False,
                'input_policy': 'Missing samples erased; frames touching missing samples excluded'}

    def spectrum_arrays(self, window_ms):
        if isinstance(window_ms, bool) or window_ms not in self.WINDOWS:
            raise ValueError('Allowed windows are 32 or 128 ms')
        window = int(window_ms * self.sample_rate / 1000)
        hop = window // 8
        starts = np.arange(0, len(self.audio) - window + 1, hop)
        known = np.array([self.observed[start:start + window].all() for start in starts])
        if not known.any():
            raise ValueError('No fully observed analysis frame')
        frames = np.stack([self.audio[start:start + window] for start in starts])
        hann = np.hanning(window + 1)[:-1]
        power = np.abs(np.fft.rfft(frames * hann, axis=1)) ** 2 / (self.sample_rate * np.sum(hann ** 2))
        power[:, 1:-1] *= 2
        power[~known] = np.nan
        frequencies = np.fft.rfftfreq(window, 1 / self.sample_rate)
        return (starts + window / 2) / self.sample_rate, frequencies, power.T, known

    def analyze_spectrum(self, window_ms):
        times, frequencies, power, known = self.spectrum_arrays(window_ms)
        band = (frequencies >= 20) & (frequencies <= 600)
        mean_power = np.mean(power[:, known], axis=1)
        peak = float(frequencies[band][np.argmax(mean_power[band])])
        self.analyzed.add(window_ms)
        return {'window_ms': window_ms, 'hop_ms': window_ms / 8,
                'bin_spacing_hz': 1000 / window_ms,
                'analysis_band_hz': [20, 600], 'observed_peak_bin_hz': peak,
                'fully_observed_frames': int(known.sum()), 'excluded_gap_frames': int((~known).sum()),
                'time_support_ms': window_ms, 'signal_units': 'relative; not calibrated pressure',
                'source_identity_evidence_available': False,
                'caveat': 'Peak bin describes observed sound; it is not an object label or proof of continuity inside the gap'}

    def configure_display(self, window_ms):
        if window_ms not in self.WINDOWS or isinstance(window_ms, bool):
            raise ValueError('Allowed windows are 32 or 128 ms')
        if window_ms not in self.analyzed:
            raise ValueError('Analyze the requested setting before configuring the display')
        before = self.window_ms
        self.window_ms = window_ms
        return {'previous_window_ms': before, 'current_window_ms': self.window_ms,
                'hop_ms': self.window_ms / 8, 'applied_to': 'virtual analysis display only',
                'physical_hardware_changed': False, 'audio_samples_changed': self.audio_hash() != self.initial_sha256}

    def verify_display(self):
        return {'current_window_ms': self.window_ms,
                'measurement': self.analyze_spectrum(self.window_ms),
                'missing_samples_remain_erased': bool(np.all(self.audio[~self.observed] == 0)),
                'audio_unchanged': self.audio_hash() == self.initial_sha256,
                'physical_hardware_changed': False}

    def call(self, name, arguments):
        functions = {name: getattr(self, name) for name in
                     ('inspect_recording', 'analyze_spectrum', 'configure_display', 'verify_display')}
        if name not in functions:
            raise ValueError('Unknown analysis tool')
        if not isinstance(arguments, dict):
            raise ValueError('Tool arguments must be an object')
        return functions[name](**arguments)


def tool_schemas():
    tools = []
    descriptions = {
        'inspect_recording': 'Inspect clip duration, missing intervals, current display settings, and available evidence.',
        'analyze_spectrum': 'Measure the dominant observed frequency bin between 20 and 600 Hz at a chosen window. Does not change display settings.',
        'configure_display': 'Apply a previously analyzed window to the virtual display. No acquisition hardware is connected.',
        'verify_display': 'Remeasure the configured display and verify the audio and missing-sample mask are unchanged.'}
    for name, description in descriptions.items():
        window = name in ('analyze_spectrum', 'configure_display')
        parameters = {'type': 'object', 'properties': {}, 'required': []}
        if window:
            parameters['properties']['window_ms'] = {'type': 'integer', 'description': 'Analysis window length in milliseconds; allowed values are 32 or 128'}
            parameters['required'] = ['window_ms']
        tools.append({'type': 'function', 'function': {'name': name, 'description': description, 'parameters': parameters}})
    return tools
