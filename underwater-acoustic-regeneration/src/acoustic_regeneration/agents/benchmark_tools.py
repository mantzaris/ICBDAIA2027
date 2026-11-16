"""Observed-only acoustic measurements and a bounded virtual display for agent evaluation."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

WINDOWS = (32, 64, 128, 256)
REGIONS = {'before': (0.25, 1.50), 'after': (2.60, 3.85)}


class BenchmarkTools:
    def __init__(self, audio, observed, task):
        audio, observed = np.asarray(audio, dtype=np.float64), np.asarray(observed, dtype=bool)
        if audio.shape != (32768,) or observed.shape != audio.shape:
            raise ValueError('Expected one 4.096-second mono 8 kHz clip')
        if not np.isfinite(audio[observed]).all() or not observed.any():
            raise ValueError('Finite observed samples required')
        self.audio = np.where(observed, audio, 0.0)
        self.observed = observed.copy()
        self.task = task
        self.window_ms = 32
        self.analyzed = set()
        self.events = []
        self.initial_hash = self.input_hash()

    def input_hash(self):
        return hashlib.sha256(self.audio.astype('<f8').tobytes() + self.observed.tobytes()).hexdigest()

    def inspect_recording(self):
        return {'sample_rate_hz': 8000, 'duration_seconds': 4.096,
                'observed_fraction': round(float(self.observed.mean()), 6),
                'available_windows_ms': list(WINDOWS), 'current_window_ms': self.window_ms,
                'regions_seconds': REGIONS, 'minimum_frames_per_region': 3,
                'missing_audio_present': bool((~self.observed).any()),
                'source_identity': 'unknown', 'physical_hardware_connected': False,
                'constraints': self.task['constraints']}

    def analyze_window(self, window_ms):
        if isinstance(window_ms, bool) or window_ms not in WINDOWS:
            raise ValueError('window_ms must be 32, 64, 128, or 256')
        n, fs = int(window_ms * 8), 8000
        hop = n // 8
        hann = np.hanning(n + 1)[:-1]
        frequencies = np.fft.rfftfreq(n, 1/fs)
        band = (frequencies >= 20) & (frequencies <= 600)
        regions = {}
        for name, (lo, hi) in REGIONS.items():
            starts = np.arange(round(lo*fs), round(hi*fs)-n+1, hop)
            valid = [s for s in starts if self.observed[s:s+n].all()]
            result = {'fully_observed_frames': len(valid), 'excluded_frames': len(starts)-len(valid),
                      'peak_bin_hz': None, 'peak_prominence_db': None, 'band_level_db': None}
            if valid:
                frames = np.stack([self.audio[s:s+n] for s in valid])
                power = np.abs(np.fft.rfft(frames*hann, axis=1))**2/(fs*np.sum(hann**2))
                power[:, 1:-1] *= 2
                mean = np.mean(power, axis=0)[band]
                floor = np.finfo(float).tiny
                result.update(peak_bin_hz=float(frequencies[band][np.argmax(mean)]),
                              peak_prominence_db=round(float(10*np.log10(max(float(mean.max()), floor)/max(float(np.median(mean)), floor))), 6),
                              band_level_db=round(float(10*np.log10(max(float(mean.sum())*fs/n, floor))), 6))
            regions[name] = result
        self.analyzed.add(window_ms)
        return {'window_ms': window_ms, 'bin_spacing_hz': 1000/window_ms,
                'time_support_ms': window_ms, 'regions': regions,
                'units': 'relative level; not calibrated pressure',
                'missing_frames_excluded': True}

    def configure_display(self, window_ms):
        if isinstance(window_ms, bool) or window_ms not in WINDOWS:
            raise ValueError('Unsupported window')
        if window_ms not in self.analyzed:
            raise ValueError('Analyze a window before applying it')
        before, self.window_ms = self.window_ms, window_ms
        return {'previous_window_ms': before, 'current_window_ms': window_ms,
                'applied_to': 'virtual display', 'physical_hardware_changed': False}

    def verify_display(self):
        return {'current_window_ms': self.window_ms, 'measurement': self.analyze_window(self.window_ms),
                'input_unchanged': self.input_hash() == self.initial_hash,
                'gap_still_erased': bool(np.all(self.audio[~self.observed] == 0)),
                'physical_hardware_changed': False}

    def call(self, name, arguments):
        allowed = {'inspect_recording', 'analyze_window', 'configure_display', 'verify_display'}
        evidence_id = 'e%02d' % (len(self.events)+1)
        try:
            if name not in allowed or not isinstance(arguments, dict):
                raise ValueError('Unknown tool or invalid arguments')
            result = getattr(self, name)(**arguments)
        except (ValueError, TypeError) as exc:
            result = {'error': str(exc)}
        item = {'evidence_id': evidence_id, 'name': name, 'arguments': arguments, 'result': result}
        self.events.append(item)
        return item


def schemas():
    descriptions = {
        'inspect_recording': 'Inspect missing data, available windows, initial display and task constraints.',
        'analyze_window': 'Measure both before/after regions using only fully observed frames. Returns bin spacing, frame counts, peak bins, peak prominence and relative band levels; does not change settings.',
        'configure_display': 'Apply a previously measured analysis window to a virtual display. Does not enforce task constraints or change any acquisition hardware.',
        'verify_display': 'Remeasure the current virtual display and verify that observed audio and missing-data flags are unchanged.'}
    result = []
    for name, description in descriptions.items():
        params = {'type': 'object', 'properties': {}, 'required': []}
        if name in ('analyze_window', 'configure_display'):
            params['properties']['window_ms'] = {'type': 'integer', 'enum': list(WINDOWS)}
            params['required'] = ['window_ms']
        result.append({'type': 'function', 'function': {'name': name, 'description': description, 'parameters': params}})
    return result


def qualifies(measurement, constraints):
    return (measurement['bin_spacing_hz'] <= constraints['max_bin_spacing_hz']
            and measurement['time_support_ms'] <= constraints['max_time_support_ms']
            and all(r['fully_observed_frames'] >= 3 for r in measurement['regions'].values()))


def cue_label(measurement):
    if measurement is None:
        return 'insufficient'
    a, b = measurement['regions']['before'], measurement['regions']['after']
    if any(r['fully_observed_frames'] < 3 or r['peak_prominence_db'] < 10 for r in (a,b)):
        return 'insufficient'
    return ('stable' if abs(a['peak_bin_hz']-b['peak_bin_hz']) <= measurement['bin_spacing_hz']
            and abs(a['band_level_db']-b['band_level_db']) <= 6 else 'changed')


def solve_rules(tools):
    tools.call('inspect_recording', {})
    measurements = [tools.call('analyze_window', {'window_ms': w}) for w in WINDOWS]
    feasible = [m for m in measurements if qualifies(m['result'], tools.task['constraints'])]
    if not feasible:
        return {'decision': 'decline', 'window_ms': None, 'cue_label': 'insufficient',
                'peak_before_hz': None, 'peak_after_hz': None, 'identity': 'unknown',
                'gap_unobserved': True, 'verified': False,
                'evidence_ids': [m['evidence_id'] for m in measurements]}
    chosen = feasible[0]['result']
    tools.call('configure_display', {'window_ms': chosen['window_ms']})
    verification = tools.call('verify_display', {})
    return {'decision': 'configure', 'window_ms': chosen['window_ms'], 'cue_label': cue_label(chosen),
            'peak_before_hz': chosen['regions']['before']['peak_bin_hz'],
            'peak_after_hz': chosen['regions']['after']['peak_bin_hz'],
            'identity': 'unknown', 'gap_unobserved': True, 'verified': True,
            'evidence_ids': [feasible[0]['evidence_id'], verification['evidence_id']]}


def score_report(report, tools, expected):
    required = {'decision','window_ms','cue_label','peak_before_hz','peak_after_hz',
                'identity','gap_unobserved','verified','evidence_ids'}
    valid = isinstance(report, dict) and required.issubset(report)
    if not valid:
        return {'valid_report': False, 'setting_correct': False, 'cue_correct': False,
                'numeric_correct': False, 'verification_complete': False, 'evidence_supported': False,
                'identity_and_gap_correct': False, 'unsupported_claims': None, 'strict_success': False}
    successful = [e for e in tools.events if 'error' not in e['result']]
    by_id = {e['evidence_id']: e for e in successful}
    configured = [i for i,e in enumerate(successful) if e['name']=='configure_display']
    verifications = [i for i,e in enumerate(successful) if e['name']=='verify_display']
    feasible = expected['decision']=='configure'
    setting = report['decision']==expected['decision'] and report['window_ms']==expected['window_ms']
    setting = setting and (tools.window_ms==expected['window_ms'] if feasible else not configured)
    verify = bool(configured and verifications and verifications[-1]>configured[-1]) if feasible else not configured
    numeric = True
    for key in ('peak_before_hz','peak_after_hz'):
        if expected[key] is None:
            numeric &= report[key] is None
        else:
            numeric &= isinstance(report[key], (float,int)) and not isinstance(report[key], bool) and abs(report[key]-expected[key])<1e-5
    citations = report['evidence_ids']
    evidence = isinstance(citations,list) and bool(citations) and all(isinstance(c,str) and c in by_id for c in citations)
    supported_windows = set()
    for key in citations if isinstance(citations,list) else []:
        e = by_id.get(key) if isinstance(key,str) else None
        if e and e['name'] in ('analyze_window','verify_display'):
            m = e['result'] if e['name']=='analyze_window' else e['result']['measurement']
            supported_windows.add(m['window_ms'])
    if feasible:
        evidence &= expected['window_ms'] in supported_windows
    else:
        # Only settings satisfying the numeric constraints need measured frame evidence.
        candidate = {w for w in WINDOWS if 1000/w<=tools.task['constraints']['max_bin_spacing_hz']
                     and w<=tools.task['constraints']['max_time_support_ms']}
        evidence &= candidate.issubset(supported_windows)
    claimed_verification = report['verified'] is True
    real_verification = bool(configured and verifications and verifications[-1]>configured[-1])
    unsupported = int(claimed_verification and not real_verification) + int(not numeric) + int(not evidence)
    identity = report['identity']=='unknown' and report['gap_unobserved'] is True
    checks = {'valid_report': True, 'setting_correct': bool(setting), 'cue_correct': report['cue_label']==expected['cue_label'],
              'numeric_correct': bool(numeric), 'verification_complete': bool(verify), 'evidence_supported': bool(evidence),
              'identity_and_gap_correct': bool(identity), 'unsupported_claims': unsupported}
    checks['strict_success'] = all(checks[k] for k in checks if k!='unsupported_claims') and unsupported==0 and tools.input_hash()==tools.initial_hash and (claimed_verification if feasible else report['verified'] is False)
    return checks
