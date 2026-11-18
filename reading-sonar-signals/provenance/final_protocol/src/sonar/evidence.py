"""Traceable, overlap-preserving acoustic tools and conservative consistency checks."""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any
import torch

from .signal import (SpectralView, require_cuda, stft_psd, lofar, demon,
                     spectral_peaks, band_power, harmonic_fit, propose_components,
                     broadband_events)


class EvidenceGraph:
    def __init__(self, scene_id: str):
        self.scene_id = scene_id
        self.components: list[dict] = []
        self.measurements: list[dict] = []
        self.relations: list[dict] = []

    def add_measurement(self, measurement: dict, components: list[dict] | None = None) -> dict:
        value = copy.deepcopy(measurement)
        value["id"] = f"E{len(self.measurements):04d}"
        value["scene_id"] = self.scene_id
        value["component_ids"] = []
        value["observed_components"] = []
        for observation_index, item in enumerate(components or []):
            component = copy.deepcopy(item)
            component["evidence_ids"] = [value["id"]]
            # Preserve the raw per-call observation before canonical merging.
            # Its identity is scoped by scene_id and measurement, so no-graph
            # experiments can retain duplicate observations without consulting
            # the deduplicated canonical component list or its associations.
            value["observed_components"].append(dict(
                copy.deepcopy(component), id=f"{value['id']}:O{observation_index:03d}",
                scene_id=self.scene_id))
            # Only compatible geometric duplicates are merged. Different domains,
            # morphologies and carrier bands are never suppressed as alternatives.
            duplicate = self._duplicate(component)
            if duplicate is None:
                component["id"] = f"C{len(self.components):04d}"
                component["scene_id"] = self.scene_id
                self.components.append(component)
            else:
                component = duplicate
                component["evidence_ids"].append(value["id"])
            value["component_ids"].append(component["id"])
        self.measurements.append(value)
        self._update_relations()
        return value

    def _duplicate(self, candidate: dict) -> dict | None:
        device = require_cuda()
        for old in self.components:
            if any(old.get(k) != candidate.get(k) for k in ("type", "frequency_domain", "carrier_band_hz")):
                continue
            t = torch.tensor([old["time_s"], candidate["time_s"]], device=device)
            overlap = (t[:, 1].min() - t[:, 0].max()).clamp_min(0)
            union = t[:, 1].max() - t[:, 0].min()
            if not overlap / union.clamp_min(1e-6) > .75:
                continue
            freq = torch.tensor([old["frequency_hz"], candidate["frequency_hz"]], device=device)
            tolerance = max(old.get("bin_spacing_hz", 1), candidate.get("bin_spacing_hz", 1)) * 2
            if (freq[0] - freq[1]).abs().max() <= tolerance:
                # Crossing ridges may have the same time/frequency bounding box.
                # Compare trajectories, never merge merely because boxes agree.
                if old.get("frequency_track") and candidate.get("frequency_track"):
                    a = torch.tensor(old["frequency_track"], device=device)
                    b = torch.tensor(candidate["frequency_track"], device=device)
                    start = torch.maximum(a[0, 0], b[0, 0])
                    stop = torch.minimum(a[-1, 0], b[-1, 0])
                    if stop <= start:
                        continue
                    times = start + torch.linspace(0, 1, 17, device=device) * (stop - start)
                    def interpolate(track):
                        if track.shape[0] == 1:
                            return track[0, 1].expand_as(times)
                        right = torch.searchsorted(track[:, 0].contiguous(), times).clamp(1, track.shape[0] - 1)
                        left = right - 1
                        fraction = (times - track[left, 0]) / (track[right, 0] - track[left, 0]).clamp_min(1e-8)
                        return track[left, 1] + fraction * (track[right, 1] - track[left, 1])
                    if (interpolate(a) - interpolate(b)).abs().mean() > tolerance:
                        continue
                return old
        return None

    def _update_relations(self):
        device = require_cuda()
        relations = []
        if len(self.components) < 2:
            self.relations = relations
            return
        intervals = torch.tensor([c["time_s"] for c in self.components], device=device)
        overlap = (torch.minimum(intervals[:, None, 1], intervals[None, :, 1]) -
                   torch.maximum(intervals[:, None, 0], intervals[None, :, 0])).clamp_min(0)
        durations = intervals[:, 1] - intervals[:, 0]
        ratio = overlap / torch.minimum(durations[:, None], durations[None]).clamp_min(1e-6)
        pairs = torch.nonzero(torch.triu(ratio > .1, diagonal=1))
        for left, right in pairs.tolist():
            a, b = self.components[left], self.components[right]
            relations.append(dict(type="overlaps_time", component_ids=[a["id"], b["id"]],
                                  overlap_fraction=ratio[left, right].item(),
                                  evidence_ids=sorted(set(a["evidence_ids"] + b["evidence_ids"]))))
            if a["frequency_domain"] != b["frequency_domain"]:
                relations.append(dict(type="unresolved_association", component_ids=[a["id"], b["id"]],
                                      reason="Shared time support does not identify a common physical source"))
        self.relations = relations

    def state(self) -> dict:
        return copy.deepcopy(dict(components=self.components, measurements=self.measurements,
                                  relations=self.relations))


class AcousticTools:
    """One input waveform; inference has no access to annotation or simulator truth.

    All times are original-recording coordinates. Tool failures are returned as
    explicit unavailable evidence and still belong in the controller's action log.
    """
    ACTIONS = ["inspect_region", "inspect_modulation", "measure_peaks", "track_component",
               "compare_intervals", "test_harmonic_relation", "inspect_overlap", "finalize_or_abstain"]

    def __init__(self, x: torch.Tensor, sample_rate: float, scene_id: str,
                 offset_s: float = 0., proposal_threshold_db: float = 12.):
        require_cuda(x)
        if x.ndim != 1:
            raise ValueError("Expected mono decoded waveform")
        self.x, self.sample_rate = x.float(), float(sample_rate)
        self.scene_id, self.offset_s = scene_id, float(offset_s)
        self.end_s = self.offset_s + x.numel() / sample_rate
        self.graph = EvidenceGraph(scene_id)
        self.latest_views: dict[str, SpectralView] = {}
        self.proposal_threshold_db = proposal_threshold_db
        self.action_log: list[dict] = []

    def _interval(self, interval):
        if interval is None:
            return [self.offset_s, self.end_s]
        start, end = map(float, interval)
        if not self.offset_s <= start < end <= self.end_s + 1e-5:
            raise ValueError("Interval is outside the original scene")
        return [start, min(end, self.end_s)]

    def _waveform_interval(self, interval):
        start, end = self._interval(interval)
        first = round((start - self.offset_s) * self.sample_rate)
        last = round((end - self.offset_s) * self.sample_rate)
        return self.x[first:last], start

    def _find_component(self, component_id: str) -> dict:
        """Resolve either a canonical node or an unmerged per-call observation.

        Raw observation lookup does not follow any canonical association. This
        keeps the flat-memory ablation's tool access equivalent while preserving
        its duplicates and unresolved identity relationships.
        """
        for component in self.graph.components:
            if component["id"] == component_id:
                return component
        for measurement in self.graph.measurements:
            for observation in measurement.get("observed_components", []):
                if observation["id"] == component_id:
                    return observation
        raise ValueError("Unknown component or observation ID")

    def _overlap_relations(self, components: list[dict]) -> list[dict]:
        """Measure selected intervals directly, without consulting graph memory."""
        intervals = torch.tensor([c["time_s"] for c in components], device=self.x.device)
        duration = intervals[:, 1] - intervals[:, 0]
        overlap = (torch.minimum(intervals[:, None, 1], intervals[None, :, 1]) -
                   torch.maximum(intervals[:, None, 0], intervals[None, :, 0])).clamp_min(0)
        fraction = overlap / torch.minimum(duration[:, None], duration[None, :]).clamp_min(1e-8)
        pairs = torch.nonzero(torch.triu(fraction > .1, diagonal=1))
        relations = []
        for left, right in pairs.tolist():
            a, b = components[left], components[right]
            relations.append(dict(type="overlaps_time", component_ids=[a["id"], b["id"]],
                                  overlap_fraction=fraction[left, right].item(),
                                  evidence_ids=sorted(set(a["evidence_ids"] + b["evidence_ids"])),
                                  identity_interpretation="Overlapping observations can describe the same event; independence is not established"))
            if a["frequency_domain"] != b["frequency_domain"]:
                relations.append(dict(type="unresolved_association", component_ids=[a["id"], b["id"]],
                                      reason="Shared time support does not identify a common physical source"))
        return relations

    def _record_view(self, name: str, view: SpectralView, proposals: bool = True) -> dict:
        components = propose_components(view, min_contrast_db=self.proposal_threshold_db) if proposals else []
        if proposals and view.metadata["frequency_domain"] == "acoustic":
            components.extend(broadband_events(view))
        measurement = dict(tool=name, scene_id=self.scene_id,
                           frame_center_time_s=[view.times_s[0].item(), view.times_s[-1].item()],
                           frequency_hz=[view.frequencies_hz[0].item(), view.frequencies_hz[-1].item()],
                           frequency_domain=view.metadata["frequency_domain"],
                           carrier_band_hz=view.metadata.get("carrier_band_hz"),
                           configuration=copy.deepcopy(view.metadata),
                           peaks=spectral_peaks(view), band_power=band_power(view),
                           interpretation="measurement_only; source_identity_unresolved")
        result = self.graph.add_measurement(measurement, components)
        self.latest_views[result["id"]] = view
        return copy.deepcopy(result)

    def overview(self, include_demon: bool = True) -> dict[str, SpectralView]:
        views = {}
        duration = self.end_s - self.offset_s
        window = min(1., duration / 2)
        views["lofar"] = lofar(self.x, self.sample_rate, window_s=window, offset_s=self.offset_s)
        self._record_view("global_lofar", views["lofar"])
        if include_demon and self.sample_rate / 2 - 100 >= 2000 and duration >= 2.5:
            mod_window = min(4., duration - 1)
            views["demon"] = demon(self.x, self.sample_rate, (500., 2000.),
                                     window_s=mod_window, offset_s=self.offset_s)
            self._record_view("global_demon", views["demon"])
        return views

    def report_state(self) -> dict:
        return dict(scene_id=self.scene_id, time_s=[self.offset_s, self.end_s],
                    sample_rate_hz=self.sample_rate, **self.graph.state(),
                    available_actions=self.ACTIONS,
                    limitations=["No calibrated pressure or source-level estimate",
                                 "Acoustic Hz and envelope-modulation Hz are different quantities",
                                 "Measurement consistency is not independently assessed correctness",
                                 "Contrast scores are not calibrated probabilities"],
                    actions_attempted=len(self.action_log))

    def execute(self, action: dict) -> dict:
        name = action.get("name", action.get("action", ""))
        try:
            result = self._execute(name, action)
            result = dict(status="ok", **result)
        except (ValueError, KeyError, TypeError) as exc:
            result = dict(status="unavailable", reason=str(exc), tool=name)
        self.action_log.append(dict(action=copy.deepcopy(action), result=copy.deepcopy(result)))
        return result

    def _execute(self, name: str, action: dict) -> dict:
        if name not in self.ACTIONS:
            raise ValueError("Unknown action")
        interval = self._interval(action.get("time_s"))
        if name in ("inspect_region", "measure_peaks", "track_component"):
            frequency = action.get("frequency_hz", [0., min(1800., .45 * self.sample_rate)])
            if name == "track_component":
                cid = action.get("component_id")
                component = self._find_component(cid)
                if component["frequency_domain"] != "acoustic":
                    raise ValueError("Use inspect_modulation to track an envelope component")
                frequency = [max(0., component["frequency_hz"][0] - 20), component["frequency_hz"][1] + 20]
            audio, origin = self._waveform_interval(interval)
            window = float(action.get("window_s", 1.))
            view = lofar(audio, self.sample_rate, window_s=window, offset_s=origin,
                         frequency_hz=tuple(frequency))
            return self._record_view(name, view)
        if name == "inspect_modulation":
            carrier = action.get("carrier_band_hz", [500., 2000.])
            view = demon(self.x, self.sample_rate, tuple(carrier),
                         window_s=float(action.get("window_s", 4.)), offset_s=self.offset_s,
                         interval_s=tuple(interval), modulation_hz=tuple(action.get("modulation_hz", [.5, 200.])))
            return self._record_view(name, view)
        if name == "test_harmonic_relation":
            ids = action.get("component_ids", [])
            if ids:
                if len(ids) != len(set(ids)):
                    raise ValueError("Harmonic inputs require distinct observation IDs")
                selected = [self._find_component(component_id) for component_id in ids]
                if any(c["frequency_domain"] != "acoustic" for c in selected):
                    raise ValueError("Harmonic fit cannot combine acoustic and modulation frequencies")
                frequencies = [c.get("center_frequency_hz", sum(c["frequency_hz"]) / 2) for c in selected]
            else:
                audio, origin = self._waveform_interval(interval)
                view = lofar(audio, self.sample_rate, offset_s=origin)
                frequencies = [p["frequency_hz"] for p in spectral_peaks(view)]
            measured = harmonic_fit(frequencies, float(action.get("tolerance_hz", 2.)))
            return self.graph.add_measurement(dict(tool=name, time_s=interval, measured=measured,
                                                   input_component_ids=ids))
        if name == "compare_intervals":
            intervals = action.get("intervals_s")
            if not isinstance(intervals, list) or len(intervals) != 2:
                raise ValueError("compare_intervals requires exactly two intervals_s")
            frequency = action.get("frequency_hz", [0., 1800.])
            powers = []
            for chosen in intervals:
                audio, origin = self._waveform_interval(chosen)
                view = lofar(audio, self.sample_rate, window_s=float(action.get("window_s", 1.)), offset_s=origin)
                powers.append(band_power(view, frequency_hz=frequency))
            values = torch.tensor([p["mean_square"] for p in powers], device=self.x.device)
            delta = (10 * torch.log10(values[1].clamp_min(1e-20) / values[0].clamp_min(1e-20))).item()
            return self.graph.add_measurement(dict(tool=name, intervals_s=intervals, frequency_hz=frequency,
                                                   band_powers=powers, received_level_change_db=delta,
                                                   interpretation="received_band_power; not emitted source power"))
        if name == "inspect_overlap":
            ids = action.get("component_ids", [])
            if len(ids) != len(set(ids)):
                raise ValueError("Overlap inputs require distinct observation IDs")
            selected = [self._find_component(component_id) for component_id in ids] if ids else self.graph.components
            if len(selected) < 2:
                raise ValueError("At least two distinct supported components are required")
            valid = {c["id"] for c in selected}
            relations = self._overlap_relations(selected)
            return self.graph.add_measurement(dict(tool=name, input_component_ids=sorted(valid),
                                                   relations=relations, source_association="unresolved",
                                                   graph_memory_used=False))
        return dict(tool=name, ready_for_report=True,
                    unresolved="Physical source assignments remain hypotheses without independent evidence")


def verify_claims(claims: list[dict], state: dict) -> dict:
    """Internal schema/numerical checks only; never a supported-claim truth score.

    Claim schema: {'type':'frequency'|'overlap'|'modulation'|'observation',
      'evidence_ids':[...], 'value_hz':optional, 'frequency_domain':optional}.
    Interpretations such as blade count, vessel identity, range or SPL are withheld.
    """
    require_cuda()
    evidence = {v["id"]: v for v in state["measurements"]}
    results = []
    forbidden = {"blade_count", "shaft_speed", "source_identity", "range", "absolute_spl", "source_level"}
    for claim in claims:
        errors = []
        ids = claim.get("evidence_ids", [])
        if not isinstance(ids, list) or not ids or any(i not in evidence for i in ids):
            errors.append("missing_or_invalid_evidence_id")
        if claim.get("type") in forbidden:
            errors.append("physical_source_claim_exceeds_measurements")
        if "value_hz" in claim and not errors:
            peaks = [p for eid in ids for p in evidence[eid].get("peaks", [])]
            domain = claim.get("frequency_domain")
            peaks = [p for p in peaks if p["frequency_domain"] == domain]
            if not peaks:
                errors.append("no_peak_in_claimed_frequency_domain")
            else:
                observed = torch.tensor([p["frequency_hz"] for p in peaks], device="cuda")
                tolerance = torch.tensor([2 * p["bin_spacing_hz"] for p in peaks], device="cuda")
                if not ((observed - float(claim["value_hz"])).abs() <= tolerance).any():
                    errors.append("frequency_disagrees_with_cited_measurements")
        if claim.get("type") == "modulation" and not claim.get("carrier_band_hz"):
            errors.append("modulation_claim_requires_carrier_band")
        results.append(dict(claim=claim, internally_consistent=not errors, errors=errors))
    return dict(check="internal_consistency_not_independent_correctness", claims=results,
                accepted=[r["claim"] for r in results if r["internally_consistent"]],
                withheld=[r["claim"] for r in results if not r["internally_consistent"]])
