#!/usr/bin/env python3
"""Build nine recorded-output figures and a local offline interactive acoustic case viewer.

No model generation or annotation is performed. Signal transforms and numerical
aggregation execute on CUDA; Matplotlib, HTML construction and serialization are
host work. Missing experiments do not produce placeholder or invented figures.

After AI references are frozen, before main inference:
  python scripts/make_figures.py --freeze-cases-only
After evaluation:
  python scripts/make_figures.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

from sonar.signal import require_cuda, lofar, demon
from sonar.data import load_waveform
from sonar.metrics import aggregate_event_metrics
from evaluate_benchmark import (read_json, read_jsonl, digest, write_json, report_events,
                                event_in_reference_scope, score_scope)

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
METHOD_COLORS = {"F": "#0072B2", "G4": "#D55E00", "G3": "#009E73", "G2": "#CC79A7", "G1": "#6B7280"}
CASE_WIDTH_IN = 6.4
CASE_FONT_PT = 10


class UnavailableFigure(RuntimeError):
    pass


def ablation_display_scope(real_methods, synthetic_methods, amendment, user_cap=None):
    """Validate declared method inventories; this does not calculate results."""
    order = lambda name: (name != "F", name != "G4", name)
    real_methods, synthetic_methods = sorted(real_methods, key=order), sorted(synthetic_methods, key=order)
    if user_cap is None:
        if set(real_methods) != set(synthetic_methods) or set(real_methods) != set(amendment["methods"]):
            raise UnavailableFigure("Uncapped ablations require the full declared identical method inventories")
        return {"user_cap_active": False, "real_methods": real_methods,
                "synthetic_methods": synthetic_methods, "real_scene_count": None,
                "synthetic_scene_count": amendment["scenes"]}
    if user_cap.get("schema_version") != 1 or user_cap.get("status") != "user_authorized_runtime_cap":
        raise UnavailableFigure("Different ablation inventories require an explicitly authorized runtime-cap config")
    capped, reuse = user_cap.get("capped", {}), user_cap.get("synthetic_reuse", {})
    if (set(real_methods) != set(capped.get("composed_methods", [])) or
            set(synthetic_methods) != set(reuse.get("methods", [])) or
            set(synthetic_methods) != {"F", "G4", "no_verifier"} or
            set(real_methods) != set(capped.get("methods", [])) | set(synthetic_methods) or
            set(reuse.get("omitted_methods", [])) != set(capped.get("methods", [])) or
            set(real_methods) != set(amendment["methods"])):
        raise UnavailableFigure("Ablation inventories disagree with the explicit user-cap/reuse scope")
    if (reuse.get("new_inference_episodes") != 0 or
            reuse.get("scene_count") != amendment["scenes"] or
            not isinstance(capped.get("scene_count"), int) or capped["scene_count"] <= 0 or
            capped.get("episodes") != user_cap.get("planned_new_real_ablation_episodes") or
            capped["episodes"] > user_cap.get("max_new_ablation_episodes", -1)):
        raise UnavailableFigure("Invalid declared runtime cap or zero-inference synthetic scope")
    return {"user_cap_active": True, "real_methods": real_methods,
            "synthetic_methods": synthetic_methods, "real_scene_count": capped["scene_count"],
            "synthetic_scene_count": reuse["scene_count"]}


def tensor(values, dtype=torch.float64):
    return torch.tensor(values, device="cuda", dtype=dtype)


def relative(path, root):
    return os.path.relpath(Path(path).resolve(), Path(root).resolve())


def get_events(reference):
    if not reference:
        return []
    annotation = reference.get("annotation", reference)
    return [event for event in annotation.get("events", []) if event_in_reference_scope(event, annotation)]


def component_callouts(events, prefix="M"):
    """Display saved coordinates/IDs in report order; do not rank by outcome."""
    rows = []
    for index, event in enumerate(events):
        rows.append({"callout": f"{prefix}{index + 1}", "component_id": event.get("id", "unnamed"),
                     "type": event.get("type"), "frequency_domain": event.get("frequency_domain", "acoustic"),
                     "time_s": event.get("time_s"), "frequency_hz": event.get("frequency_hz"),
                     "carrier_band_hz": event.get("carrier_band_hz"),
                     "evidence_ids": event.get("evidence_ids", [])})
    return rows


def component_description(row, coordinates=False):
    evidence = ",".join(str(value) for value in row["evidence_ids"]) or "none cited"
    result = f"{row['callout']} = {row['component_id']} ({row['type']}), evidence {evidence}"
    if coordinates:
        def span(value, unit):
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                return "unavailable"
            return f"{value[0]:g}–{value[1]:g} {unit}"
        result += f"; {span(row['time_s'], 's')}; {span(row['frequency_hz'], 'Hz')}"
        if row.get("carrier_band_hz"):
            result += f"; carrier {span(row['carrier_band_hz'], 'Hz')}"
    return result


def has_overlap(events):
    if len(events) < 2:
        return False
    times = tensor([event["time_s"] for event in events])
    duration = times[:, 1] - times[:, 0]
    overlap = (torch.minimum(times[:, None, 1], times[None, :, 1]) -
               torch.maximum(times[:, None, 0], times[None, :, 0])).clamp_min(0)
    ratio = overlap / torch.minimum(duration[:, None], duration[None, :]).clamp_min(1e-8)
    return bool(torch.triu(ratio > .1, diagonal=1).any())


def frozen_case_policy(records, references, manifest_path, references_path):
    real = sorted([r for r in records if r["stratum"] == "real"], key=lambda r: r["scene_id"])
    tonal = [r["scene_id"] for r in real if any(e["type"] in ("tone", "chirp") for e in get_events(references.get(r["scene_id"])))]
    overlapping = [r["scene_id"] for r in real if has_overlap(get_events(references.get(r["scene_id"])))]
    return {
        "version": "case-selection-v2-eligible-AI-reference-scope", "manifest_sha256": digest(manifest_path),
        "references_sha256": digest(references_path),
        "tonal_candidate_order": tonal, "overlap_candidate_order": overlapping,
        "all_real_candidate_order": [r["scene_id"] for r in real],
        "diversity_rule": "Prefer an unused original recording; then prefer an unrepresented domain; then lowest opaque scene ID. Never use performance for cases2/3/4.",
        "case2": "First qualifying tonal/chirp AI-reference scene under diversity ordering; if none, show first real scene explicitly as no qualifying tonal reference.",
        "case3": "First AI-reference temporal-overlap scene under diversity ordering. Co-occurring components do not establish multiple physical sources. If none, disclose that no qualifying overlap reference exists.",
        "case4": "First remaining real scene under diversity ordering with at least two saved F action images. If none, show the first recorded F action trace and disclose the missing successful-zoom criterion.",
        "case5": "First remaining real F schema failure or abstention under diversity ordering. If none, select the largest AI-reference disagreement, tie by opaque scene ID, and label this outcome-based selection explicitly.",
        "figure1_trace": "Use the case4 recorded F trace, including failed/invalid calls.",
        "reference_meaning": "Calibration-qualified morphology/domain AI references; selected checkpoint per type. Shared InternVL/Qwen ancestry allows correlated errors; no independent physical truth. Unqualified types do not supply negative labels.",
    }


class Builder:
    def __init__(self, args):
        self.args = args
        self.root = args.project_root.resolve()
        os.chdir(self.root)
        self.output = (self.root / args.output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.inputs = self.output / "inputs"; self.inputs.mkdir(exist_ok=True)
        self.viewer = (self.root / args.viewer).resolve(); self.viewer.mkdir(parents=True, exist_ok=True)
        self.main = (self.root / args.main_run).resolve()
        self.references_path = (self.root / args.references).resolve()
        self.references = {r["scene_id"]: r for r in read_jsonl(self.references_path)}
        main_config = read_json(self.main / "run.json") if (self.main / "run.json").exists() else None
        self.manifest_path = (self.root / args.manifest).resolve()
        if main_config:
            supplied = (self.root / main_config["manifest"]).resolve()
            if digest(supplied) != main_config["manifest_sha256"]:
                raise ValueError("Main run manifest hash mismatch")
            self.manifest_path = supplied
        self.records = read_jsonl(self.manifest_path)
        self.by_id = {r["scene_id"]: r for r in self.records}
        self.summary_cache = {}
        self.rows_cache = {}
        self.case_cache = {}
        self.paper_display_scales = {}
        self.artifacts = []
        self.unavailable = []
        self.sources = {}
        self.case_policy = frozen_case_policy(self.records, self.references, self.manifest_path, self.references_path)
        self.policy_path = self.output / "case_selection_policy.json"
        if self.policy_path.exists() and read_json(self.policy_path) != self.case_policy:
            raise RuntimeError("Frozen case policy/reference manifest changed; use a new figure output directory and record an amendment")
        if not self.policy_path.exists():
            write_json(self.output / "case_selection_freeze.json", {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "explicit_freeze_command": args.freeze_cases_only,
                "main_run_configuration_already_exists": (self.main / "run.json").exists(),
                "main_episode_files_already_exist": any(self.main.glob("*/*/episode.json")),
                "code_sha256": digest(Path(__file__)),
                "policy_sha256": hashlib.sha256(json.dumps(self.case_policy, sort_keys=True).encode()).hexdigest(),
            })
        write_json(self.policy_path, self.case_policy)
        self.use_source(self.manifest_path); self.use_source(self.references_path)
        self.use_source(self.policy_path); self.use_source(self.output / "case_selection_freeze.json")
        self.use_source(self.main / "run.json")
        self.selections = None
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": CASE_FONT_PT,
                             "axes.titlesize": CASE_FONT_PT, "axes.labelsize": CASE_FONT_PT,
                             "legend.fontsize": CASE_FONT_PT, "xtick.labelsize": CASE_FONT_PT, "ytick.labelsize": CASE_FONT_PT,
                             "svg.fonttype": "none", "pdf.fonttype": 42,
                             "axes.spines.top": False, "axes.spines.right": False})

    def use_source(self, path):
        path = Path(path)
        if path.exists():
            self.sources[relative(path, self.root)] = digest(path)

    def phase(self, name):
        return self.main if name == "main" else (self.root / "results" / name).resolve()

    def summary(self, name):
        path = self.phase(name) / "evaluation/summary.json"
        if not path.exists():
            raise UnavailableFigure(f"Missing evaluated phase: {name}")
        self.use_source(path)
        if name not in self.summary_cache:
            self.summary_cache[name] = read_json(path)
        return self.summary_cache[name]

    def rows(self, name):
        path = self.phase(name) / "evaluation/per_episode.jsonl"
        if not path.exists():
            raise UnavailableFigure(f"Missing per-episode evaluation: {name}")
        self.use_source(path)
        if name not in self.rows_cache:
            self.rows_cache[name] = read_jsonl(path)
        return self.rows_cache[name]

    def episode(self, scene_id, method="F"):
        path = self.main / scene_id / method / "episode.json"
        if path.exists():
            self.use_source(path)
            return read_json(path)
        failure = self.main / scene_id / method / "failure.json"
        if failure.exists():
            self.use_source(failure)
            return {"scene_id": scene_id, "method": method, "failure": read_json(failure),
                    "report": None, "raw_report": None, "actions": [], "schema_valid": False}
        raise UnavailableFigure(f"No recorded {method} episode for {scene_id}")

    def select_cases(self):
        if self.selections is not None:
            return self.selections
        used_groups, used_domains, used_scenes = set(), set(), set()
        all_ids = self.case_policy["all_real_candidate_order"]
        if not all_ids:
            raise UnavailableFigure("No held-out real cases")
        def ordered(ids):
            return sorted(ids, key=lambda sid: (self.by_id[sid].get("group_id") in used_groups,
                                                self.by_id[sid].get("domain") in used_domains,
                                                sid in used_scenes, sid))
        def choose(ids, fallback, purpose):
            options = ordered(ids or fallback)
            sid = options[0]
            used_groups.add(self.by_id[sid].get("group_id")); used_domains.add(self.by_id[sid].get("domain")); used_scenes.add(sid)
            return {"scene_id": sid, "purpose": purpose, "criterion_met": bool(ids),
                    "group_id": self.by_id[sid].get("group_id"), "domain": self.by_id[sid].get("domain")}
        choices = {
            "figure02": choose(self.case_policy["tonal_candidate_order"], all_ids, "tonal/harmonic morphology reference"),
            "figure03": choose(self.case_policy["overlap_candidate_order"], all_ids, "co-occurring AI-reference components"),
        }
        zoom_ids = [sid for sid in all_ids if len(list((self.main / sid / "F").glob("action_*.png"))) >= 2]
        recorded_ids = [sid for sid in all_ids if (self.main / sid / "F/episode.json").exists() or (self.main / sid / "F/failure.json").exists()]
        if not recorded_ids:
            raise UnavailableFigure("No held-out F case records")
        choices["figure04"] = choose(zoom_ids, recorded_ids, "recorded adaptive inspections")
        evaluation = {r["scene_id"]: r for r in self.rows("main") if r["method"] == "F" and r["stratum"] == "real"}
        failures = [sid for sid in recorded_ids if sid in evaluation and (evaluation[sid]["schema_failed"] or evaluation[sid]["empty_report"])]
        if failures:
            choices["figure05"] = choose(failures, recorded_ids, "first schema failure or abstention")
        else:
            remaining = [sid for sid in recorded_ids if sid not in used_scenes] or recorded_ids
            candidates = [sid for sid in sorted(remaining) if evaluation.get(sid, {}).get("event_metrics") is not None]
            if not candidates:
                choices["figure05"] = choose([], recorded_ids, "recorded unresolved case; reference disagreement unavailable")
            else:
                differences = 1 - tensor([evaluation[sid]["event_metrics"]["f1"] for sid in candidates])
                sid = candidates[differences.argmax().item()]
                choices["figure05"] = choose([sid], recorded_ids, "largest AI-reference disagreement; outcome-based selection disclosed")
        choices["selection_note"] = "Cases2/3 depend on AI references and opaque IDs; case4 depends on recorded image availability, not accuracy; case5 is a declared failure/outcome selection."
        choices["original_recording_count"] = len({choices[k]["group_id"] for k in ("figure02", "figure03", "figure04", "figure05")})
        choices["domain_count"] = len({choices[k]["domain"] for k in ("figure02", "figure03", "figure04", "figure05")})
        write_json(self.output / "selected_cases.json", choices)
        self.selections = choices
        return choices

    def save(self, number, figure, payload, caption):
        name = f"figure{number:02d}"
        typography = None
        if number <= 7:
            # Rendering dimensions/font sizes are host presentation metadata,
            # not new experimental arithmetic or metric aggregation.
            figure.canvas.draw()
            width_in = figure.get_tightbbox(figure.canvas.get_renderer()).width + .2
            sizes = [item.get_fontsize() for item in figure.findobj(matplotlib.text.Text)
                     if item.get_visible() and item.get_text().strip()]
            printed_font = min(sizes) * 4.8 / width_in
            typography = {"paper_width_in": 4.8, "export_width_in": width_in,
                          "minimum_external_source_font_pt": min(sizes),
                          "minimum_external_printed_font_pt": printed_font,
                          "embedded_action_raster_text": "Preserved unchanged; readable external coordinates/evidence labels supplied"}
            if printed_font < 7:
                raise UnavailableFigure(f"{name} external text would be {printed_font:.2f}pt at paper width; reflow the presentation")
        payload = dict(payload, artifact=name, caption=caption,
                       numeric_device="cuda", rendering="Matplotlib on host",
                       paper_typography=typography,
                       overview_display_scales=dict(self.paper_display_scales),
                       source_files=dict(self.sources))
        input_path = self.inputs / (name + ".json")
        write_json(input_path, payload)
        outputs = []
        for suffix in ("pdf", "png", "svg"):
            path = self.output / f"{name}.{suffix}"
            figure.savefig(path, dpi=300, bbox_inches="tight", metadata={"Creator": "Recorded acoustic experiment figure builder"} if suffix == "pdf" else None)
            outputs.append({"path": relative(path, self.root), "sha256": digest(path)})
        plt.close(figure)
        paper = self.root / self.args.paper_dir
        paper.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.output / f"{name}.pdf", paper / f"{name}.pdf")
        self.artifacts.append({"figure": number, "caption": caption, "plot_inputs": relative(input_path, self.root),
                               "plot_inputs_sha256": digest(input_path), "outputs": outputs})

    def case_data(self, sid):
        if sid in self.case_cache:
            return self.case_cache[sid]
        record = self.by_id[sid]
        self.use_source(self.root / record["path"])
        waveform, rate, origin, _ = load_waveform(record)
        views = {"lofar": lofar(waveform, rate, window_s=1., hop_s=.25, offset_s=origin)}
        for name, band in (("demon_low", (500., 2000.)), ("demon_high", (2000., 5000.))):
            try:
                views[name] = demon(waveform, rate, band, offset_s=origin)
            except ValueError:
                continue
        for name, view in views.items():
            view.metadata["paper_scene_id"] = sid
            view.metadata["paper_view_name"] = name
        episode = self.episode(sid)
        reference = self.references.get(sid, {})
        data = {"record": record, "waveform": waveform, "rate": rate, "origin": origin,
                "views": views, "episode": episode, "reference": reference}
        arrays_path = self.inputs / (sid + "_views.pt")
        torch.save({name: {"psd": v.psd.cpu(), "time_s": v.times_s.cpu(),
                           "frequency_hz": v.frequencies_hz.cpu(), "metadata": v.metadata}
                    for name, v in views.items()}, arrays_path)
        data["view_arrays"] = {"path": relative(arrays_path, self.root), "sha256": digest(arrays_path)}
        self.case_cache[sid] = data
        return data

    def draw_view(self, axis, view, title, reference=(), prediction=(), overlap=False,
                  reference_prefix="R", prediction_prefix="M", reference_uncertainty=True):
        power = view.log_power()
        limits = torch.quantile(power.flatten(), tensor([.05, .995], dtype=power.dtype)).tolist()
        display_key = f"{view.metadata['paper_scene_id']}:{view.metadata['paper_view_name']}"
        if display_key not in self.paper_display_scales:
            display_path = self.inputs / (display_key.replace(":", "_") + "_paper_display.pt")
            # Preserve values already computed above on CUDA, so later CPU
            # layout work need not recalculate the log/quantile transforms.
            torch.save({"log_PSD_db": power.cpu(), "time_s": view.times_s.cpu(),
                        "frequency_hz": view.frequencies_hz.cpu(), "limits_db": limits,
                        "metadata": view.metadata}, display_path)
            self.paper_display_scales[display_key] = {
                "log_PSD_limits_db": limits, "percentiles": [5, 99.5],
                "display_tensor": {"path": relative(display_path, self.root), "sha256": digest(display_path)},
                "meaning": "Redrawn paper overview contrast; not identical to the recorded tool/model-input raster display",
                "numeric_device": "cuda"}
        mesh = axis.pcolormesh(view.times_s.cpu().numpy(), view.frequencies_hz.cpu().numpy(),
                               power.cpu().numpy(), cmap="magma", vmin=limits[0], vmax=limits[1],
                               shading="nearest", rasterized=True)
        domain = view.metadata["frequency_domain"]
        carrier = view.metadata.get("carrier_band_hz")
        if carrier:
            title += f"\nCarrier {carrier[0]:g}–{carrier[1]:g} Hz"
        axis.set(title=title, xlabel="Original-recording time (s)",
                 ylabel=("Modulation" if domain == "modulation" else "Acoustic") + " frequency (Hz)")
        axis.tick_params(labelsize=CASE_FONT_PT)
        axis.title.set_fontsize(CASE_FONT_PT)
        axis.xaxis.label.set_fontsize(CASE_FONT_PT)
        axis.yaxis.label.set_fontsize(CASE_FONT_PT)
        for events, style, width, alpha in ((reference, "--", 1.3, .9), (prediction, "-", 1.3, .9)):
            for index, event in enumerate(events):
                if event.get("frequency_domain", "acoustic") != domain:
                    continue
                if domain == "modulation" and event.get("carrier_band_hz") != carrier:
                    continue
                color = "#67E8F9" if style == "--" else COLORS[index % len(COLORS)]
                track = event.get("frequency_track")
                interval = event.get("time_s")
                frequencies = event.get("frequency_hz")
                if not interval or not frequencies:
                    continue
                narrowband = event.get("type") in ("tone", "chirp", "modulation")
                if track and narrowband:
                    coordinates = tensor(track)
                    axis.plot(coordinates[:, 0].cpu(), coordinates[:, 1].cpu(), style, color=color, lw=width, alpha=alpha)
                    anchor = coordinates[-1]
                elif narrowband:
                    center = tensor(frequencies).mean().item()
                    axis.plot(interval, [center, center], style, color=color, lw=width, alpha=alpha)
                    anchor = tensor([interval[-1], center])
                else:
                    span = tensor([interval, frequencies])
                    sizes = (span[:, 1] - span[:, 0]).tolist()
                    axis.add_patch(Rectangle((interval[0], frequencies[0]), sizes[0], sizes[1],
                                             fill=False, edgecolor=color, linestyle=style, linewidth=width))
                    anchor = span[:, 1]
                # Label geometry in the same coordinate system as its saved
                # report. Off-view proposals remain in the callout key only.
                support = tensor([interval, frequencies])
                visible = ((support[0, 1] >= view.times_s[0]) & (support[0, 0] <= view.times_s[-1]) &
                           (support[1, 1] >= view.frequencies_hz[0]) & (support[1, 0] <= view.frequencies_hz[-1]))
                if bool(visible):
                    anchor = torch.stack((anchor[0].clamp(view.times_s[0], view.times_s[-1]),
                                          anchor[1].clamp(view.frequencies_hz[0], view.frequencies_hz[-1])))
                    prefix = reference_prefix if style == "--" else prediction_prefix
                    axis.annotate(f"{prefix}{index + 1}", anchor.tolist(), xytext=(-3, 3),
                                  textcoords="offset points", ha="right", fontsize=CASE_FONT_PT,
                                  color="#111827", bbox={"facecolor": "white", "alpha": .92,
                                                         "edgecolor": color, "linewidth": .6, "pad": .7})
                if reference_uncertainty and style == "--" and event.get("frequency_uncertainty_hz", 0) and narrowband:
                    uncertainty = tensor(event["frequency_uncertainty_hz"]).clamp(max=8.)
                    if track:
                        axis.fill_between(coordinates[:, 0].cpu(), (coordinates[:, 1] - uncertainty).cpu(),
                                          (coordinates[:, 1] + uncertainty).cpu(), color=color, alpha=.1, linewidth=0)
                    else:
                        center = tensor(frequencies).mean()
                        axis.fill_between(interval, (center - uncertainty).item(), (center + uncertainty).item(),
                                          color=color, alpha=.1, linewidth=0)
        if overlap:
            relevant = [e for e in reference if e.get("frequency_domain", "acoustic") == domain]
            for first in range(len(relevant)):
                for second in range(first + 1, len(relevant)):
                    time = tensor([relevant[first]["time_s"], relevant[second]["time_s"]])
                    low, high = time[:, 0].max(), time[:, 1].min()
                    if bool(high > low):
                        axis.axvspan(low.item(), high.item(), color="#56B4E9", alpha=.035)
        axis.set_xlim(view.times_s[0].item(), view.times_s[-1].item())
        axis.set_ylim(view.frequencies_hz[0].item(), view.frequencies_hz[-1].item())
        return mesh

    def callout_key(self, axis, events, prefix="M"):
        rows = component_callouts(events, prefix)
        axis.axis("off")
        if not rows:
            axis.text(0, .8, "No retained report components; no callouts to draw.", fontsize=CASE_FONT_PT)
        else:
            for index, row in enumerate(rows):
                column, line = index % 2, index // 2
                evidence = ", ".join(str(value) for value in row["evidence_ids"][:3]) or "none cited"
                if len(row["evidence_ids"]) > 3:
                    evidence += f" (+{len(row['evidence_ids']) - 3} more)"
                label = f"{row['callout']} = {row['component_id']} ({row['type']})\nEvidence: {evidence}"
                axis.text(column * .51, 1 - line * .32,
                          label,
                          va="top", fontsize=CASE_FONT_PT, color="#111827")
        return rows

    def component_legend(self, fig, reference_present):
        handles = [Line2D([], [], color=COLORS[1], label="F report proposals")]
        if reference_present:
            handles.insert(0, Line2D([], [], color="#0891B2", linestyle="--", label="Eligible AI reference (uncertain)"))
        fig.legend(handles=handles,
                   loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, -.02), fontsize=CASE_FONT_PT)

    def figure1(self):
        chosen = self.select_cases()["figure04"]
        episode = self.episode(chosen["scene_id"])
        actions = episode.get("actions", [])
        if not actions:
            raise UnavailableFigure("No recorded action trace for method diagram")
        fig, axes = plt.subplots(2, 1, figsize=(CASE_WIDTH_IN, 4.4), gridspec_kw={"height_ratios": [1, 1.45]}, constrained_layout=True)
        axes[0].set(xlim=(-.2, 10.2), ylim=(-.45, 2)); axes[0].axis("off")
        labels = ["Waveform", "CUDA views\nLOFAR/DEMON", "Evidence\ngraph", "Inspection\ncontroller", "Verifier +\nreport"]
        for index, label in enumerate(labels):
            x = index * 2
            axes[0].add_patch(FancyBboxPatch((x + .05, .65), 1.7, .7, boxstyle="round,pad=.08", facecolor="#EAF2F8", edgecolor="#44647A"))
            axes[0].text(x + .9, 1, label, ha="center", va="center", fontsize=CASE_FONT_PT)
            if index < 4:
                axes[0].annotate("", xy=(x + 2.03, 1), xytext=(x + 1.85, 1), arrowprops={"arrowstyle": "->", "color": "#44647A"})
        axes[0].annotate("", xy=(2.8, .55), xytext=(6.9, .55), arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=-.25", "color": COLORS[1]})
        axes[0].text(4.9, -.3, "Bounded reinspection of the original waveform", ha="center", fontsize=CASE_FONT_PT)
        axes[0].set_title("Conceptual computation path", loc="left")
        axes[1].axis("off")
        selected = actions[:3]
        table_rows = [[str(a.get("step", i)), textwrap.fill(a.get("action", {}).get("name", "invalid action").replace("_", " "), width=17),
                       textwrap.fill(a.get("status", "unknown").replace("_", " "), width=12),
                       textwrap.fill(textwrap.shorten(str(a.get("reason", "")), width=64, placeholder="…"), width=35)] for i, a in enumerate(selected)]
        table = axes[1].table(cellText=table_rows, colLabels=["Step", "Operation", "Outcome", "Recorded purpose"],
                              cellLoc="left", colWidths=[.07, .24, .19, .50], bbox=[0, 0, 1, .84])
        table.auto_set_font_size(False); table.set_fontsize(CASE_FONT_PT)
        axes[1].set_title(f"Recorded F trace · {chosen['scene_id']}\nFirst {len(selected)} of {len(actions)} attempts; full trace in artifact", loc="left")
        self.save(1, fig, {"scene_id": chosen["scene_id"], "actions": selected, "complete_actions": actions,
                          "shown_trace": "First three attempted actions; purpose text shortened for display only"},
                  "Conceptual method and the first three recorded attempts from a held-out action trace. Failures in the shown prefix remain visible; the complete trace is preserved.")

    def acoustic_case(self, number):
        chosen = self.select_cases()[f"figure{number:02d}"]
        data = self.case_data(chosen["scene_id"])
        reference = get_events(data["reference"])
        predicted, _, _ = report_events(data["episode"].get("report"))
        overlay_title = "LOFAR: reference / proposals" if reference else "LOFAR: report proposals"
        panels = [("lofar", "LOFAR: raw", [], []), ("lofar", overlay_title, reference, predicted)]
        for name in ("demon_low", "demon_high"):
            if name in data["views"]:
                panels.append((name, "Linked DEMON", reference, predicted))
        fig = plt.figure(figsize=(CASE_WIDTH_IN, 6.6), constrained_layout=True)
        grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, .7])
        axes = [fig.add_subplot(grid[row, column]) for row in range(2) for column in range(2)]
        for axis, (name, title, refs, preds) in zip(axes, panels):
            mesh = self.draw_view(axis, data["views"][name], title, refs, preds, overlap=number == 3)
            bar = fig.colorbar(mesh, ax=axis, label="PSD (dB)", shrink=.82, pad=.015)
            bar.ax.tick_params(labelsize=CASE_FONT_PT)
            bar.ax.yaxis.label.set_fontsize(CASE_FONT_PT)
        for axis in axes[len(panels):]:
            fig.delaxes(axis)
        callouts = self.callout_key(fig.add_subplot(grid[2, :]), predicted)
        title = "Tonal morphology" if number == 2 and chosen["criterion_met"] else "Co-occurring components" if number == 3 and chosen["criterion_met"] else "Real recording: reference criterion unmet"
        scope = "Eligible AI references remain uncertain" if reference else "No qualified AI reference; model proposals only"
        fig.suptitle(f"{title}\n{chosen['scene_id']} ({chosen['domain']})\n{scope}", fontsize=CASE_FONT_PT + 1)
        self.component_legend(fig, bool(reference))
        self.save(number, fig, {"selection": chosen, "reference": data["reference"],
                               "prediction": data["episode"].get("report"), "component_callouts": callouts,
                               "callout_rule": "Original report order; matching labels across compatible domains/carriers. Up to three evidence IDs shown per key entry; all IDs retained in inputs. Off-view components remain in key.",
                               "view_arrays": data["view_arrays"]},
                  f"Recorded real-scene displays for {chosen['scene_id']}; " +
                  ("dashed AI references are eligibility-limited and not independent physical truth. Other morphologies remain unscored. " if reference else
                   "reference qualification is unavailable. Solid overlays are model-proposed morphology, not physical truth. ") +
                  ("Temporal coexistence does not establish multiple physical sources. " if number == 3 else "Frequencies and carrier bands retain their distinct meanings. ") +
                  "PSD is dB re uncalibrated digital amplitude squared per Hz.")

    def figure4(self):
        chosen = self.select_cases()["figure04"]
        data = self.case_data(chosen["scene_id"])
        actions = data["episode"].get("actions", [])
        selected, images = [], []
        for entry in actions:
            recorded = sorted((self.main / chosen["scene_id"] / "F").glob(f"action_{entry.get('step')}_*.png"))
            for path in recorded:
                self.use_source(path)
                images.append(path); selected.append(entry)
                if len(images) == 2:
                    break
            if len(images) == 2:
                break
        if not images:
            raise UnavailableFigure("Selected adaptive case has no recorded action-view image")
        fixed = self.episode(chosen["scene_id"], "G3")
        fig = plt.figure(figsize=(CASE_WIDTH_IN, 7.0), constrained_layout=True)
        grid = fig.add_gridspec(3, len(images), height_ratios=[1.6, 1.7, 2.0])
        global_axis = fig.add_subplot(grid[0, :])
        self.draw_view(global_axis, data["views"]["lofar"], "Global LOFAR: retained proposals", get_events(data["reference"]),
                       report_events(data["episode"].get("report"))[0])
        action_labels = []
        for index, (path, entry) in enumerate(zip(images, selected)):
            axis = fig.add_subplot(grid[1, index])
            # Reading an existing raster is rendering only; the recorded tool
            # image preserves the exact crop, window and evidence identifier.
            axis.imshow(plt.imread(str(path))); axis.axis("off")
            action = entry.get("action", {})
            window = action.get("window_s", 4. if action.get("name") == "inspect_modulation" else 1.)
            evidence_id = entry.get("result", {}).get("id", "no evidence ID")
            result = entry.get("result", {})
            time = action.get("time_s", result.get("frame_center_time_s"))
            frequency = action.get("frequency_hz", result.get("frequency_hz"))
            carrier = action.get("carrier_band_hz", result.get("carrier_band_hz"))
            interval = lambda value, unit: f"{value[0]:.4g}–{value[1]:.4g} {unit}" if value else "not recorded"
            coordinate_label = interval(time, "s") + " · " + interval(frequency, "Hz")
            if carrier:
                coordinate_label += "\ncarrier " + interval(carrier, "Hz")
            label = (f"Step {entry.get('step')} · {evidence_id}\n{action.get('name')} · {window:g}s window\n" + coordinate_label)
            action_labels.append(label)
            axis.set_title(label, fontsize=CASE_FONT_PT)
        comparison_axis = fig.add_subplot(grid[2, :]); comparison_axis.axis("off")
        comparison = []
        for method, saved in (("G3", fixed), ("F", data["episode"])):
            components = component_callouts(report_events(saved.get("report"))[0])
            excerpt = "\n".join(component_description(row, coordinates=True) for row in components[:1])
            if not excerpt:
                report = saved.get("report")
                unresolved = report.get("unresolved_questions", []) if isinstance(report, dict) else []
                excerpt = "No retained components. " + (str(unresolved[0]) if unresolved else "No unresolved text recorded.")
            comparison.append({"method": method, "attempted_calls": len(saved.get("actions", [])),
                               "components": components, "shown_component_order": "first component in saved report order",
                               "excerpt": excerpt})
        for index, row in enumerate(comparison):
            text = f"{row['method']} · {row['attempted_calls']} attempted calls\n" + textwrap.fill(row["excerpt"], width=82)
            comparison_axis.text(0, 1 - index * .49, text, va="top", fontsize=CASE_FONT_PT,
                                 linespacing=1.2, color="#111827")
        fig.suptitle(f"Recorded adaptive inspections\n{chosen['scene_id']}", fontsize=CASE_FONT_PT + 1)
        fig.text(.5, -.015, "First component per report; full records in artifact. No accuracy claim.", ha="center", fontsize=CASE_FONT_PT)
        self.save(4, fig, {"selection": chosen, "selected_actions": selected,
                          "visible_fixed_adaptive_comparison": comparison,
                          "external_action_coordinate_labels": action_labels,
                          "fixed_G3_report": fixed.get("report"), "F_report": data["episode"].get("report"),
                          "recorded_action_images": [{"path": relative(path, self.root), "sha256": digest(path)} for path in images],
                          "view_arrays": data["view_arrays"]},
                  "Global evidence, exact recorded F inspection images, and saved G3/F report excerpts, with coordinates and evidence IDs retained. Report-order excerpts are descriptive, not correctness judgments.")

    def figure5(self):
        chosen = self.select_cases()["figure05"]
        data = self.case_data(chosen["scene_id"])
        episode = data["episode"]
        fig = plt.figure(figsize=(CASE_WIDTH_IN, 6.5), constrained_layout=True)
        grid = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.1, 1.35])
        raw = report_events(episode.get("raw_report"))[0]
        retained = report_events(episode.get("report"))[0]
        acoustic_axis = fig.add_subplot(grid[0, 0])
        self.draw_view(acoustic_axis, data["views"]["lofar"], "Raw / retained acoustic proposals", raw, retained,
                       reference_prefix="U", reference_uncertainty=False)
        modulation_axis = fig.add_subplot(grid[1, 0])
        modulation = data["views"].get("demon_low")
        if modulation is not None:
            self.draw_view(modulation_axis, modulation, "Linked envelope evidence", raw, retained,
                           reference_prefix="U", reference_uncertainty=False)
        else:
            modulation_axis.axis("off")
            modulation_axis.text(0, .8, "DEMON unavailable for this waveform.", fontsize=CASE_FONT_PT)
        text_axis = fig.add_subplot(grid[2, 0]); text_axis.axis("off")
        audit = episode.get("verification_audit", [])
        withheld = [a for a in audit if a.get("status") == "withheld"]
        schema_valid = episode.get("schema_valid")
        schema_status = "valid" if schema_valid is True else "invalid" if schema_valid is False else "not recorded"
        schema_errors = episode.get("schema_errors", [])
        lines = [textwrap.shorten(chosen["purpose"], width=100, placeholder="…"),
                 f"Report schema: {schema_status}.",
                 f"Raw components: {len(raw)}; retained: {len(retained)}; numeric filter withheld: {len(withheld)}."]
        if schema_valid is False and schema_errors:
            lines.append("Schema error: " + textwrap.shorten(str(schema_errors[0]), width=105, placeholder="…"))
        if episode.get("failure"):
            lines.append("Execution failure: " + str(episode["failure"].get("error", "unknown"))[:160])
        for row in withheld[:1]:
            lines.append(textwrap.shorten(str(row.get("component_id", "component")) + ": " + "; ".join(row.get("reasons", [])), width=160, placeholder="…"))
        report = episode.get("report") if isinstance(episode.get("report"), dict) else {}
        if schema_valid is not False:
            lines += [textwrap.shorten(str(question), width=120, placeholder="…") for question in report.get("unresolved_questions", [])[:1]]
        callouts = component_callouts(raw, "U") + component_callouts(retained)
        lines += [component_description(row) for row in callouts[:1]]
        lines.append("First recorded items shown; complete audit and callouts in artifact.")
        wrapped = "\n".join(textwrap.fill(line, width=82) if line else "" for line in lines)
        text_axis.text(0, 1, wrapped, va="top", fontsize=CASE_FONT_PT, linespacing=1.2)
        fig.suptitle(f"Failure / abstention record\n{chosen['scene_id']}", fontsize=CASE_FONT_PT + 1)
        fig.legend(handles=[Line2D([], [], ls="--", color="#0891B2", label="U: plottable raw component"),
                            Line2D([], [], color=COLORS[0], label="M: retained report proposal")],
                   loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(.5, .015), fontsize=CASE_FONT_PT)
        self.save(5, fig, {"selection": chosen, "raw_report": episode.get("raw_report"),
                          "report": episode.get("report"), "verification_audit": audit,
                          "report_schema_valid": schema_valid, "report_schema_errors": schema_errors,
                          "withheld_count_scope": "Component numerical filter only; schema-invalid whole-report rejection is shown separately.",
                          "component_callouts": callouts,
                          "malformed_raw_components": "Unplottable/malformed rows remain in raw report and verification audit; they are not converted into geometry.",
                          "view_arrays": data["view_arrays"]},
                  "A declared failure or disagreement case, separating report schema failure from numerical component filtering. Raw components can belong to an invalid report; no verified physical interpretation is claimed.")

    def score_panel(self, axis, results, title):
        methods = [m for m in ("G1", "G2", "G3", "G4", "F") if m in results]
        points = []
        for index, method in enumerate(methods):
            value = results[method]
            metric = value.get("event_metrics")
            if metric is None:
                continue
            interval = value.get("event_group_intervals", {}).get("metrics", {}).get("f1", {}).get("ci95")
            color = METHOD_COLORS[method]
            axis.plot(metric["f1"], index, "o", color=color)
            if interval:
                axis.plot(interval, [index, index], color=color, lw=1.5)
            points.append({"method": method, "f1": metric["f1"], "ci95": interval,
                           "episodes": value["reference_scored_episodes"], "groups": value["independent_groups"]})
        if not points:
            raise UnavailableFigure("No defined evaluated event scores")
        axis.set(yticks=list(range(len(methods))), yticklabels=methods, xlim=(0, 1), xlabel="Micro event F1", title=title)
        axis.invert_yaxis(); axis.grid(axis="x", color="#E5E7EB")
        return points

    def operation_panel(self, axis, results):
        methods = [m for m in ("G1", "G2", "G3", "G4", "F") if m in results]
        points = []
        for index, method in enumerate(methods):
            value = results[method]
            schema_valid = (1 - tensor(value["totals"]["schema_failed"]) / tensor(value["expected_episodes"])).item()
            axis.plot(value["report_emission_coverage"], index, "o", color=METHOD_COLORS[method])
            axis.plot(schema_valid, index, "s", markerfacecolor="none", color=METHOD_COLORS[method])
            points.append({"method": method, "report_emission": value["report_emission_coverage"],
                           "schema_valid_fraction": schema_valid, "expected_episodes": value["expected_episodes"],
                           "schema_failures": value["totals"]["schema_failed"]})
        axis.set(yticks=list(range(len(methods))), yticklabels=methods, xlim=(0, 1), xlabel="Recorded episode fraction",
                 title="Real: reference unavailable\nReport production only")
        axis.invert_yaxis(); axis.grid(axis="x", color="#E5E7EB")
        return points

    def figure6(self):
        summary = self.summary("main")
        available = [s for s in ("real", "synthetic") if s in summary["results"]]
        if not available:
            raise UnavailableFigure("Main quantitative strata absent")
        perception_path = self.root / "results/perception/summary.json"
        perception = read_json(perception_path) if perception_path.exists() else None
        localization_path = self.root / "results/localization/predictions/evaluation/summary.json"
        localization = read_json(localization_path) if localization_path.exists() else None
        if (self.root / "results/localization/COMPLETED.json").exists() and localization is None:
            raise UnavailableFigure("Completed localizers require common-evaluator scores before the comparison figure")
        count = len(available) + int(perception is not None) + int(localization is not None)
        if count == 4:
            fig, grid = plt.subplots(2, 2, figsize=(CASE_WIDTH_IN, 6.1), constrained_layout=True)
            axes = [axis for row in grid for axis in row]
        else:
            fig, grid = plt.subplots(count, 1, figsize=(CASE_WIDTH_IN, 2.4 * count), constrained_layout=True)
            axes = [grid] if count == 1 else list(grid)
        payload = {"unavailable_strata": [s for s in ("real", "synthetic") if s not in available],
                   "unavailable_rule": "No defined eligible-reference score is plotted or replaced by zero."}
        for axis, stratum in zip(axes, available):
            results = summary["results"][stratum]["all"]
            if stratum == "real" and not any(v.get("event_metrics") is not None for v in results.values()):
                payload[stratum] = {"primary_agreement": None, "operation_diagnostics": self.operation_panel(axis, results)}
                freeze = self.root / "configs/ai_reference_freeze.json"
                if freeze.exists():
                    self.use_source(freeze); payload["reference_qualification"] = read_json(freeze)
            else:
                payload[stratum] = self.score_panel(axis, results,
                    "Real: AI-reference agreement" if stratum == "real" else "Exact synthetic recovery")
            payload[stratum + "_reference_scopes"] = {m: v.get("reference_scopes") for m, v in summary["results"][stratum]["all"].items()}
        if localization:
            self.use_source(localization_path)
            axis = axes[len(available)]
            localized = localization.get("results", {}).get("synthetic", {}).get("all", {})
            values = []
            for index, prefix in enumerate(("P1_localizer", "P2_localizer")):
                members = [(name, entry) for name, entry in sorted(localized.items()) if name.startswith(prefix + "_seed")]
                if len(members) != 3 or any(entry.get("event_metrics") is None for _, entry in members):
                    raise UnavailableFigure("Localizer comparison requires three defined common-evaluator synthetic seed scores")
                scores = tensor([entry["event_metrics"]["f1"] for _, entry in members])
                mean, sd = scores.mean(), scores.std(unbiased=True)
                axis.plot(torch.full_like(scores, index).cpu(), scores.cpu(), "x", color=COLORS[index], alpha=.7)
                axis.plot(index, mean.item(), "o", color=COLORS[index])
                limits = (mean + scores.new_tensor([-1., 1.]) * sd).clamp(0, 1).tolist()
                axis.plot([index, index], limits, color=COLORS[index])
                values.append({"method": prefix, "mean": mean.item(), "sample_sd": sd.item(),
                               "seeds": [{"method": name, "f1": entry["event_metrics"]["f1"],
                                          "episodes": entry["reference_scored_episodes"],
                                          "groups": entry["independent_groups"]} for name, entry in members]})
            axis.set(xticks=[0, 1], xticklabels=["P1", "P2"], ylim=(0, 1),
                     ylabel="Micro event F1", title="Exact synthetic: localizers\nThree seeds; mean ± SD")
            payload["localization"] = {"values": values,
                "scope": "Common exact synthetic evaluator; seed variation is conditional on the shared test scenes, not independent data replication."}
        if perception:
            self.use_source(perception_path)
            axis = axes[-1]
            values = []
            for index, method in enumerate(("P1", "P2")):
                entry = perception["seed_variability"][method].get("macro_f1_present_reference_classes")
                if entry is None:
                    raise UnavailableFigure("Perception summary predates present-class metric correction")
                axis.plot(index, entry["mean"], "o", color=COLORS[index])
                if entry["sample_sd"] is not None:
                    interval = (tensor(entry["mean"]) + tensor([-1., 1.]) * tensor(entry["sample_sd"])).clamp(0, 1).tolist()
                    axis.plot([index, index], interval, color=COLORS[index])
                values.append({"method": method, **entry})
            axis.set(xticks=[0, 1], xticklabels=["P1", "P2"], ylim=(0, 1),
                     ylabel="Present-class macro F1", title="Weak-label classification\nThree seeds; mean ± SD")
            payload["perception"] = {"values": values, "test_class_support": perception.get("test_class_support"),
                                      "scope": "No localization score; no generalization claim for test-absent classes."}
        if "operation_diagnostics" in payload.get("real", {}):
            fig.legend(handles=[Line2D([], [], marker="o", ls="", color="#475569", label="Real: nonempty report"),
                                Line2D([], [], marker="s", ls="", mfc="none", color="#475569", label="Real: valid schema")],
                       loc="lower center", bbox_to_anchor=(.5, -.025), ncol=2, frameon=False, fontsize=CASE_FONT_PT)
        self.save(6, fig, payload, "Controller and auxiliary-trained localizer exact synthetic outcomes, weak-label classification, and real report production are separate quantities. When AI qualification fails, primary real agreement stays unavailable; seed SD is not a confidence interval over independent recordings.")

    def figure7(self):
        real = self.summary("ablations")["results"]["real"]["all"]
        synthetic = self.summary("synthetic_ablations")["results"]["synthetic"]["all"]
        amendment_path = self.root / "configs/synthetic_ablation_amendment.json"
        if not amendment_path.exists():
            raise UnavailableFigure("Synthetic ablation plot requires its recorded pre-main amendment")
        self.use_source(amendment_path)
        amendment = read_json(amendment_path)
        cap_path = self.root / "configs/ablation_user_cap.json"
        cap = read_json(cap_path) if cap_path.exists() else None
        scope = ablation_display_scope(real, synthetic, amendment, cap)
        methods, synthetic_methods = scope["real_methods"], scope["synthetic_methods"]
        cap_provenance = {}
        if scope["user_cap_active"]:
            self.use_source(cap_path)
            for key, phase in (("capped", "ablations"), ("synthetic_reuse", "synthetic_ablations")):
                declared = cap[key]
                run_path = self.phase(phase) / "run.json"
                manifest_path = self.root / declared["manifest"]
                if not run_path.exists() or not manifest_path.exists():
                    raise UnavailableFigure(f"User-cap provenance absent: {key}")
                run = read_json(run_path)
                expected_methods = methods if key == "capped" else synthetic_methods
                if (self.phase(phase) != (self.root / declared["composed_run"]).resolve() or
                        digest(manifest_path) != declared["manifest_sha256"] or
                        run.get("manifest_sha256") != declared["manifest_sha256"] or
                        run.get("scene_count") != declared["scene_count"] or
                        set(run.get("methods", [])) != set(expected_methods)):
                    raise UnavailableFigure(f"Evaluated cohort differs from user-cap declaration: {key}")
                self.use_source(run_path); self.use_source(manifest_path)
                cap_provenance[key] = {"run": relative(run_path, self.root), "run_sha256": digest(run_path),
                                       "manifest": relative(manifest_path, self.root),
                                       "manifest_sha256": digest(manifest_path)}
            stub_path = self.root / cap["synthetic_reuse"]["stub_run"] / "run.json"
            proof_path = self.root / cap["synthetic_reuse"]["proof"]
            if not stub_path.exists() or not proof_path.exists():
                raise UnavailableFigure("Synthetic reuse-only proof is absent")
            if read_json(stub_path).get("methods") != []:
                raise UnavailableFigure("Capped synthetic display requires zero new inference methods")
            self.use_source(stub_path); self.use_source(proof_path)
            cap_provenance["synthetic_reuse_proof"] = read_json(proof_path)
            for key in ("reuse_provenance", "original_inventory"):
                path = self.root / cap["capped"][key]
                if not path.exists():
                    raise UnavailableFigure(f"Retained partial-run provenance absent: {key}")
                self.use_source(path)
                cap_provenance[key] = {"path": relative(path, self.root), "sha256": digest(path)}
        for name in ("run.json", "composition_provenance.json", "exact_truth_provenance.json"):
            source = self.phase("synthetic_ablations") / name
            if not source.exists():
                raise UnavailableFigure(f"Synthetic ablation provenance absent: {name}")
            self.use_source(source)
        labels = {"no_demon": "−DEMON", "no_graph": "−graph", "no_quantitative": "−tables",
                  "forced_exclusivity": "exclusive", "fixed_resolution": "fixed LOFAR",
                  "random": "random", "no_verifier": "F without\nnumeric filtering"}
        displayed = [labels.get(method, method) for method in methods]
        synthetic_displayed = [labels.get(method, method) for method in synthetic_methods]
        real_columns = ["Report\nemission", "Valid\nschema"]
        exact_columns = ["Event F1", "Precision", "Recall"]
        real_rows, exact_rows, cost = [], [], []
        for method in methods:
            value = real[method]
            schema_valid = (1 - tensor(value["totals"]["schema_failed"]) / tensor(value["expected_episodes"])).item()
            real_rows.append([value["report_emission_coverage"], schema_valid])
            cost.append(value["compute"]["cuda_event_seconds"]["mean_per_expected_episode"])
        for method in synthetic_methods:
            event = synthetic[method].get("event_metrics")
            if event is None:
                raise UnavailableFigure("Exact synthetic comparison event scoring is absent")
            exact_rows.append([event[key] if event[key] is not None else float("nan")
                               for key in ("f1", "precision", "recall")])
        real_values, exact_values = tensor(real_rows), tensor(exact_rows)
        fig = plt.figure(figsize=(CASE_WIDTH_IN, max(6.5, .64 * len(methods))), constrained_layout=True)
        grid = fig.add_gridspec(2, 2, width_ratios=[3, 2],
                               height_ratios=[2, 1] if scope["user_cap_active"] else [1, 1])
        axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
        real_n = sorted({entry["expected_episodes"] for entry in real.values()})
        exact_n = sorted({entry["expected_episodes"] for entry in synthetic.values()})
        if (len(real_n) != 1 or exact_n != [scope["synthetic_scene_count"]] or
                (scope["user_cap_active"] and real_n != [scope["real_scene_count"]])):
            raise UnavailableFigure("Comparison methods do not share their declared within-stratum denominator")
        real_title = f"Real n={real_n[0]}: execution"
        exact_title = f"Synthetic n={exact_n[0]}: exact"
        if scope["user_cap_active"]:
            domains = cap["capped"]["domain_counts"]
            real_title = f"Real n={real_n[0]}, {len(methods)} policies\n{domains['deepship']} DeepShip + {domains['mbari']} MBARI"
            exact_title = f"Synthetic n={exact_n[0]}: {len(synthetic_methods)} reused policies\nSix new variants omitted"
        for axis, values, columns, title, row_labels in (
                (axes[0], real_values, real_columns, real_title, displayed),
                (axes[2], exact_values, exact_columns, exact_title, synthetic_displayed)):
            axis.imshow(values.cpu().numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
            axis.set(yticks=list(range(len(row_labels))), yticklabels=row_labels,
                     xticks=list(range(len(columns))), xticklabels=columns, title=title)
            axis.tick_params(axis="both", labelsize=CASE_FONT_PT)
            for i, row in enumerate(values.tolist()):
                for j, value in enumerate(row):
                    finite = bool(torch.isfinite(values[i, j]))
                    axis.text(j, i, f"{value:.2f}" if finite else "undef.", ha="center", va="center",
                              color="white" if finite and value < .55 else "black", fontsize=CASE_FONT_PT)
        positions = torch.arange(len(methods), device="cuda")
        axes[1].barh(positions.cpu(), tensor(cost).cpu(), color=COLORS[0]); axes[1].invert_yaxis()
        axes[1].set(yticks=list(range(len(methods))), yticklabels=[""] * len(methods),
                    xlabel="Recorded CUDA s/\nepisode", title="Real: spans")
        axes[1].set_ylim(axes[0].get_ylim())
        axes[1].tick_params(axis="x", labelsize=CASE_FONT_PT)
        fig.suptitle("Post-main user cap: separate comparisons" if scope["user_cap_active"]
                     else "Secondary ablations: separate cohorts", fontsize=CASE_FONT_PT + 1)
        def clean(values):
            return [[None if not bool(torch.isfinite(x)) else x.item() for x in row] for row in values]
        pooling = ("None; user-capped real execution and the three-policy synthetic reuse comparison are separate."
                   if scope["user_cap_active"] else "None; original real execution and synthetic exact quality are separate cohorts.")
        caption = ("User-requested runtime cap after main scoring: real-cohort report production/accounting and a limited synthetic reuse comparison, with separate method inventories. The real manifest prefix is domain-imbalanced; all six new synthetic variants are omitted."
                   if scope["user_cap_active"] else "Original real-cohort report production/accounting and pre-main amended exact-synthetic ablations remain separate.")
        self.save(7, fig, {"methods": methods, "displayed_methods": displayed,
            "method_inventories": {"real": methods, "synthetic": synthetic_methods},
            "real": {"methods": methods, "displayed_methods": displayed,
                     "columns": real_columns, "metrics": clean(real_values),
                     "primary_agreement_available": any(entry.get("event_metrics") is not None for entry in real.values()),
                     "cuda_seconds_per_episode": cost,
                     "reference_scored_episodes": [real[m]["reference_scored_episodes"] for m in methods],
                     "expected_episodes": [real[m]["expected_episodes"] for m in methods]},
            "synthetic": {"methods": synthetic_methods, "displayed_methods": synthetic_displayed,
                          "columns": exact_columns, "metrics": clean(exact_values),
                          "reference_scored_episodes": [synthetic[m]["reference_scored_episodes"] for m in synthetic_methods],
                          "expected_episodes": [synthetic[m]["expected_episodes"] for m in synthetic_methods],
                          "reference_scopes": {m: synthetic[m].get("reference_scopes") for m in synthetic_methods},
                          "event_group_intervals": {m: synthetic[m].get("event_group_intervals") for m in synthetic_methods}},
            "amendment": amendment, "user_cap": cap, "user_cap_provenance": cap_provenance,
            "scope_validation": scope, "pooling": pooling,
            "comparison": "Secondary within-cohort comparison; not confirmatory real accuracy or deltas against another cohort."},
            caption + " Primary real agreement is unavailable. F without numerical filtering retains report schema checks; F/G4 and this condition reuse existing inference, and their recorded spans do not estimate verifier savings.")

    def figure8(self):
        rows = self.rows("robustness")
        run = read_json(self.phase("robustness") / "run.json")
        self.use_source(self.phase("robustness") / "run.json")
        self.use_source(self.root / run["manifest"])
        manifest = read_jsonl(self.root / run["manifest"])
        by_id = {r["scene_id"]: r for r in manifest}
        eligible = [r for r in rows if r.get("event_metrics") is not None and r.get("reference_valid_for_primary")]
        if not eligible:
            raise UnavailableFigure("No scored controlled robustness episodes")
        methods = sorted({r["method"] for r in eligible}, key=lambda m: (m != "F", m))
        def condition(record):
            intervention = record["intervention"]
            if intervention["name"] == "added_noise":
                return "noise " + str(intervention["nominal_ratio_db"]) + " dB"
            if intervention["name"] == "gain":
                return "gain ×" + str(intervention["linear_gain"])
            return intervention["name"].replace("_", " ")
        conditions = sorted({condition(by_id[r["scene_id"]]) for r in eligible})
        condition_labels = ["blank: valid empty report" if name == "blank" else name for name in conditions]
        kinds = sorted({by_id[r["scene_id"]]["base_record"]["kind"] for r in eligible})
        fig, panels = plt.subplots(2, len(methods), figsize=(max(5, 4 * len(methods)), 6.6), squeeze=False, constrained_layout=True)
        axes=list(panels[0])
        payload = {"conditions": conditions, "synthetic_kinds": kinds, "methods": {},
                   "cell_metric": {name: "fraction_schema_valid_empty_reports" if name == "blank" else "micro_event_f1" for name in conditions}}
        for axis, method in zip(axes, methods):
            matrix = torch.full((len(conditions), len(kinds)), float("nan"), device="cuda")
            cell_counts = torch.zeros_like(matrix, dtype=torch.long)
            for i, name in enumerate(conditions):
                for j, kind in enumerate(kinds):
                    selected = [r for r in eligible if r["method"] == method and condition(by_id[r["scene_id"]]) == name
                                and by_id[r["scene_id"]]["base_record"]["kind"] == kind]
                    if selected:
                        if name == "blank":
                            matrix[i, j] = tensor([not r["schema_failed"] and r["reported_component_count"] == 0 for r in selected]).mean()
                        else:
                            matrix[i, j] = aggregate_event_metrics([r["event_metrics"] for r in selected])["f1"]
                        cell_counts[i, j] = len(selected)
            mesh = axis.imshow(matrix.cpu().numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
            axis.set(yticks=list(range(len(conditions))), yticklabels=condition_labels,
                     xticks=list(range(len(kinds))), xticklabels=kinds, title=method)
            axis.tick_params(axis="x", rotation=45)
            payload["methods"][method] = {"f1": [[v.item() if bool(torch.isfinite(v)) else None for v in row] for row in matrix],
                                          "episodes": cell_counts.tolist()}
        fig.colorbar(mesh, ax=axes, shrink=.75, label="Event F1; blank row: valid empty fraction")
        mixture_records={r['scene_id']:r for r in self.records if r['stratum']=='semi_synthetic'}
        mixture_rows=[r for r in self.rows('main') if r['scene_id'] in mixture_records]
        fractions=[0.,.25,.5,.75,1.];levels=[-10,0,10]
        payload['real_excerpt_mixtures']={'overlap_fraction':fractions,'first_to_second_RMS_power_ratio_db':levels,'methods':{},
            'meaning':'AI-reference agreement for mixtures of real excerpts, not source-separation accuracy. Sparse cells have different parents; no factorial interaction inference.'}
        for axis,method in zip(panels[1],methods):
            matrix=torch.full((len(fractions),len(levels)),float('nan'),device='cuda')
            counts=torch.zeros_like(matrix,dtype=torch.long)
            admitted=torch.zeros_like(matrix,dtype=torch.long)
            for i,fraction in enumerate(fractions):
                for j,level in enumerate(levels):
                    selected=[r for r in mixture_rows if r['method']==method and
                        mixture_records[r['scene_id']].get('nominal_overlap_fraction')==fraction and
                        mixture_records[r['scene_id']]['ratio_db']==level]
                    counts[i,j]=len(selected)
                    scored=[r for r in selected if r.get('reference_valid_for_primary') and r.get('event_metrics') is not None]
                    admitted[i,j]=len(scored)
                    if scored:matrix[i,j]=aggregate_event_metrics([r['event_metrics'] for r in scored])['f1']
            cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#e5e7eb')
            mixture_mesh=axis.imshow(matrix.cpu().numpy(),vmin=0,vmax=1,cmap=cmap,aspect='auto')
            for i in range(len(fractions)):
                for j in range(len(levels)):
                    label=f'{matrix[i,j].item():.2f}' if bool(torch.isfinite(matrix[i,j])) else ('no reference' if counts[i,j] else 'not sampled')
                    axis.text(j,i,label,ha='center',va='center',fontsize=6,color='black')
            axis.set(yticks=list(range(len(fractions))),yticklabels=['0%','25%','50%','75%','100%'],
                xticks=list(range(len(levels))),xticklabels=levels,xlabel='First/second excerpt power ratio (dB)',
                ylabel='Temporal overlap',title=f'{method} · real-excerpt mixtures')
            payload['real_excerpt_mixtures']['methods'][method]={'event_agreement_f1':[[v.item() if bool(torch.isfinite(v)) else None for v in row] for row in matrix],
                'sampled_episodes':counts.tolist(),'scorable_episodes':admitted.tolist()}
        fig.colorbar(mixture_mesh,ax=list(panels[1]),shrink=.75,label='AI-reference event agreement F1')
        fig.suptitle("Controlled sensitivity: exact synthetic truth and real-excerpt mixtures", fontsize=10)
        self.save(8, fig, payload, "Top: exact scoped synthetic outcomes; blank-input cells show the fraction of schema-valid empty reports. Bottom: the separately labelled sparse overlap/level grid of real-excerpt mixtures, scored only against admitted AI references. Unsampled and unavailable-reference cells are explicit. Different cells can have different parents; this is not a replicated factorial interaction analysis.")

    def risk_curve(self):
        rows = [r for r in self.rows("main") if r["method"] == "F" and r["stratum"] == "real"
                and r.get("event_metrics") is not None and r.get("reference_valid_for_primary")]
        scores, correct, provenance = [], [], []
        for row in rows:
            episode = self.episode(row["scene_id"])
            predictions, _, _ = report_events(episode.get("report"))
            predictions = score_scope(predictions, [], episode.get("report"),
                                      self.references.get(row["scene_id"], {}))[0]
            accepted = {m["prediction_index"] for m in row["event_metrics"]["matches"]}
            candidates = {c["id"]: c for c in episode.get("final_state", {}).get("components", [])}
            audits = {a.get("component_id"): a for a in episode.get("verification_audit", [])}
            for index, prediction in enumerate(predictions):
                matched = audits.get(prediction.get("id"), {}).get("matched_measurement_components", [])
                contrasts = [candidates[cid]["contrast_db"] for cid in matched if cid in candidates and "contrast_db" in candidates[cid]]
                if not contrasts:
                    continue
                score = tensor(contrasts).mean().item()
                scores.append(score); correct.append(index in accepted)
                provenance.append({"scene_id": row["scene_id"], "component_id": prediction.get("id"),
                                   "measurement_component_ids": matched, "contrast_db": score,
                                   "matches_AI_reference": index in accepted})
        if not scores:
            return None
        contrast, labels = tensor(scores), tensor(correct)
        order = torch.argsort(contrast, descending=True, stable=True)
        _, counts = torch.unique_consecutive(contrast[order], return_counts=True)
        endpoints = counts.cumsum(dim=0) - 1
        retained = (endpoints + 1).to(torch.float64)
        total_emitted = tensor([row.get("scoring_scope", {}).get("eligible_valid_predictions", row["valid_component_count"])
                                for row in rows]).sum() + tensor([row.get("scoring_scope", {}).get("eligible_invalid_predictions", 0)
                                                               for row in rows]).sum()
        return {"coverage": (retained / len(scores)).tolist(),
                "risk": ((1 - labels[order]).cumsum(dim=0)[endpoints] / retained).tolist(),
                "threshold_contrast_db": contrast[order][endpoints].tolist(), "ranked_components": len(scores),
                "eligible_verified_emitted_components": total_emitted.item(),
                "rankable_fraction_of_verified_emitted_components": (tensor(len(scores)) / total_emitted).item() if bool(total_emitted > 0) else None,
                "tie_policy": "Equal measurement-contrast scores enter together; no arbitrary within-tie curve points.",
                "prediction_provenance": provenance,
                "meaning": "Uncalibrated measurement-contrast ranking; risk is AI-reference disagreement, not physical error probability."}

    def budget_points(self, stratum="real", methods=("F",)):
        phases = ("budget2", "budget4", "main")
        if any(not (self.phase(phase) / "evaluation/per_episode.jsonl").exists() for phase in phases):
            raise UnavailableFigure("Matched budget plots require completed 2/4/6-slot evaluations")
        sets = {(phase, method): {r["scene_id"]: r for r in self.rows(phase)
                                 if r["method"] == method and r["stratum"] == stratum}
                for phase in phases for method in methods}
        common = set.intersection(*[set(values) for values in sets.values()])
        if not common:
            return []
        budget_config = read_json(self.phase("budget2") / "run.json")
        budget_manifest = self.root / budget_config["manifest"]
        self.use_source(budget_manifest)
        expected = {r["scene_id"] for r in read_jsonl(budget_manifest) if r["stratum"] == stratum}
        if common != expected:
            raise UnavailableFigure("Matched budget cohort differs from its declared stratum")
        points = []
        for (phase, method), values in sets.items():
            selected = [values[sid] for sid in sorted(common)]
            timings = tensor([[r["timing"][k] for k in ("preprocess_gpu_seconds", "model_gpu_seconds", "tool_gpu_seconds")] for r in selected])
            config = read_json(self.phase(phase) / "run.json")
            self.use_source(self.phase(phase) / "run.json")
            all_scored = all(r.get("event_metrics") is not None and r.get("reference_valid_for_primary") for r in selected)
            interval = None
            if all_scored and set(values) == common:
                entry = self.summary(phase)["results"][stratum]["all"][method]
                interval = entry.get("event_group_intervals", {}).get("metrics", {}).get("f1", {}).get("ci95")
            points.append({"phase": phase, "method": method, "stratum": stratum, "budget": config["budget"],
                           "event_f1": aggregate_event_metrics([r["event_metrics"] for r in selected])["f1"] if all_scored else None,
                           "event_f1_group_ci95": interval,
                           "mean_accounted_cuda_seconds": timings.sum(dim=1).mean().item(),
                           "mean_attempted_slots": tensor([r["actions_attempted"] for r in selected]).mean().item(),
                           "report_emission_fraction": (1 - tensor([r["empty_report"] for r in selected]).mean()).item(),
                           "schema_valid_fraction": (1 - tensor([r["schema_failed"] for r in selected]).mean()).item(),
                           "completed_episodes": tensor([r["execution_status"] == "completed" for r in selected]).sum().item(),
                           "common_scene_ids": sorted(common), "workers": config.get("workers", 1),
                           "cost_interpretation": config.get("timing_note", "Serial CUDA elapsed accounting")})
        return points

    def execution_accounting(self):
        result = {}
        pilot = (self.root / self.args.pilot_run).resolve()
        pilot_config = read_json(pilot / "run.json") if (pilot / "run.json").exists() else None
        if pilot_config and pilot_config.get("workers", 1) == 1:
            self.use_source(pilot / "run.json")
            paths = sorted(pilot.glob("*/*/episode.json"))
            for path in paths:
                self.use_source(path)
            episodes = [read_json(path) for path in paths]
            if episodes:
                wall = tensor([e["wall_seconds"] for e in episodes])
                result["serial_pilot"] = {"episodes": len(episodes), "median_wall_seconds": wall.median().item(),
                                          "mean_wall_seconds": wall.mean().item(),
                                          "scope": "Historical development protocol and serial pilot cohort only; different grammar/token protocol from final main run. No matched speedup inference."}
        completion = self.main / "COMPLETED.json"
        if completion.exists():
            self.use_source(completion)
            complete = read_json(completion)
            n = complete.get("new_completed_episodes")
            if n is not None:
                seconds = tensor(complete["finished_at_unix"]) - tensor(complete["started_at_unix"])
                if bool(seconds > 0):
                    result["main_last_invocation"] = {"new_completed_episodes": n, "wall_seconds": seconds.item(),
                        "episodes_per_minute": (tensor(n) / seconds * 60).item(), "workers": complete.get("workers", 1),
                        "scope": "New episodes in the recorded last invocation, including model setup; not serial episode latency"}
        return result

    def figure9(self):
        risk = self.risk_curve()
        budgets = self.budget_points("real", ("F", "G3", "G4"))
        synthetic_budgets = self.budget_points("synthetic", ("F", "G3", "G4"))
        accounting = self.execution_accounting()
        if synthetic_budgets and any(point["event_f1"] is None for point in synthetic_budgets):
            raise UnavailableFigure("Matched synthetic budget quality requires all scoped exact scores")
        count = int(risk is not None) + int(bool(budgets)) + int(bool(synthetic_budgets)) + int(bool(accounting))
        if not count:
            raise UnavailableFigure("No completed budget or execution-accounting records")
        if count == 3 and accounting:
            fig = plt.figure(figsize=(6.4, 4.2), constrained_layout=True)
            grid = fig.add_gridspec(2, 2, height_ratios=[2.7, 1.1])
            axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
        elif count == 4:
            fig, grid = plt.subplots(2, 2, figsize=(6.4, 5.2), constrained_layout=True)
            axes = [axis for row in grid for axis in row]
        else:
            fig, grid = plt.subplots(1, count, figsize=(3.1 * count, 3.2), constrained_layout=True)
            axes = [grid] if count == 1 else list(grid)
        panel = 0
        if risk:
            axes[panel].plot(risk["coverage"], risk["risk"], color=COLORS[0], marker=".", markersize=3)
            axes[panel].set(xlabel="Retained fraction of rankable components", ylabel="Unmatched-to-AI-reference fraction",
                            xlim=(0, 1), ylim=(0, 1), title="Uncalibrated contrast ranking")
            panel += 1
        if budgets:
            for method, marker in (("F", "o"), ("G3", "s"), ("G4", "^")):
                points = sorted([point for point in budgets if point["method"] == method], key=lambda point: point["budget"])
                axes[panel].plot([point["budget"] for point in points],
                                 [point["mean_accounted_cuda_seconds"] for point in points],
                                 marker=marker, color=METHOD_COLORS[method], label=method)
            axes[panel].set(xlabel="Attempted-slot ceiling", ylabel="Accounted CUDA s/expected episode",
                            xticks=sorted({point["budget"] for point in budgets}),
                            title=f"Real n={len(budgets[0]['common_scene_ids'])}: recorded spans\nPrimary agreement unavailable")
            axes[panel].legend(frameon=False, fontsize=8)
            panel += 1
        if synthetic_budgets:
            for method, marker in (("F", "o"), ("G3", "s"), ("G4", "^")):
                points = sorted([point for point in synthetic_budgets if point["method"] == method], key=lambda point: point["budget"])
                axes[panel].plot([point["mean_accounted_cuda_seconds"] for point in points],
                                 [point["event_f1"] for point in points], marker=marker,
                                 color=METHOD_COLORS[method], label=method)
                for point in points:
                    if point["event_f1_group_ci95"]:
                        axes[panel].plot([point["mean_accounted_cuda_seconds"]] * 2, point["event_f1_group_ci95"],
                                         color=METHOD_COLORS[method], alpha=.35, lw=1)
                    axes[panel].annotate(str(point["budget"]),
                        (point["mean_accounted_cuda_seconds"], point["event_f1"]),
                        xytext=(3, 3), textcoords="offset points", fontsize=7)
            axes[panel].set(xlabel="Accounted CUDA s/expected episode", ylabel="Exact synthetic micro event F1", ylim=(0, 1),
                            title=f"Synthetic n={len(synthetic_budgets[0]['common_scene_ids'])}: matched quality/cost\nLabels: 2/4/6-slot ceilings")
            axes[panel].legend(frameon=False, fontsize=8)
            panel += 1
        if accounting:
            axes[panel].axis("off")
            lines = []
            if risk is None:
                lines.append("Primary real risk–coverage unavailable: AI reference qualification failed.")
            if "serial_pilot" in accounting:
                entry = accounting["serial_pilot"]
                lines.append(f"Historical serial pilot: {entry['episodes']} episodes; median {entry['median_wall_seconds']:.1f} s/episode (earlier protocol).")
            if "main_last_invocation" in accounting:
                entry = accounting["main_last_invocation"]
                lines.append(f"Main: {entry['workers']} workers; {entry['episodes_per_minute']:.2f} episodes/min; {entry['new_completed_episodes']} new episodes.")
            lines.append("Verifier work is untimed separately and can overlap recorded spans: incomplete, nonexclusive accounting.")
            lines.append("Pilot/main protocols differ; no matched speedup or marginal/billable GPU-cost claim.")
            axes[panel].text(0, .97, "\n".join(lines), va="top", fontsize=8, linespacing=1.35)
        self.save(9, fig, {"contrast_ranking": risk, "matched_budget_points": budgets,
                          "matched_synthetic_budget_points": synthetic_budgets,
                          "pooling": "None: six-real cost and twelve-synthetic exact quality/cost use separate matched cohorts.",
                          "synthetic_interval_scope": "Bootstrap over declared generator-case groups; not real-population uncertainty.",
                          "synthetic_comparison_scope": "Preplanned secondary controlled budget comparison, not a substitute for real accuracy.",
                          "execution_accounting": accounting,
                          "timing_scope": "Overview/generation/action spans only; verification has no dedicated CUDA timer and can overlap another request span. Equal per-request allocation is incomplete/nonexclusive schedule accounting, not pure model time or marginal/billable GPU cost.",
                          "calibration": "No probabilities or calibration curves inferred from contrast."},
                  "Separate matched real recorded spans and exact synthetic quality/accounting comparisons. Synthetic intervals resample generator-case groups only. Verification can overlap recorded spans and lacks dedicated timing; CUDA accounting is incomplete/nonexclusive. Real risk–coverage remains unavailable; serial latency and concurrent throughput are distinct.")

    def gallery(self):
        if not self.selections:
            return
        from build_interactive_viewer import build_viewer
        selected_ids = {value["scene_id"] for value in self.selections.values()
                        if isinstance(value, dict) and "scene_id" in value}
        views = {sid: self.case_data(sid)["views"] for sid in sorted(selected_ids)}
        result = build_viewer(self.root, self.viewer, self.main, self.selections,
                              self.references_path, self.manifest_path, views_by_scene=views)
        self.use_source(Path(__file__).parent / "build_interactive_viewer.py")
        self.use_source(self.viewer / "case_manifest.json")
        print(json.dumps({"viewer": result}), flush=True)

    def run(self):
        if self.args.freeze_cases_only:
            print(json.dumps({"status": "case_policy_frozen", "path": str(self.policy_path)}))
            return 0
        functions = [(1, self.figure1), (2, lambda: self.acoustic_case(2)), (3, lambda: self.acoustic_case(3)),
                     (4, self.figure4), (5, self.figure5), (6, self.figure6), (7, self.figure7),
                     (8, self.figure8), (9, self.figure9)]
        for number, callback in functions:
            try:
                callback()
                print(json.dumps({"figure": number, "status": "created"}), flush=True)
            except UnavailableFigure as error:
                self.unavailable.append({"figure": number, "reason": str(error)})
                print(json.dumps({"figure": number, "status": "not_created", "reason": str(error)}), flush=True)
        self.gallery()
        manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "artifacts": self.artifacts,
                    "unavailable": self.unavailable, "code_sha256": digest(Path(__file__)),
                    "source_files": self.sources, "numeric_device": "cuda", "no_placeholder_figures": True}
        write_json(self.output / "manifest.json", manifest)
        return 0 if not self.unavailable or self.args.allow_partial else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--main-run", default="results/main")
    parser.add_argument("--pilot-run", default="results/pilot")
    parser.add_argument("--manifest", default="data/manifests/main_test.jsonl")
    parser.add_argument("--references", default="results/ai_annotations/primary_reference.jsonl")
    parser.add_argument("--output", default="figures")
    parser.add_argument("--viewer", default="viewer")
    parser.add_argument("--paper-dir", default="manuscript")
    parser.add_argument("--freeze-cases-only", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    require_cuda()
    raise SystemExit(Builder(args).run())


if __name__ == "__main__":
    main()
