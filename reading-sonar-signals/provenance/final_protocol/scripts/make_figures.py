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


class UnavailableFigure(RuntimeError):
    pass


def tensor(values, dtype=torch.float64):
    return torch.tensor(values, device="cuda", dtype=dtype)


def relative(path, root):
    return os.path.relpath(Path(path).resolve(), Path(root).resolve())


def get_events(reference):
    if not reference:
        return []
    annotation = reference.get("annotation", reference)
    return [event for event in annotation.get("events", []) if event_in_reference_scope(event, annotation)]


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
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                             "axes.titlesize": 9, "axes.labelsize": 9,
                             "legend.fontsize": 7, "xtick.labelsize": 8, "ytick.labelsize": 8,
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
        payload = dict(payload, artifact=name, caption=caption,
                       numeric_device="cuda", rendering="Matplotlib on host",
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

    def draw_view(self, axis, view, title, reference=(), prediction=(), overlap=False):
        power = view.log_power()
        limits = torch.quantile(power.flatten(), tensor([.05, .995], dtype=power.dtype)).tolist()
        mesh = axis.pcolormesh(view.times_s.cpu().numpy(), view.frequencies_hz.cpu().numpy(),
                               power.cpu().numpy(), cmap="magma", vmin=limits[0], vmax=limits[1],
                               shading="nearest", rasterized=True)
        domain = view.metadata["frequency_domain"]
        carrier = view.metadata.get("carrier_band_hz")
        if carrier:
            title += f"\nCarrier {carrier[0]:g}–{carrier[1]:g} Hz"
        axis.set(title=title, xlabel="Original-recording time (s)",
                 ylabel=("Modulation" if domain == "modulation" else "Acoustic") + " frequency (Hz)")
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
                if track:
                    coordinates = tensor(track)
                    axis.plot(coordinates[:, 0].cpu(), coordinates[:, 1].cpu(), style, color=color, lw=width, alpha=alpha)
                elif event.get("type") in ("tone", "chirp", "modulation"):
                    center = tensor(frequencies).mean().item()
                    axis.plot(interval, [center, center], style, color=color, lw=width, alpha=alpha)
                else:
                    span = tensor([interval, frequencies])
                    sizes = (span[:, 1] - span[:, 0]).tolist()
                    axis.add_patch(Rectangle((interval[0], frequencies[0]), sizes[0], sizes[1],
                                             fill=False, edgecolor=color, linestyle=style, linewidth=width))
                if style == "--" and event.get("frequency_uncertainty_hz", 0) and event.get("type") in ("tone", "chirp", "modulation"):
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

    def component_legend(self, fig):
        fig.legend(handles=[Line2D([], [], color="#0891B2", linestyle="--", label="AI reference (uncertain)"),
                            Line2D([], [], color=COLORS[1], label="F report components")],
                   loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, -.02))

    def figure1(self):
        chosen = self.select_cases()["figure04"]
        episode = self.episode(chosen["scene_id"])
        actions = episode.get("actions", [])
        if not actions:
            raise UnavailableFigure("No recorded action trace for method diagram")
        fig, axes = plt.subplots(2, 1, figsize=(7.4, 3.5), gridspec_kw={"height_ratios": [1, 1.1]}, constrained_layout=True)
        axes[0].set(xlim=(0, 10), ylim=(0, 2)); axes[0].axis("off")
        labels = ["Waveform", "CUDA views\nLOFAR + DEMON", "Evidence\ngraph", "Inspection\ncontroller", "Verifier +\nreport"]
        for index, label in enumerate(labels):
            x = index * 2
            axes[0].add_patch(FancyBboxPatch((x + .05, .65), 1.7, .7, boxstyle="round,pad=.08", facecolor="#EAF2F8", edgecolor="#44647A"))
            axes[0].text(x + .9, 1, label, ha="center", va="center", fontsize=9)
            if index < 4:
                axes[0].annotate("", xy=(x + 2.03, 1), xytext=(x + 1.85, 1), arrowprops={"arrowstyle": "->", "color": "#44647A"})
        axes[0].annotate("", xy=(2.8, .55), xytext=(6.9, .55), arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=-.25", "color": COLORS[1]})
        axes[0].text(4.9, .04, "Bounded reinspection of the original waveform", ha="center", fontsize=8)
        axes[0].set_title("Conceptual computation path", loc="left")
        axes[1].axis("off")
        selected = actions[:4]
        table_rows = [[str(a.get("step", i)), a.get("action", {}).get("name", "invalid action"),
                       a.get("status", "unknown"), textwrap.fill(str(a.get("reason", ""))[:110], width=48)] for i, a in enumerate(selected)]
        table = axes[1].table(cellText=table_rows, colLabels=["Step", "Recorded operation", "Outcome", "Recorded purpose"],
                              cellLoc="left", colWidths=[.07, .26, .16, .51], loc="center")
        table.auto_set_font_size(False); table.set_fontsize(7.4); table.scale(1, 2.3)
        axes[1].set_title(f"Actual F trace: {chosen['scene_id']}", loc="left")
        self.save(1, fig, {"scene_id": chosen["scene_id"], "actions": selected},
                  "Conceptual method and an actual held-out action trace; failed calls remain visible.")

    def acoustic_case(self, number):
        chosen = self.select_cases()[f"figure{number:02d}"]
        data = self.case_data(chosen["scene_id"])
        reference = get_events(data["reference"])
        predicted, _, _ = report_events(data["episode"].get("report"))
        panels = [("lofar", "Raw LOFAR", [], []), ("lofar", "Reference / model overlay", reference, predicted)]
        for name in ("demon_low", "demon_high"):
            if name in data["views"]:
                panels.append((name, "Linked DEMON", reference, predicted))
        fig, axes = plt.subplots(2, 2, figsize=(7.5, 4.9), constrained_layout=True)
        for axis, (name, title, refs, preds) in zip(axes.flat, panels):
            mesh = self.draw_view(axis, data["views"][name], title, refs, preds, overlap=number == 3)
            fig.colorbar(mesh, ax=axis, label="dB re digital amplitude²/Hz", shrink=.82, pad=.015)
        for axis in list(axes.flat)[len(panels):]:
            fig.delaxes(axis)
        title = "Tonal morphology" if number == 2 and chosen["criterion_met"] else "Co-occurring components" if number == 3 and chosen["criterion_met"] else "Real scene; requested reference criterion absent"
        fig.suptitle(f"{title} · {chosen['scene_id']} ({chosen['domain']})\nAI-reference agreement only; physical source associations remain unknown", fontsize=10)
        self.component_legend(fig)
        self.save(number, fig, {"selection": chosen, "reference": data["reference"],
                               "prediction": data["episode"].get("report"), "view_arrays": data["view_arrays"]},
                  f"Recorded real-scene displays for {chosen['scene_id']}; dashed AI references are eligibility-limited and not independent physical truth. Other morphologies remain unscored. " +
                  ("Temporal coexistence does not establish multiple physical sources." if number == 3 else "Frequencies and carrier bands retain their distinct meanings."))

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
        fig, axes = plt.subplots(1, len(images) + 1, figsize=(9.2, 3.1), constrained_layout=True)
        self.draw_view(axes[0], data["views"]["lofar"], "Global LOFAR", get_events(data["reference"]),
                       report_events(data["episode"].get("report"))[0])
        for axis, path, entry in zip(axes[1:], images, selected):
            # Reading an existing raster is rendering only; the recorded tool
            # image preserves the exact crop, window and evidence identifier.
            axis.imshow(plt.imread(path)); axis.axis("off")
            action = entry.get("action", {})
            window = action.get("window_s", 4. if action.get("name") == "inspect_modulation" else 1.)
            axis.set_title(f"Step {entry.get('step')}: {action.get('name')}\nRecorded tool view; {window:g}s window")
        fig.suptitle(f"Actual adaptive trace · {chosen['scene_id']} · original coordinates retained", fontsize=10)
        fig.text(.5, -.045, f"F: {len(actions)} attempted calls; G3: {len(fixed.get('actions', []))} attempted calls. Recorded inspection is not proof of an accuracy benefit.", ha="center", fontsize=8)
        self.save(4, fig, {"selection": chosen, "selected_actions": selected,
                          "fixed_G3_report": fixed.get("report"), "F_report": data["episode"].get("report"),
                          "recorded_action_images": [{"path": relative(path, self.root), "sha256": digest(path)} for path in images],
                          "view_arrays": data["view_arrays"]},
                  "Global evidence and exact recorded F inspection images, with original coordinates and evidence IDs retained. The fixed-inspection report is preserved in the plotting inputs.")

    def figure5(self):
        chosen = self.select_cases()["figure05"]
        data = self.case_data(chosen["scene_id"])
        episode = data["episode"]
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw={"width_ratios": [1.4, 1]}, constrained_layout=True)
        self.draw_view(axes[0], data["views"]["lofar"], "Recorded evidence", get_events(data["reference"]), report_events(episode.get("report"))[0])
        axes[1].axis("off")
        audit = episode.get("verification_audit", [])
        withheld = [a for a in audit if a.get("status") == "withheld"]
        lines = [chosen["purpose"], "", f"Scene: {chosen['scene_id']}",
                 f"Raw components: {len(report_events(episode.get('raw_report'))[0])}",
                 f"Retained components: {len(report_events(episode.get('report'))[0])}",
                 f"Verifier withheld entries: {len(withheld)}", ""]
        if episode.get("failure"):
            lines.append("Execution failure: " + str(episode["failure"].get("error", "unknown"))[:160])
        for row in withheld[:2]:
            lines.append(str(row.get("component_id", "component")) + ": " + "; ".join(row.get("reasons", [])))
        report = episode.get("report") or {}
        lines.extend(report.get("unresolved_questions", [])[:2])
        wrapped = "\n".join(textwrap.fill(line, width=46) if line else "" for line in lines)
        axes[1].text(0, 1, wrapped, va="top", fontsize=8.2, linespacing=1.35)
        fig.suptitle("Failure / abstention record; internal checks do not establish correctness", fontsize=10)
        self.save(5, fig, {"selection": chosen, "raw_report": episode.get("raw_report"),
                          "report": episode.get("report"), "verification_audit": audit,
                          "view_arrays": data["view_arrays"]},
                  "A declared failure or disagreement case, showing preserved evidence and withheld or unresolved findings without claiming verified physical interpretation.")

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
                 title="Real primary agreement unavailable\nReport-production diagnostics only")
        axis.invert_yaxis(); axis.grid(axis="x", color="#E5E7EB")
        axis.legend(handles=[Line2D([], [], marker="o", ls="", color="#475569", label="Nonempty report"),
                             Line2D([], [], marker="s", ls="", mfc="none", color="#475569", label="Valid schema")],
                    loc="best", fontsize=6.5)
        return points

    def figure6(self):
        summary = self.summary("main")
        available = [s for s in ("real", "synthetic") if s in summary["results"]]
        if not available:
            raise UnavailableFigure("Main quantitative strata absent")
        perception_path = self.root / "results/perception/summary.json"
        perception = read_json(perception_path) if perception_path.exists() else None
        count = len(available) + int(perception is not None)
        fig, axes = plt.subplots(1, count, figsize=(3.0 * count, 3.2), constrained_layout=True)
        axes = [axes] if count == 1 else list(axes)
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
                    "Real: eligible AI-reference agreement" if stratum == "real" else "Controlled: exact synthetic truth")
            payload[stratum + "_reference_scopes"] = {m: v.get("reference_scopes") for m, v in summary["results"][stratum]["all"].items()}
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
            axis.set(xticks=[0, 1], xticklabels=["P1: LOFAR", "P2: dual"], ylim=(0, 1),
                     ylabel="Present-reference-class macro F1", title="Secondary weak-label classification\nMean ± SD across three training seeds")
            payload["perception"] = {"values": values, "test_class_support": perception.get("test_class_support"),
                                      "scope": "No localization score; no generalization claim for test-absent classes."}
        self.save(6, fig, payload, "Exact synthetic outcomes and weak-label classification are separate tasks. When AI qualification fails, real primary agreement is unavailable; observed emission and schema fractions describe execution rather than accuracy.")

    def figure7(self):
        summary = self.summary("ablations")
        stratum = "real" if "real" in summary["results"] else next(iter(summary["results"]))
        results = summary["results"][stratum]["all"]
        methods = sorted(results, key=lambda name: (name != "F", name))
        agreement_available = any(value.get("event_metrics") is not None for value in results.values())
        columns = ["Event F1", "Overlap F1", "Ref. coverage", "Report emission"] if agreement_available else ["Report emission", "Valid schema"]
        rows, cost, counts = [], [], []
        for method in methods:
            value = results[method]
            event, overlap = value.get("event_metrics"), value.get("geometric_overlap_metrics")
            if agreement_available:
                rows.append([event["f1"] if event else float("nan"), overlap["f1"] if overlap else float("nan"),
                             value.get("reference_event_coverage") if value.get("reference_event_coverage") is not None else float("nan"),
                             value["report_emission_coverage"]])
            else:
                schema_valid = (1 - tensor(value["totals"]["schema_failed"]) / tensor(value["expected_episodes"])).item()
                rows.append([value["report_emission_coverage"], schema_valid])
            cost.append(value["compute"]["cuda_event_seconds"]["mean_per_expected_episode"])
            counts.append(value["reference_scored_episodes"])
        values = tensor(rows)
        if not bool(torch.isfinite(values).any()):
            raise UnavailableFigure("Ablation metrics are undefined")
        fig, axes = plt.subplots(1, 2, figsize=(7, max(2.6, .36 * len(methods))),
                                 gridspec_kw={"width_ratios": [3, 1]}, constrained_layout=True)
        mesh = axes[0].imshow(values.cpu().numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axes[0].set(yticks=list(range(len(methods))), yticklabels=methods,
                    xticks=list(range(len(columns))), xticklabels=columns)
        for i, row in enumerate(values.tolist()):
            for j, value in enumerate(row):
                finite = bool(torch.isfinite(values[i, j]))
                axes[0].text(j, i, f"{value:.2f}" if finite else "undefined", ha="center", va="center",
                             color="white" if finite and value < .55 else "black", fontsize=7)
        positions = torch.arange(len(methods), device="cuda")
        axes[1].barh(positions.cpu(), tensor(cost).cpu(), color=COLORS[0]); axes[1].invert_yaxis()
        axes[1].set(yticks=list(range(len(methods))), yticklabels=[""] * len(methods), xlabel="Accounted CUDA s/episode")
        fig.colorbar(mesh, ax=axes[0], shrink=.6, label="Score")
        fig.suptitle(f"Frozen ablation subset · {stratum}" + (" · AI agreement if real" if agreement_available else "\nPrimary agreement unavailable; execution diagnostics and cost only"), fontsize=10)
        self.save(7, fig, {"stratum": stratum, "methods": methods,
                          "columns": columns, "primary_agreement_available": agreement_available,
                          "metrics": [[None if not bool(torch.isfinite(x)) else x.item() for x in row] for row in values],
                          "cuda_seconds_per_episode": cost, "reference_scored_episodes": counts,
                          "comparison": "Raw scores on one ablation cohort, not deltas against a different main cohort."},
                  "Ablations on the declared common subset. With no qualified AI-reference classes, report production, schema validity, and cost remain measurable, while primary agreement stays unavailable.")

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

    def budget_points(self):
        phases = [p for p in ("budget2", "budget4", "main") if (self.phase(p) / "evaluation/per_episode.jsonl").exists()]
        sets = {phase: {r["scene_id"]: r for r in self.rows(phase) if r["method"] == "F" and r["stratum"] == "real"
                        and r.get("event_metrics") is not None and r.get("reference_valid_for_primary")} for phase in phases}
        if len(sets) < 2:
            return []
        common = set.intersection(*[set(values) for values in sets.values()])
        if not common:
            return []
        points = []
        for phase, values in sets.items():
            selected = [values[sid] for sid in sorted(common)]
            timings = tensor([[r["timing"][k] for k in ("preprocess_gpu_seconds", "model_gpu_seconds", "tool_gpu_seconds")] for r in selected])
            config = read_json(self.phase(phase) / "run.json")
            self.use_source(self.phase(phase) / "run.json")
            points.append({"phase": phase, "budget": config["budget"],
                           "event_f1": aggregate_event_metrics([r["event_metrics"] for r in selected])["f1"],
                           "mean_accounted_cuda_seconds": timings.sum(dim=1).mean().item(),
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
                                          "mean_wall_seconds": wall.mean().item(), "scope": "Pilot cohort and serial execution only"}
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
        risk, budgets, accounting = self.risk_curve(), self.budget_points(), self.execution_accounting()
        count = int(risk is not None) + int(bool(budgets)) + int(bool(accounting))
        if not count:
            raise UnavailableFigure("No rankable verified components, matched budget results, or completed execution-accounting record")
        fig, axes = plt.subplots(1, count, figsize=(3.1 * count, 3.2), constrained_layout=True)
        axes = [axes] if count == 1 else list(axes)
        panel = 0
        if risk:
            axes[panel].plot(risk["coverage"], risk["risk"], color=COLORS[0], marker=".", markersize=3)
            axes[panel].set(xlabel="Retained fraction of rankable components", ylabel="Unmatched-to-AI-reference fraction",
                            xlim=(0, 1), ylim=(0, 1), title="Uncalibrated contrast ranking")
            panel += 1
        if budgets:
            for point in budgets:
                axes[panel].plot(point["mean_accounted_cuda_seconds"], point["event_f1"], "o", color=COLORS[0])
                axes[panel].annotate(f" {point['budget']} calls", (point["mean_accounted_cuda_seconds"], point["event_f1"]), fontsize=8)
            axes[panel].set(xlabel="Accounted CUDA s/episode", ylabel="Real AI-reference event F1", ylim=(0, 1),
                            title=f"Matched budget cohort (n={len(budgets[0]['common_scene_ids'])})")
            panel += 1
        if accounting:
            axes[panel].axis("off")
            lines = ["Execution quantities have distinct units", ""]
            if "serial_pilot" in accounting:
                entry = accounting["serial_pilot"]
                lines += [f"Serial pilot: {entry['episodes']} episodes", f"Median latency: {entry['median_wall_seconds']:.1f} s/episode", ""]
            if "main_last_invocation" in accounting:
                entry = accounting["main_last_invocation"]
                lines += [f"Main run: {entry['workers']} workers", f"Observed throughput: {entry['episodes_per_minute']:.2f} episodes/min",
                          f"New completed episodes: {entry['new_completed_episodes']}", "", "Amortized CUDA time is not serial latency."]
            axes[panel].text(0, .95, "\n".join(lines), va="top", fontsize=9, linespacing=1.5)
        self.save(9, fig, {"contrast_ranking": risk, "matched_budget_points": budgets,
                          "execution_accounting": accounting, "calibration": "No probabilities or calibration curves inferred from contrast."},
                  "Reference-agreement risk under an uncalibrated measurement ranking and measured efficiency. Serial latency and concurrent throughput are reported as different quantities.")

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
