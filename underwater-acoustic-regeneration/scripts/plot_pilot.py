"""Rebuild vector and high-resolution figures from completed development runs."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8,
                     "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"full": "#00725D", "no_probes": "#9467BD", "duration_only": "#D58318", "dispersion_only": "#2171B5"}


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUT / (name + "." + extension), dpi=450, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def workflow():
    fig, ax = plt.subplots(figsize=(4.7, 2.45))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    def box(x, y, width, label, color="#EDF3F7"):
        ax.add_patch(FancyBboxPatch((x, y), width, 0.23, boxstyle="round,pad=0.01,rounding_size=0.015",
                                   edgecolor="#526371", facecolor=color, linewidth=0.7))
        ax.text(x + width / 2, y + 0.115, label, ha="center", va="center", fontsize=7.5)
    def arrow(start, end, dashed=False):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, linewidth=0.7,
                                    linestyle="--" if dashed else "-", color="#526371"))
    box(0.01, 0.69, .17, "Observed\naudio + mask")
    box(.25, .69, .20, "4 candidates\n+ 2 context probes")
    box(.52, .69, .19, "Diagnostics\n+ candidate score")
    box(.79, .69, .19, "Accept or\nuse fallback B", "#E4F0E8")
    arrow((.19, .805), (.24, .805)); arrow((.46, .805), (.51, .805)); arrow((.72, .805), (.78, .805))
    box(.01, .27, .20, "Train generator\non G only")
    box(.27, .27, .20, "Choose B\non tuning V")
    box(.53, .27, .20, "Fit score on D\nFreeze on V")
    box(.79, .27, .19, "Dry calibration\non reserved V")
    arrow((.13, .51), (.31, .68), True); arrow((.63, .51), (.63, .68), True)
    arrow((.88, .51), (.88, .68), True)
    ax.text(.5, .095, "Exact final projection preserves every observed sample.\nAll repaired positions retain an imputation flag.", ha="center", va="center", fontsize=7.5)
    save(fig, "method")


def ranking(directory):
    curves = json.loads((directory / "ranking_curves.json").read_text())["rows"]
    fig, ax = plt.subplots(figsize=(4.7, 2.25))
    for name, label in (("full", "Full diagnostics"), ("no_probes", "No probes"),
                        ("duration_only", "Duration gate"), ("dispersion_only", "Dispersion gate")):
        rows = [row for row in curves if row["policy"] == name]
        ax.plot([row["coverage"] for row in rows], [row["conditional_harm"] for row in rows],
                color=COLORS[name], label=label, linewidth=1.1)
    ax.axhline(.10, color="#9C2525", linestyle="--", linewidth=.8, label="Target harm limit")
    ax.set(xlim=(0, 1), ylim=(0, 1.03), xlabel="Counterfactual acceptance fraction", ylabel="Harm among accepted cases")
    ax.text(.03, .19, "Calibrated policy:\nzero acceptance", fontsize=7, color="#555555")
    ax.grid(alpha=.18, linewidth=.5)
    ax.legend(loc="lower right", frameon=False, ncol=2, columnspacing=1.0)
    ax.set_title("Development ranking only; these curves are not calibrated policies", fontsize=7.3, pad=8)
    fig.tight_layout(pad=.7)
    save(fig, "development_ranking")


def examples(directory):
    cases = {row["case_id"]: row for row in map(json.loads, (directory / "cases.jsonl").read_text().splitlines())
             if row["pilot_role"] == "dev_test"}
    decisions = [row for row in map(json.loads, (directory / "dry_test_decisions.jsonl").read_text().splitlines()) if row["policy"] == "full"]
    with gzip.open(ROOT / "manifests/pilot_sources.jsonl.gz", "rt") as stream:
        sources = {row["original_file_id"]: row for row in map(json.loads, stream)}
    differences = {row["case_id"]: cases[row["case_id"]]["candidate_metrics"][row["selected_index"]]["Q"] -
                                  cases[row["case_id"]]["fallback_metrics"]["Q"] for row in decisions}
    ordered = sorted(decisions, key=lambda row: (differences[row["case_id"]], row["case_id"]))
    chosen = [ordered[0], ordered[-1]]
    evidence = []
    fig, axes = plt.subplots(2, 2, figsize=(4.7, 3.8), gridspec_kw={"width_ratios": [1.25, 1]})
    for index, decision in enumerate(chosen):
        case = cases[decision["case_id"]]
        reference = np.load(ROOT / sources[case["source_id"]]["processed_path"], allow_pickle=False)[:32768].astype(float)
        target = (reference - case["normalization_mean"]) / case["normalization_scale"]
        a, b = case["gap_start"], case["gap_end"]
        with np.load(ROOT / case["pool_path"], allow_pickle=False) as pool:
            baseline, selected = target.copy(), target.copy()
            baseline[a:b] = pool["fallback_gap"]
            selected[a:b] = pool["candidates_gap"][decision["selected_index"]]
        intervals = slice(a - 400, b + 400)
        time = (np.arange(len(target))[intervals] - a) / 8000 * 1000
        ax, spectrum = axes[index]
        ax.axvspan(0, (b - a) / 8, facecolor="#EEEEEE", zorder=0)
        for waveform, color, label, width in ((target, "#222222", "Reference", .5),
                                             (baseline, "#2171B5", "B = P", .7),
                                             (selected, "#D24C37", "C3", .6)):
            ax.plot(time, waveform[intervals], color=color, linewidth=width, label=label, alpha=.85)
            frames = np.lib.stride_tricks.sliding_window_view(waveform[a:b], 256)[::64]
            power = np.mean(np.abs(np.fft.rfft(frames * np.hanning(257)[:-1], axis=-1)) ** 2, axis=0)
            spectrum.plot(np.fft.rfftfreq(256, 1 / 8000) / 1000, 10 * np.log10(np.maximum(power, 1e-12)),
                          color=color, linewidth=.8, label=label)
        label = "Most favorable C3 case" if index == 0 else "Most harmful C3 case"
        ax.set_title(label + "\nΔQ = " + format(differences[decision["case_id"]], ".3f"), fontsize=7.4)
        ax.set(xlabel="Time from gap start (ms)", ylabel="Observed-RMS units")
        spectrum.set(xlabel="Frequency (kHz)", ylabel="Gap power (dB)", xlim=(0, 4))
        spectrum.grid(alpha=.15)
        evidence.append({"case_id": case["case_id"], "source_id": case["source_id"], "selection": label,
                         "C3_minus_B_Q": differences[decision["case_id"]], "source_partition": "V", "pilot_role": "dev_test"})
    axes[0, 1].legend(frameon=False, loc="best")
    fig.tight_layout(pad=.8, h_pad=1.0, w_pad=1.0)
    save(fig, "development_examples")
    (OUT / "example_selection.json").write_text(json.dumps({"rule": "Extremes of C3-minus-B Q in the dry development test; illustrative and explicitly selected using evaluation outcomes", "cases": evidence}, indent=2) + "\n")


def main():
    directory = ROOT / "results/pilot_pools_clipped_seed17"
    assert json.loads((directory / "pilot_summary.json").read_text())["stage"] == "development_only_pilot"
    workflow(); ranking(directory); examples(directory)
    print("Created three development-only figures in PDF, SVG, and PNG")


if __name__ == "__main__":
    main()
