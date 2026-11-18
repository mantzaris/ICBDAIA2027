#!/usr/bin/env python3
"""Render existing case and seed results, with bounded markers and panel legends.

This presentation revision reads previously computed CUDA scores. It does not
run experiments, recompute metrics, aggregate observations, or access RunPod.
"""
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "provenance/presentation_versions/figure06_before_user_review"
METHODS = ["G1", "G2", "G3", "G4", "F"]
COLORS = {"G1": "#64748B", "G2": "#A85291", "G3": "#009E73",
          "G4": "#D55E00", "F": "#0072B2", "P1": "#0072B2", "P2": "#D55E00"}
SOURCES = ["figures/inputs/figure06.json", "results/main/evaluation/per_episode.jsonl",
           "results/localization/predictions/evaluation/per_episode.jsonl",
           "results/perception/summary.json"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def preserve(path):
    target = BACKUP / path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(path, target)


def case_records(path, method):
    result = []
    for index, row in enumerate(read_rows(path)):
        if row["stratum"] != "synthetic" or row["method"] != method:
            continue
        assert row["execution_status"] == "completed"
        metrics = row["event_metrics"]
        assert metrics is not None
        result.append({"scene_id": row["scene_id"], "method": method,
                       "reference_event_count": row["reference_event_count"],
                       "f1": metrics["f1"], "false_positives": metrics["fp"],
                       "schema_failed": row["schema_failed"], "empty_report": row["empty_report"],
                       "plotted_as_recovery": row["reference_event_count"] > 0,
                       "source": str(path.relative_to(ROOT)), "jsonl_row_index": index,
                       "f1_pointer": "/event_metrics/f1"})
    assert len(result) == 12 and len([r for r in result if r["plotted_as_recovery"]]) == 10
    return sorted(result, key=lambda r: r["scene_id"])


def fraction_axis(axis, labels, xlabel):
    axis.set_xlim(-.06, 1.06)
    axis.set_xticks([0, .25, .5, .75, 1])
    axis.set_xticklabels(["0", ".25", ".5", ".75", "1"])
    axis.set_yticks(list(range(len(labels))))
    axis.set_yticklabels(labels)
    axis.set_ylim(len(labels) - .35, -.65)
    axis.set_xlabel(xlabel)
    axis.grid(axis="x", color="#E2E8F0", linewidth=.65)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def cases(axis, row_index, records):
    plotted = [r for r in records if r["plotted_as_recovery"]]
    # Fixed vertical layout offsets only. X always equals the saved case score.
    offsets = [-.02, .025, .07, .115, .16, .205, .25, .295, .34, .385]
    for record, offset in zip(plotted, offsets):
        axis.scatter([record["f1"]], [row_index + offset],
                     marker="x" if record["schema_failed"] else "o", s=14,
                     color="#B91C1C" if record["schema_failed"] else "#94A3B8",
                     linewidths=.65, alpha=.9, clip_on=True,
                     zorder=6 if record["schema_failed"] else 3)


def legend(axis, handles, labels, columns=1):
    axis.axis("off")
    axis.legend(handles, labels, loc="upper left", bbox_to_anchor=(0, 1),
                frameon=False, ncol=columns, borderaxespad=0,
                handletextpad=.35, columnspacing=.9, labelspacing=.2, fontsize=10)


def main():
    before = {p: digest(ROOT / p) for p in SOURCES}
    for path in [ROOT / f"figures/figure06.{ext}" for ext in ("pdf", "png", "svg")] + [
        ROOT / "figures/manifest.json", ROOT / "manuscript/main.tex", ROOT / "manuscript/main.pdf",
        ROOT / "manuscript/figure06.pdf", ROOT / "figures/inputs/figure06.json"]:
        preserve(path)
    payload = json.loads((ROOT / SOURCES[0]).read_text())
    controllers = {m: case_records(ROOT / SOURCES[1], m) for m in METHODS}
    localizers = {seed["method"]: case_records(ROOT / SOURCES[2], seed["method"])
                  for model in payload["localization"]["values"] for seed in model["seeds"]}
    perception = json.loads((ROOT / SOURCES[3]).read_text())
    classifier_points = [{"method": row["method"], "seed": row["seed"],
                          "f1": row["test_metrics"]["macro_f1_present_reference_classes"],
                          "pointer": f"/runs/{index}/test_metrics/macro_f1_present_reference_classes"}
                         for index, row in enumerate(perception["runs"])]
    assert len(classifier_points) == 6
    for model in payload["perception"]["values"]:
        assert [r["f1"] for r in classifier_points if r["method"] == model["method"]] == model["values"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 10, "axes.labelsize": 10,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(6.4, 5.9))
    outer = GridSpec(2, 2, figure=fig, left=.10, right=.985, top=.93, bottom=.035,
                     wspace=.46, hspace=.39)
    axes, legends = [], []
    for slot in outer:
        sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=slot, height_ratios=[1, .26], hspace=.48)
        axes.append(fig.add_subplot(sub[0])); legends.append(fig.add_subplot(sub[1]))

    a, b, c, d = axes
    fraction_axis(a, METHODS, "Fraction of 90 real clips")
    a.set_title("A  Real report production\nAccuracy unavailable", loc="left", pad=8)
    for index, point in enumerate(payload["real"]["operation_diagnostics"]):
        color = COLORS[point["method"]]
        a.plot(point["report_emission"], index, "o", color=color, ms=5, clip_on=True)
        a.plot(point["schema_valid_fraction"], index, "s", mfc="white", color=color, ms=5, clip_on=True)
    legend(legends[0], [Line2D([], [], color="#475569", marker="o", ls=""),
                        Line2D([], [], color="#475569", marker="s", mfc="white", ls="")],
           ["Nonempty report", "Valid schema"])

    fraction_axis(b, METHODS, "Exact synthetic event F1")
    b.set_title("B  Controller case scores\n10 supported cases / policy", loc="left", pad=8)
    for index, point in enumerate(payload["synthetic"]):
        cases(b, index, controllers[point["method"]])
        b.plot(point["ci95"], [index - .25] * 2, color=COLORS[point["method"]], lw=1.4, zorder=4)
        b.plot(point["f1"], index - .25, "D", color=COLORS[point["method"]], ms=4, zorder=5)
    legend(legends[1], [Line2D([], [], color="#94A3B8", marker="o", ls="", ms=4),
                        Line2D([], [], color="#475569", marker="D", lw=1, ms=4)],
           ["Case (red ×: invalid schema)", "Pooled score + 95% CI"])

    seed_labels = []
    seeds = []
    for model in payload["localization"]["values"]:
        for seed in model["seeds"]:
            seeds.append(seed)
            seed_labels.append(seed["method"].replace("_localizer_seed", " / "))
    fraction_axis(c, seed_labels, "Exact synthetic event F1")
    c.set_title("C  Localizers: cases by seed\n10 shared cases / seed", loc="left", pad=8)
    for index, seed in enumerate(seeds):
        cases(c, index, localizers[seed["method"]])
        c.plot(seed["f1"], index - .25, "D", color=COLORS[seed["method"][:2]], ms=4, zorder=5)
    legend(legends[2], [Line2D([], [], color="#94A3B8", marker="o", ls="", ms=4),
                        Line2D([], [], color="#475569", marker="D", ls="", ms=4)],
           ["Case (red ×: invalid schema)", "Pooled score for each seed"])

    d.set_title("D  Weak-label classification\nAll 3 training seeds shown", loc="left", pad=8)
    for index, model in enumerate(payload["perception"]["values"]):
        offsets = [-.22, -.07, .08]
        for offset, value in zip(offsets, model["values"]):
            d.scatter([index + offset], [value], s=22, color=COLORS[model["method"]], zorder=4)
        # Rendering a saved mean and saved SD; no statistic is recalculated.
        d.errorbar(index + .30, model["mean"], yerr=model["sample_sd"],
                   fmt="D", ms=4, color="#334155", capsize=3, lw=1.1, zorder=3)
    d.set_xlim(-.45, 1.55); d.set_ylim(-.04, .67)
    d.set_xticks([0, 1]); d.set_xticklabels(["P1", "P2"])
    d.set_yticks([0, .2, .4, .6]); d.set_ylabel("Present-class macro F1")
    d.set_xlabel("Training method")
    d.grid(axis="y", color="#E2E8F0", linewidth=.65); d.set_axisbelow(True)
    for side in ("top", "right"):
        d.spines[side].set_visible(False)
    legend(legends[3], [Line2D([], [], color="#475569", marker="o", ls="", ms=4),
                        Line2D([], [], color="#334155", marker="D", ls="", ms=4)],
           ["Individual seed", "Mean ± seed SD"])

    fig.canvas.draw()
    width = fig.get_tightbbox(fig.canvas.get_renderer()).width + .2
    assert 10 * 4.8 / width >= 7, (width, 10 * 4.8 / width)
    outputs = []
    for ext in ("pdf", "png", "svg"):
        path = ROOT / f"figures/figure06.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path)})
    plt.close(fig)
    assert before == {p: digest(ROOT / p) for p in SOURCES}
    expanded = {
        "artifact": "figure06_observations_revision", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": before, "original_aggregate_input_unchanged": True,
        "scientific_metrics_recomputed": False, "remote_calls": False,
        "controller_cases": controllers, "localizer_cases": localizers, "classifier_seeds": classifier_points,
        "case_support_rule": "Small recovery dots show only the ten positive-required-gold cases; noise and silence recovery is N/A. Their FP/schema/empty diagnostics remain here and their FP remain in existing pooled scores over all twelve controls.",
        "uncertainty_rule": "Controller bars are saved generator-case bootstrap 95% intervals. Classifier bars are saved training-seed SD, not confidence intervals. Localizer case/seed points share ten cases and are not independent replications.",
        "display_counts": {"real_summary_markers": 10, "controller_case_dots": 50,
                           "controller_pooled_diamonds": 5, "localizer_case_dots": 60,
                           "localizer_seed_pooled_diamonds": 6, "classifier_seed_dots": 6,
                           "classifier_mean_diamonds": 2},
        "paper_typography": {"minimum_source_font_pt": 10, "printed_width_in": 4.8,
                             "export_width_in": width, "minimum_printed_font_pt": 10 * 4.8 / width},
        "script": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": digest(__file__)},
        "outputs": outputs,
    }
    out = ROOT / "figures/inputs/figure06_observations_revision.json"
    out.write_text(json.dumps(expanded, indent=2) + "\n")
    print(json.dumps({"status": "rendered", "display_counts": expanded["display_counts"],
                      "min_printed_font_pt": expanded["paper_typography"]["minimum_printed_font_pt"]}))


if __name__ == "__main__":
    main()
