"""Host-only Figure 8 layout repair using the already exported CUDA values.

No experiment, acoustic transform, metric, aggregation or scientific arithmetic
is executed. Original renderings, the exact input JSON and builder are archived
before the canonical PDF/PNG/SVG files are replaced. The input JSON stays intact.
"""
from __future__ import annotations

from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def display_array(values):
    """Convert the saved display cells to renderer storage; never derive a score."""
    return np.ma.masked_invalid(np.asarray(
        [[float("nan") if value is None else value for value in row] for row in values],
        dtype=float))


def grid(axis, rows, columns):
    axis.set_xticks(np.arange(columns + 1) - .5, minor=True)
    axis.set_yticks(np.arange(rows + 1) - .5, minor=True)
    axis.grid(which="minor", color="white", linewidth=.65)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.tick_params(length=0, pad=4)
    for spine in axis.spines.values():
        spine.set_visible(False)


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "figures/inputs/figure08.json"
    builder = root / "scripts/make_figures.py"
    original_hash = digest(source)
    data = json.loads(source.read_text())
    if data["numeric_device"] != "cuda":
        raise ValueError("Requires a previously completed CUDA figure-input export")
    methods = list(data["methods"])
    if methods != ["F", "G4"]:
        raise ValueError("This layout is bound to the saved two-method Figure 8")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    archive = root / "provenance/presentation_versions" / ("figure08_before_layout_" + timestamp)
    archive.mkdir(parents=True, exist_ok=False)
    preserved = []
    originals = [root / ("figures/figure08." + extension) for extension in ("pdf", "png", "svg")]
    originals += [source, builder]
    for path in originals:
        target = archive / path.name
        shutil.copy2(path, target)
        if digest(path) != digest(target):
            raise RuntimeError("Preservation mismatch: " + str(path))
        preserved.append({"original_path": str(path.relative_to(root)),
                          "preserved_path": str(target.relative_to(root)),
                          "sha256": digest(path)})

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig = plt.figure(figsize=(7.4, 6.9))
    spec = fig.add_gridspec(2, 3, left=.225, right=.93, bottom=.16, top=.90,
                           width_ratios=[1, 1, .055], height_ratios=[1.75, 1],
                           wspace=.16, hspace=.50)
    axes = [[fig.add_subplot(spec[row, col]) for col in range(2)] for row in range(2)]
    top_bar, bottom_bar = fig.add_subplot(spec[0, 2]), fig.add_subplot(spec[1, 2])
    palette = copy.copy(plt.get_cmap("viridis"))
    palette.set_bad("#e5e7eb")
    labels = {"bandwidth loss": "Bandwidth loss", "blank": "Blank (valid empty)",
              "gain ×0.5": "Gain ×0.5", "plot style": "Plot style",
              "remove component": "Remove component"}
    condition_labels = [labels.get(name, name.replace("noise", "Noise")) for name in data["conditions"]]
    kind_labels = {"am_noise": "AM noise", "beating": "Beating", "chirp": "Chirp",
                   "mixed": "Mixed", "tones": "Tones", "transient": "Transient"}
    for col, method in enumerate(methods):
        axis = axes[0][col]
        values = data["methods"][method]
        top_mesh = axis.imshow(display_array(values["display_values"]), vmin=0, vmax=1,
                               cmap=palette, aspect="auto", interpolation="nearest")
        axis.set_title(method, pad=6)
        axis.set_xticks(range(len(data["synthetic_kinds"])))
        axis.set_xticklabels([kind_labels[k] for k in data["synthetic_kinds"]],
                             rotation=42, ha="right", rotation_mode="anchor")
        axis.set_yticks(range(len(data["conditions"])))
        axis.set_yticklabels(condition_labels if col == 0 else [""] * len(data["conditions"]))
        grid(axis, len(data["conditions"]), len(data["synthetic_kinds"]))
        for i, row in enumerate(values["cells"]):
            for j, cell in enumerate(row):
                if cell["status"] == "no_required_scoped_events":
                    label = ("N/A\nFP " + str(cell["scoped_false_positives"]) +
                             "\nS " + str(cell["schema_invalid_episodes"]))
                    axis.text(j, i, label, ha="center", va="center", fontsize=6.4,
                              linespacing=1.02, color="black")
                elif cell["status"] in ("not_sampled", "undefined_event_f1"):
                    label = "Not\nsampled" if cell["status"] == "not_sampled" else "Undefined"
                    axis.text(j, i, label, ha="center", va="center", fontsize=6.4)

        axis = axes[1][col]
        mixture = data["real_excerpt_mixtures"]["methods"][method]
        bottom_mesh = axis.imshow(display_array(mixture["event_agreement_f1"]), vmin=0, vmax=1,
                                  cmap=palette, aspect="auto", interpolation="nearest")
        axis.set_title(method + " · real-excerpt mixtures", pad=6, fontsize=9)
        axis.set_xticks(range(3))
        axis.set_xticklabels(data["real_excerpt_mixtures"]["first_to_second_RMS_power_ratio_db"])
        axis.set_yticks(range(5))
        axis.set_yticklabels(["0%", "25%", "50%", "75%", "100%"] if col == 0 else [""] * 5)
        if col == 0:
            axis.set_ylabel("Temporal overlap", labelpad=6)
        grid(axis, 5, 3)
        for i, row in enumerate(mixture["event_agreement_f1"]):
            for j, value in enumerate(row):
                if value is None:
                    label = "No eligible\nreference" if mixture["sampled_episodes"][i][j] else "Not\nsampled"
                else:
                    label = str(value)
                axis.text(j, i, label, ha="center", va="center", fontsize=6.8, linespacing=1.08)

    upper = fig.colorbar(top_mesh, cax=top_bar)
    upper.set_label("Score", labelpad=5)
    lower = fig.colorbar(bottom_mesh, cax=bottom_bar)
    lower.set_label("AI-reference F1", labelpad=5)
    upper.ax.tick_params(labelsize=8, length=2)
    lower.ax.tick_params(labelsize=8, length=2)
    fig.suptitle("Controlled sensitivity: synthetic probes and real mixtures", y=.965, fontsize=11)
    fig.text(.56, .105, "First/second excerpt power ratio (dB)", ha="center", fontsize=9)
    fig.text(.225, .052, "Top: scoped event F1; blank row: valid-empty fraction. Grey: unavailable.", fontsize=7.3)
    fig.text(.225, .026, "N/A: no required scoped events. FP: scoped false positives. S: schema-invalid reports.", fontsize=7.3)

    staged = archive / "new_renderings"
    staged.mkdir()
    outputs = []
    for extension in ("pdf", "png", "svg"):
        path = staged / ("figure08." + extension)
        fig.savefig(path, dpi=300, facecolor="white")
        outputs.append({"path": "figures/figure08." + extension, "sha256": digest(path),
                        "bytes": path.stat().st_size})
    plt.close(fig)
    if digest(source) != original_hash:
        raise RuntimeError("Scientific input changed during presentation work")
    for output in outputs:
        shutil.copy2(staged / Path(output["path"]).name, root / output["path"])
        if digest(root / output["path"]) != output["sha256"]:
            raise RuntimeError("Canonical rendering copy mismatch")
    report = {"status": "rendered_from_saved_values_visual_review_pending",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "script": str(Path(__file__).resolve().relative_to(root)), "script_sha256": digest(__file__),
              "source": str(source.relative_to(root)), "source_sha256_before_and_after": original_hash,
              "scientific_input_bytes_unchanged": True, "scientific_values_recomputed": False,
              "rendering_device": "CPU", "GPU_calls": False, "remote_calls": False,
              "original_archive_modified": False, "frozen_sources_modified": False,
              "preserved": preserved, "outputs": outputs,
              "data_policy": "Render saved display_values and saved mixture agreement cells only. No scoring, reduction, scientific transforms or aggregation. NumPy only adapts saved cells for Matplotlib.",
              "layout_changes": ["Shared row labels release width for cells.",
                  "N/A, FP and S use three short lines inside the cell.",
                  "Mixture unavailable labels wrap inside cell boundaries.",
                  "Short colorbar labels; full meanings appear in the footer.",
                  "Explicit panel spacing, white cell boundaries and horizontal footer notes."],
              "manifest_update": "Parent coordinates final figure-manifest/provenance updates across presentation repairs."}
    report["preservation_history"] = [
        {"directory": str(folder.relative_to(root)),
         "files": [{"path": str(path.relative_to(root)), "sha256": digest(path)}
                   for path in sorted(folder.iterdir()) if path.is_file()]}
        for folder in sorted(archive.parent.glob("figure08_before_layout_*"))]
    report["development_attempts"] = [
        "First host render encountered an older Matplotlib without Colormap.copy; originals were preserved and no canonical outputs changed.",
        "Second host render completed all outputs but a relative __file__ path interrupted report writing; its original copies and staged renderings remain preserved.",
        "Final render uses compatible layout calls and resolves the script path before recording provenance."]
    report_path = root / "reports/figure08_saved_value_rerender.json"
    if report_path.exists():
        shutil.copy2(report_path, archive / report_path.name)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(report_path.relative_to(root)), "sha256": digest(report_path),
                      "input_sha256": original_hash, "outputs": outputs}))


if __name__ == "__main__":
    main()
