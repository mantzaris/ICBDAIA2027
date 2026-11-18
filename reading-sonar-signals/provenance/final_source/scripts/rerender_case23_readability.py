#!/usr/bin/env python3
"""CPU-only layout amendment for saved Figures 2/3, without new measurements.

Loads the original CUDA-computed log-PSD tensors and display limits. All signal
values, report coordinates, case selection, references, and captions are retained.
The only intentional display changes are separated Figure 2 callouts and spaced
absolute original-time ticks in Figure 3. Never imports the scientific builder.
"""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "provenance/presentation_versions/case23_before_readability_fix"
REPORT = ROOT / "reports/case23_readability_rerender.json"
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
# Label positions are presentation coordinates, not acoustic measurements.
# IDs retain their saved report order; each leader ends at the unchanged track.
LABEL_POSITIONS = [(0.50, .36), (.51, .73), (.88, .73),
                   (.12, .73), (.32, .88), (.70, .88)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return str(Path(path).relative_to(ROOT))


def preserve(path):
    path = Path(path)
    target = BACKUP / path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        # Inputs must still be the exact originals on subsequent layout passes.
        if "inputs" in path.parts and sha(path) != sha(target):
            raise RuntimeError("Saved scientific input changed: " + rel(path))
    else:
        shutil.copy2(path, target)
    return {"path": rel(path), "preserved_path": rel(target),
            "original_sha256": sha(target)}


def cell_edges(centers):
    """Matplotlib nearest-shading cell boundaries, for older host Matplotlib.

    This only establishes rendered cell geometry from unchanged saved coordinate
    centers; it performs no resampling, signal transform, or statistical work.
    """
    half_steps = np.diff(centers) / 2
    return np.concatenate(([centers[0] - half_steps[0]],
                           centers[:-1] + half_steps,
                           [centers[-1] + half_steps[-1]]))


def load_display(payload, name, preserved, immutable):
    key = payload["selection"]["scene_id"] + ":" + name
    entry = payload["overview_display_scales"][key]
    path = ROOT / entry["display_tensor"]["path"]
    if sha(path) != entry["display_tensor"]["sha256"]:
        raise RuntimeError("Saved display tensor checksum mismatch: " + key)
    preserved.append(preserve(path))
    immutable[rel(path)] = sha(path)
    saved = torch.load(path, map_location="cpu")
    if saved["limits_db"] != entry["log_PSD_limits_db"]:
        raise RuntimeError("Saved display limits disagree: " + key)
    if saved["metadata"]["paper_scene_id"] != payload["selection"]["scene_id"]:
        raise RuntimeError("Saved display belongs to another scene")
    for name in ("log_PSD_db", "time_s", "frequency_hz"):
        if saved[name].device.type != "cpu":
            raise RuntimeError("Only host rendering is allowed")
    return saved


def draw_view(fig, axis, saved, title, events, number):
    times = saved["time_s"].numpy()
    frequencies = saved["frequency_hz"].numpy()
    limits = saved["limits_db"]
    metadata = saved["metadata"]
    mesh = axis.pcolormesh(cell_edges(times), cell_edges(frequencies),
                           saved["log_PSD_db"].numpy(), cmap="magma",
                           vmin=limits[0], vmax=limits[1], shading="flat",
                           rasterized=True)
    domain = metadata["frequency_domain"]
    carrier = metadata.get("carrier_band_hz")
    if carrier:
        title += "\nCarrier %g–%g Hz" % tuple(carrier)
    axis.set(title=title, xlabel="Original-recording time (s)",
             ylabel=("Modulation" if domain == "modulation" else "Acoustic") + " frequency (Hz)",
             xlim=(times[0], times[-1]), ylim=(frequencies[0], frequencies[-1]))
    if number == 3:
        # Both saved LOFAR and DEMON spans contain these absolute coordinates.
        axis.xaxis.set_major_locator(FixedLocator([62105, 62125]))
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    annotations = []
    for index, event in enumerate(events):
        if event.get("frequency_domain", "acoustic") != domain:
            continue
        if domain == "modulation" and event.get("carrier_band_hz") != carrier:
            continue
        # These frozen cases have only narrowband report components. Refuse
        # silent adaptation if a future input changes their geometry/schema.
        if event.get("type") not in ("tone", "chirp", "modulation"):
            raise RuntimeError("Unexpected geometry in focused case rerender")
        track = event.get("frequency_track")
        interval, band = event["time_s"], event["frequency_hz"]
        color = COLORS[index % len(COLORS)]
        if track:
            axis.plot([p[0] for p in track], [p[1] for p in track],
                      color=color, lw=1.3, alpha=.9)
            endpoint = track[-1]
        else:
            center = (band[0] + band[1]) / 2
            axis.plot(interval, [center, center], color=color, lw=1.3, alpha=.9)
            endpoint = [interval[-1], center]
        visible = (interval[1] >= times[0] and interval[0] <= times[-1]
                   and band[1] >= frequencies[0] and band[0] <= frequencies[-1])
        if visible:
            anchor = [min(max(endpoint[0], times[0]), times[-1]),
                      min(max(endpoint[1], frequencies[0]), frequencies[-1])]
            label = axis.annotate("M%d" % (index + 1), anchor,
                                 xytext=LABEL_POSITIONS[index], textcoords="axes fraction",
                                 ha="center", va="center", fontsize=10, color="#111827",
                                 bbox={"facecolor": "white", "alpha": .95,
                                       "edgecolor": color, "linewidth": .8, "pad": .7},
                                 arrowprops={"arrowstyle": "-", "color": color,
                                             "linewidth": .9, "shrinkA": 2, "shrinkB": 1})
            annotations.append(label)
    bar = fig.colorbar(mesh, ax=axis, label="PSD (dB)", shrink=.82, pad=.015)
    bar.ax.tick_params(labelsize=10)
    bar.ax.yaxis.label.set_fontsize(10)
    return annotations


def render(number, preserved, immutable):
    name = "figure%02d" % number
    input_path = ROOT / "figures/inputs" / (name + ".json")
    payload = json.loads(input_path.read_text())
    expected_sid = {2: "ds_0c5c23f3aefbdcae", 3: "mb_020dc567758f45b0"}[number]
    if payload["selection"]["scene_id"] != expected_sid:
        raise RuntimeError("Different frozen selected case")
    if payload["reference"].get("eligible_types") != [] or payload["reference"].get("events") != []:
        raise RuntimeError("Focused rerender expects the recorded unavailable reference")
    predicted = payload["prediction"]["components"]
    if [e["id"] for e in predicted] != [e["component_id"] for e in payload["component_callouts"]]:
        raise RuntimeError("Saved callout/report identity mismatch")
    if len(predicted) != {2: 6, 3: 0}[number]:
        raise RuntimeError("Different frozen component inventory")
    for path in [input_path, ROOT / payload["view_arrays"]["path"]]:
        preserved.append(preserve(path))
        immutable[rel(path)] = sha(path)
    if immutable[payload["view_arrays"]["path"]] != payload["view_arrays"]["sha256"]:
        raise RuntimeError("Original view array checksum mismatch")
    for suffix in ("pdf", "png", "svg"):
        preserved.append(preserve(ROOT / "figures" / (name + "." + suffix)))
    views = {key: load_display(payload, key, preserved, immutable)
             for key in ("lofar", "demon_low", "demon_high")}
    fig = plt.figure(figsize=(6.4, 6.6), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, .7])
    axes = [fig.add_subplot(grid[row, column]) for row in range(2) for column in range(2)]
    panels = [("lofar", "LOFAR: raw", []),
              ("lofar", "LOFAR: report proposals", predicted),
              ("demon_low", "Linked DEMON", predicted),
              ("demon_high", "Linked DEMON", predicted)]
    callout_annotations = []
    for axis, (view, title, events) in zip(axes, panels):
        callout_annotations.extend(draw_view(fig, axis, views[view], title, events, number))
    key_axis = fig.add_subplot(grid[2, :])
    key_axis.axis("off")
    for index, row in enumerate(payload["component_callouts"]):
        evidence = ", ".join(str(value) for value in row["evidence_ids"][:3]) or "none cited"
        if len(row["evidence_ids"]) > 3:
            evidence += " (+%d more)" % (len(row["evidence_ids"]) - 3)
        label = "%s = %s (%s)\nEvidence: %s" % (row["callout"], row["component_id"], row["type"], evidence)
        key_axis.text((index % 2) * .51, 1 - (index // 2) * .32,
                      label, va="top", fontsize=10, color="#111827")
    if not payload["component_callouts"]:
        key_axis.text(0, .8, "No retained report components; no callouts to draw.", fontsize=10)
    fig.suptitle("Real recording: reference criterion unmet\n%s (%s)\nNo qualified AI reference; model proposals only"
                 % (expected_sid, payload["selection"]["domain"]), fontsize=11)
    fig.legend(handles=[Line2D([], [], color=COLORS[1], label="F report proposals")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, -.02), fontsize=10)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [item.get_bbox_patch().get_window_extent(renderer) for item in callout_annotations]
    if any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:]):
        raise RuntimeError("Component callout labels still overlap")
    tick_checks = []
    for axis in axes:
        tick_boxes = [item.get_window_extent(renderer) for item in axis.get_xticklabels() if item.get_visible()]
        if any(a.overlaps(b) for i, a in enumerate(tick_boxes) for b in tick_boxes[i + 1:]):
            raise RuntimeError("Original-time tick labels still overlap")
        tick_checks.append({"ticks": [item.get_text() for item in axis.get_xticklabels()],
                            "nonoverlapping": True})
    width_in = fig.get_tightbbox(renderer).width + .2
    sizes = [item.get_fontsize() for item in fig.findobj(matplotlib.text.Text)
             if item.get_visible() and item.get_text().strip()]
    printed_font = min(sizes) * 4.8 / width_in
    if printed_font < 7:
        raise RuntimeError("External fonts fall below 7pt at 4.8in paper width")
    outputs = []
    for suffix in ("pdf", "png", "svg"):
        path = ROOT / "figures" / (name + "." + suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight",
                    metadata={"Creator": "Saved-array CPU presentation amendment"} if suffix == "pdf" else None)
        outputs.append({"path": rel(path), "sha256": sha(path)})
    plt.close(fig)
    return {"figure": number, "scene_id": expected_sid,
            "plot_inputs": rel(input_path), "plot_inputs_sha256": sha(input_path),
            "outputs": outputs, "callout_boxes_nonoverlapping": True,
            "callout_count": len(callout_annotations), "time_ticks": tick_checks,
            "paper_typography": {"paper_width_in": 4.8, "export_width_in": width_in,
                                 "minimum_external_printed_font_pt": printed_font},
            "caption_unchanged": payload["caption"]}


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 10, "axes.labelsize": 10,
                         "legend.fontsize": 10, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    immutable = {"scripts/make_figures.py": sha(ROOT / "scripts/make_figures.py")}
    preserved = [preserve(ROOT / "scripts/make_figures.py")]
    outputs = [render(number, preserved, immutable) for number in (2, 3)]
    if any(sha(ROOT / path) != digest for path, digest in immutable.items()):
        raise RuntimeError("Input/source mutated during presentation work")
    script_path = Path(__file__).resolve()
    report = {"status": "passed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "script": {"path": rel(script_path), "sha256": sha(script_path)},
              "scope": "CPU presentation only from previously CUDA-computed saved log PSD and display limits; no signal transforms, model calls, metrics, or new scientific calculations.",
              "changes": {"figure02": "Separate six report-order callouts with leaders to unchanged saved track endpoints.",
                          "figure03": "Two spaced absolute original-time ticks, 62105 and 62125 seconds, in every panel."},
              "originals": preserved, "immutable_inputs_and_source": immutable,
              "inputs_and_source_unchanged": True, "figure_manifest_edited": False,
              "figures": outputs}
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    shutil.copy2(script_path, BACKUP / script_path.name)
    print(json.dumps({"report": rel(REPORT), "sha256": sha(REPORT), "figures": outputs}, indent=2))


if __name__ == "__main__":
    main()
