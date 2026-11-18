#!/usr/bin/env python3
"""Fix saved vector presentation only; no scientific computation or GPU access.

Figure 6 keeps boundary markers visible. Figure 9 moves crowded budget labels
with colored leaders; data marks, lines, intervals and all input bytes remain
unchanged. The original SVGs are the reproducible layout inputs.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "provenance/presentation_versions/before_local_figures06_09"
SVG = "http://www.w3.org/2000/svg"
NS = {"s": SVG}
ET.register_namespace("", SVG)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    BACKUP.mkdir(parents=True, exist_ok=True)
    originals = []
    for name in ("figure06", "figure09"):
        for path in [ROOT / f"figures/{name}.{ext}" for ext in ("svg", "pdf", "png")] + [ROOT / f"figures/inputs/{name}.json"]:
            saved = BACKUP / path.name
            if not saved.exists():
                shutil.copy2(path, saved)
            originals.append({"path": str(saved.relative_to(ROOT)), "sha256": sha(saved)})
    saved_builder = BACKUP / "make_figures.py"
    if not saved_builder.exists():
        shutil.copy2(ROOT / "scripts/make_figures.py", saved_builder)
    changes = []
    outputs = []
    for name in ("figure06", "figure09"):
        tree = ET.parse(BACKUP / f"{name}.svg")
        root = tree.getroot()
        if name == "figure06":
            axes = root.find('.//s:g[@id="axes_1"]', NS)
            count = 0
            for group in axes:
                if not group.get("id", "").startswith("line2d_"):
                    continue
                for child in group:
                    if "clip-path" in child.attrib and any("fill-opacity: 0" in item.get("style", "") for item in child):
                        del child.attrib["clip-path"]
                        count += 1
            assert count == 5, count
            changes.append({"figure": 6, "change": "Removed clip only from five existing valid-schema marker groups; marker positions and all other vector content unchanged."})
        else:
            # SVG page coordinates are presentation positions, not metric values.
            positions = {
                "text_35": ("G4:6", 305, 62, "#d55e00"),
                "text_30": ("G3:2", 305, 74, "#009e73"),
                "text_31": ("G3:4", 305, 86, "#009e73"),
                "text_33": ("G4:2", 305, 116, "#d55e00"),
                "text_32": ("G3:6", 305, 128, "#009e73"),
                "text_34": ("G4:4", 305, 140, "#d55e00"),
            }
            for ident, (label, x, y, color) in positions.items():
                group = root.find(f'.//s:g[@id="{ident}"]', NS)
                text = group.find("s:text", NS)
                assert text.text == label[-1]
                old_x, old_y = float(text.get("x")), float(text.get("y"))
                text.set("x", str(x)); text.set("y", str(y)); text.attrib.pop("transform", None)
                text.set("style", f"font-size: 7px; font-family: 'DejaVu Sans'; text-anchor: start; fill: {color}")
                text.text = label
                group.insert(0, ET.Element(f"{{{SVG}}}path", {
                    "d": f"M {old_x - 3} {old_y + 3} L {x - 2} {y - 2}",
                    "style": f"fill: none; stroke: {color}; stroke-width: 0.5; stroke-opacity: 0.7"}))
            changes.append({"figure": 9, "change": "Moved six G3/G4 slot labels into separated callouts with method names and colored leaders. Data coordinates, intervals and accounting text unchanged."})
        output = ROOT / f"figures/{name}.svg"
        tree.write(output, encoding="utf-8", xml_declaration=True)
        subprocess.run(["rsvg-convert", "--format", "pdf", "--output", str(output.with_suffix(".pdf")), str(output)], check=True)
        subprocess.run(["rsvg-convert", "--format", "png", "--dpi-x", "300", "--dpi-y", "300", "--output", str(output.with_suffix(".png")), str(output)], check=True)
        for ext in ("svg", "pdf", "png"):
            path = output.with_suffix("." + ext)
            outputs.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path)})
        assert sha(ROOT / f"figures/inputs/{name}.json") == sha(BACKUP / f"{name}.json")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "rendered_pending_visual_review",
        "scope": "CPU vector-layout changes and rasterization only; no scientific calculations, input edits, model calls or remote access",
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha(Path(__file__)),
        "originals": originals,
        "original_builder_sha256": sha(saved_builder),
        "changes": changes,
        "outputs": outputs,
        "scientific_inputs_byte_identical": True,
    }
    (ROOT / "reports/figures06_09_local_presentation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "outputs": len(outputs), "inputs_unchanged": True}))


if __name__ == "__main__":
    main()
