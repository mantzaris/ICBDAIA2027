#!/usr/bin/env python3
"""CPU rendering proof of case layouts from existing development PNGs only.

This is not a benchmark or final figure generator. It imports selected layout
methods through AST without importing torch, and substitutes previously exported
CUDA rasters for the live spectral renderer. No audio, transforms, model calls,
scores, or uncertainty estimates are calculated.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/make_figures.py"
OUTPUT = ROOT / "reports/presentation_layout_proof"
SID = "ds_0117fa3395bdaa51"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def report_events(report):
    # Presentation metadata only. These are saved development reports, not a
    # replacement for the common evaluator or a new validation pass.
    return ((report.get("components", []) if isinstance(report, dict) else []), [], [])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tree = ast.parse(SOURCE.read_text())
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name in ("component_callouts", "component_description")]
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Builder")
    chosen = [node for node in original.body if isinstance(node, ast.FunctionDef)
              and node.name in ("callout_key", "component_legend", "figure1", "acoustic_case", "figure4", "figure5")]
    layout = ast.ClassDef(name="Layout", bases=[], keywords=[], body=chosen, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=helpers + [layout], type_ignores=[]))
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    namespace = dict(plt=plt, textwrap=textwrap, Line2D=Line2D, FancyBboxPatch=FancyBboxPatch, COLORS=colors,
                     METHOD_COLORS={"F": colors[0], "G3": colors[2]},
                     CASE_WIDTH_IN=6.4, CASE_FONT_PT=10, UnavailableFigure=RuntimeError,
                     get_events=lambda value: [], report_events=report_events, digest=digest,
                     relative=lambda path, root: str(Path(path).relative_to(root)))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 10, "axes.labelsize": 10,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    sources, outputs = {str(SOURCE.relative_to(ROOT)): digest(SOURCE)}, []

    class Proof(namespace["Layout"]):
        root = ROOT
        main = ROOT / "results/pilot_v3"

        def use_source(self, path):
            sources[str(Path(path).relative_to(ROOT))] = digest(path)

        def select_cases(self):
            return {f"figure{n:02d}": {"scene_id": SID, "domain": "DeepShip development",
                                      "criterion_met": False, "purpose": "DEVELOPMENT replay; not final case selection"}
                    for n in (2, 3, 4, 5)}

        def episode(self, sid, method="F"):
            path = self.main / sid / method / "episode.json"
            self.use_source(path)
            return read_json(path)

        def case_data(self, sid):
            folder = ROOT / "viewer_development/assets" / sid
            views = {}
            for name in ("lofar", "demon_low", "demon_high"):
                path = folder / f"{name}_view.json"
                self.use_source(path)
                views[name] = read_json(path)
            return {"views": views, "episode": self.episode(sid), "reference": {},
                    "view_arrays": {"scope": "already saved CUDA raster exports; no tensors recalculated"}}

        def draw_view(self, axis, view, title, reference=(), prediction=(), **kwargs):
            assert not reference and not prediction, "Proof scene has empty saved F reports; do not invent overlays"
            raster = view["rasters"]["raw"]
            path = ROOT / "viewer_development/assets" / SID / Path(raster["path"]).name
            assert digest(path) == raster["sha256"]
            self.use_source(path)
            axis.imshow(plt.imread(str(path)), origin="upper", interpolation="nearest", aspect="auto",
                        extent=view["time_range_s"] + view["frequency_range_hz"])
            carrier = view.get("carrier_band_hz")
            if carrier:
                title += f"\nCarrier {carrier[0]:g}–{carrier[1]:g} Hz"
            axis.set(title=title, xlabel="Original-recording time (s)",
                     ylabel=("Modulation" if carrier else "Acoustic") + " frequency (Hz)")
            # The color scale is metadata from the original GPU export. This
            # does not recalculate or recolor the saved PNG.
            scalar = ScalarMappable(norm=Normalize(*raster["limits_db"]), cmap="magma")
            scalar.set_array([])
            return scalar

        def save(self, number, figure, payload, caption):
            title = figure._suptitle.get_text() if figure._suptitle else "Recorded diagram and trace prefix"
            figure.suptitle("DEVELOPMENT LAYOUT PROOF\n" + title, fontsize=11)
            figure._suptitle.set_color("#9F1239")
            name = f"development_figure{number:02d}"
            for suffix in ("pdf", "png"):
                target = OUTPUT / f"{name}.{suffix}"
                figure.savefig(target, dpi=180, bbox_inches="tight")
                outputs.append({"path": str(target.relative_to(ROOT)), "sha256": digest(target)})
            (OUTPUT / f"{name}.json").write_text(json.dumps({
                "status": "development_layout_only_not_main_results", "caption": caption,
                "payload": payload, "sources": dict(sources)}, indent=2) + "\n")
            plt.close(figure)

    proof = Proof()
    proof.figure1()
    proof.acoustic_case(2)
    proof.figure4()
    proof.figure5()
    # Exercise the actual callout typography using an existing nonempty G1
    # report. Its components are not overlaid on another scene's waveform.
    callout_path = ROOT / "results/pilot_v3/ds_070c32d03728f5ea/G1/episode.json"
    proof.use_source(callout_path)
    actual = read_json(callout_path)["report"]["components"]
    fig, axis = plt.subplots(figsize=(6.4, 2.4), constrained_layout=True)
    proof.callout_key(axis, actual)
    fig.suptitle("DEVELOPMENT G1 CALLOUT TYPOGRAPHY\nds_070c32d03728f5ea; saved proposals, not truth", fontsize=10)
    fig.savefig(OUTPUT / "development_callout_key.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "development_callout_key.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    latex = r"""\documentclass{article}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\pagestyle{empty}
\begin{document}
"""
    for name in ("development_figure01", "development_figure02", "development_figure04", "development_figure05", "development_callout_key"):
        latex += r"\noindent\includegraphics[width=4.8in]{" + name + ".pdf}\\clearpage\n"
    latex += "\\end{document}\n"
    (OUTPUT / "paper_width_proof.tex").write_text(latex)
    compilation = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "paper_width_proof.tex"],
                                 cwd=OUTPUT, text=True, capture_output=True, check=True)
    (OUTPUT / "compile.log").write_text(compilation.stdout + compilation.stderr)
    # These are rendering dimensions/font scaling, not experimental metrics.
    typography = []
    for name in ("development_figure01", "development_figure02", "development_figure04", "development_figure05", "development_callout_key"):
        path = OUTPUT / f"{name}.pdf"
        info = subprocess.run(["pdfinfo", str(path)], text=True, capture_output=True, check=True).stdout
        width = float(re.search(r"Page size:\s+([0-9.]+) x", info).group(1))
        printed_font = 10 * (4.8 * 72) / width
        assert printed_font >= 7, "External text fell below the paper-scale font target"
        typography.append({"path": str(path.relative_to(ROOT)), "source_pdf_width_pt": width,
                           "external_10pt_text_at_4p8inch_width_pt": printed_font})
    (OUTPUT / "receipt.json").write_text(json.dumps({
        "status": "development_layout_proof_only", "scientific_computation": "none",
        "GPU_use": False, "rendering": "CPU composition of existing hash-verified CUDA-exported PNGs",
        "target_paper_width_in": 4.8, "source_width_in": 6.4,
        "external_text_source_pt": 10, "nominal_external_text_at_paper_width_pt": 7.5,
        "paper_width_verification": typography,
        "compiled_paper_width_proof": {"path": str((OUTPUT / "paper_width_proof.pdf").relative_to(ROOT)),
                                       "sha256": digest(OUTPUT / "paper_width_proof.pdf")},
        "limitations": "Saved action rasters retain their original embedded text; readable external coordinates/evidence labels are added. Final cases and geometry require the actual final GPU render.",
        "sources": sources, "outputs": outputs}, indent=2) + "\n")


if __name__ == "__main__":
    main()
