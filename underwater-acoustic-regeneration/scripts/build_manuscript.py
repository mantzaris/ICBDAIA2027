"""Build an anonymous, evidence-linked development manuscript in the venue template.

No submission is performed. Completed diagnostics are required for a final build;
--allow-pending permits a clearly marked layout preview while jobs are running.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript"
RUN = ROOT / "results/pilot_pools_clipped_seed17"


def read(path):
    return json.loads((ROOT / path).read_text())


def content(allow_pending=False):
    pilot = read("results/pilot_pools_clipped_seed17/pilot_summary.json")
    assert pilot["stage"] == "development_only_pilot" and not pilot["actual_test_opened"]
    assert not pilot["oracle_feasibility_passed"]
    diagnostics_path = ROOT / "results/pilot_pools_clipped_seed17_diagnostics/summary.json"
    transfer_path = ROOT / "results/transfer/summary.json"
    diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else None
    transfer = json.loads(transfer_path.read_text()) if transfer_path.exists() else None
    if not allow_pending and (diagnostics is None or transfer is None):
        raise ValueError("Completed diagnostic and transfer reports required; use --allow-pending only for layout")
    if diagnostics:
        assert diagnostics["status"] == "completed" and diagnostics["completed_cases"] == 800
    policies = pilot["policies"]
    pages = []
    def p(text, **kwargs):
        return dict(kind="paragraph", text=text, **kwargs)
    def h(text):
        return dict(kind="heading", text=text)
    def f(name, caption, width=12.0):
        return dict(kind="figure", name=name, caption=caption, width_cm=width)
    def t(caption, headers, rows, evidence):
        return dict(kind="table", caption=caption, headers=headers, rows=rows, evidence=evidence)

    abstract = (
        "Waveform inpainting can produce plausible content without recovering the event that actually occurred. "
        "We examine whether observation-only probes and a separately calibrated gate can support selective "
        "completion of missing underwater acoustic samples. This paper reports a bounded development audit, "
        "not a completed held-out efficacy study. We audit 43,208 UniqueShip recordings, construct vessel/source "
        "partitions, and train a 2.72-million-parameter conditional diffusion model on approximately two hours "
        "of generator-training audio. Four candidate completions and two context probes feed a predictor of "
        "quality change relative to a validation-selected deterministic fallback. A numerical failure at the "
        "cosine schedule endpoint motivates an explicitly documented, training-only sampler amendment. On "
        "64 tuning groups, only five have a candidate improving the prespecified quality score by at least "
        "0.02; both 32-step and 64-step sampling fail the 20% feasibility trigger. Dry calibration on 64 "
        "separate development groups selects no accepting threshold. The resulting policy equals its fallback "
        "on all 64 dry-test groups. Paired hidden-event constructions additionally test the impossibility of "
        "identifying content absent from the observation. These findings support a no-go decision for this "
        "pilot configuration, while leaving generalization, downstream utility, and the selective-method "
        "hypothesis unresolved."
    )
    assert 150 <= len(abstract.split()) <= 250
    pages.append([
        dict(kind="title", text="Selective Underwater Waveform Inpainting:\nA Development Feasibility Audit"),
        dict(kind="author", text="Anonymous submission"),
        dict(kind="abstract", text="Abstract. " + abstract),
        p("Keywords: underwater acoustics · waveform inpainting · selective prediction · negative results", style="p1a", size=9),
        h("1  Problem and scope"),
        p("Known missing samples arise from acoustic acquisition or transmission failures. Filling them is distinct "
          "from denoising: an undeleted field recording is a reconstruction reference, not noise-free physical truth. "
          "Our question is whether diagnostics from observed context can identify useful generative completions "
          "at a fixed inference budget. The proposed decision mechanism is evaluated only as a feasibility pilot. "
          "A failed predeclared opportunity check stops the main study; it cannot establish that diffusion or "
          "selective restoration is ineffective in general."),
        p("This anonymous development draft documents actual measurements, failure analysis, and reproducible "
          "artifacts. The real calibration and test outcomes remain unopened. The planned three-seed study, "
          "independent utility evaluation, and external archive experiment are not represented by pilot results.")
    ])
    pages.append([
        h("2  Related work and data audit"),
        p("UniqueShip addresses leakage in acoustic ship classification [1]. Underwater inpainting itself is "
          "already studied: Tembo et al. report mask-conditioned diffusion for underwater spectrograms [2]. "
          "Its deposited bibliographic metadata were verified, but primary full text was unavailable; we make "
          "no detailed architectural comparison or priority claim. General-audio diffusion inpainting [3], "
          "DiffWave residual blocks [4], and observed-value conditioning in CSDI [5] provide established components. "
          "Inpainting uncertainty [6], risk calibration [7], and similarity-guided music inpainting [8] further "
          "preclude claims that diagnostics, uncertainty, or context guidance are intrinsically new."),
        p("The official UniqueShip v1 twelve-class, five-hours-per-class archive contains 43,208 mono PCM32 "
          "WAVs at 32 kHz, totaling 60.008 hours. Actual headers differ from the website's 20 kHz description. "
          "Twelve clips shorter than 4.096 s are ineligible for fixed-length evaluation. We preserve the first "
          "official test split, retain validation where feasible, and allocate remaining vessel/source groups "
          "to generator training G, diagnostic fitting D, and calibration C. Table 1 reports the complete manifest.",
          evidence=["manifests/audit.json", "reports/data_summary.json"]),
        t("Table 1. Audited source partitions; counts describe available data, not evaluated outcomes.",
          ["Partition", "Sources", "Groups", "Pilot use"],
          [["G", "26,321", "2,996", "1,440 sources"], ["D", "3,962", "467", "200 groups"],
           ["V", "4,445", "400", "200 groups"], ["C", "4,003", "547", "Unopened"],
           ["Test", "4,477", "610", "Unopened"]], ["reports/data_summary.json", "manifests/audit.json"]),
        p("All descendants stay within a group; background uses hydrophone/date, pooling unknown devices by "
          "date. Decoded and normalized-quantized hashes found no duplicates. Timestamp checks found no "
          "cross-partition overlaps or intervals within 60 s, including unknown-device comparisons. However, "
          "1,332 of 1,522 dates span partitions, and original session/parent identifiers are unavailable. "
          "Vessel separation therefore does not prove environmental independence; exhaustive time-shifted "
          "near-duplicate detection was not performed."),
        p("Class support is uneven: V and D lack pilot-vessel examples; C also lacks dredger; test lacks dredger. "
          "The held-out taxonomy consequently has eleven supported classes, not twelve. No classification score "
          "is reported. Intact sources are resampled to 8 kHz before 32,768-sample crops and artificial masks; "
          "the reconstruction scope is restricted to frequencies below 4 kHz.")
    ])
    pages.append([
        h("3  Selective completion protocol"),
        p("Let m mark observed samples and y = m ⊙ x. The inference interface erases hidden coordinates before "
          "any normalization or feature computation. Mean and centered RMS use observed values only; an RMS "
          "floor is fixed on G. Every reverse step projects observed coordinates, and final inverse scaling "
          "copies original observations exactly:"),
        dict(kind="equation", text="Πᵧ(z) = m ⊙ y + (1 − m) ⊙ z.                                      (1)"),
        p("Primary gaps are 50, 100, 250, or 500 ms with at least 0.75 s context on each side. Training uses "
          "one gap 80% of the time and two separated gaps otherwise. Four independently seeded completions "
          "and two same-duration context probes use 32 DDIM steps each [9]: 192 denoiser evaluations. Each "
          "probe hides its samples before recomputing normalization. Reference waveforms and labels never "
          "enter inference."),
        f("method.png", "Fig. 1. Development workflow. Dry calibration uses reserved V groups; actual C/test "
          "outcomes are unopened. Repaired locations remain explicitly flagged as imputed."),
        p("The deterministic fallback B minimizes tuning-V mean Q among learned restoration and bidirectional "
          "AR orders 16, 32, 64, and 128. A histogram gradient-boosting regressor learns candidate ΔQ from "
          "18 observation-only features on 200 D groups, weighted equally. Features include candidate "
          "disagreement, differences from B, boundaries, and context-probe errors. Select the candidate with "
          "lowest predicted ΔQ. Duration-only and dispersion-only gates retain this same candidate selector, "
          "so their comparisons isolate acceptance rather than candidate changes."),
        p("The all-fallback output is a valid completion policy, but zero generative coverage supplies no "
          "evidence that the gate can identify useful generated content. The proof obligations for calibrated "
          "selection and waveform fidelity remain separate.")
    ])
    pages.append([
        h("4  Training, numerics, and decision rule"),
        p("The conditional model adapts DiffWave [4] with 64 channels, 36 residual layers, dilations 1–2048 "
          "repeated three times, 2,717,697 parameters, and a 3.071 s receptive field. It predicts noise on missing "
          "samples using a 1,000-step cosine schedule [10]. One seed (17), batch 16, and 5,000 updates use "
          "1,440 G sources, approximately two hours of original audio. A same-backbone deterministic model "
          "uses masked MAE plus 0.5 times multiresolution spectral convergence for 2,500 updates. Training "
          "takes 29.3 and 17.9 minutes on one RTX 4090; peak allocated memory is 2.56 GiB for each."),
        p("A 16-source, reduced-width learning diagnostic lowers finite noise-prediction loss, but 2,000 updates "
          "do not establish near-zero memorization. Under-optimization remains plausible. Checkpoint restoration "
          "does reproduce the next CUDA batch, RNG state, loss, and parameters exactly. The metric smoke test "
          "covers 200 G crops and six classical outputs per crop with finite metrics and zero observed drift."),
        p("Unbounded DDIM from the last cosine timestep fails on four G-only cases: ᾱ₉₉₉ = 2.43×10⁻⁹ amplifies "
          "noise-prediction error, producing gap NMSE in the millions. Clipping clean predictions alone keeps "
          "all four Q values saturated at one. Before candidate-V outcomes, a recorded amendment fixes the "
          "start to t = 934, the last timestep with ᾱ ≥ 0.01, and clips normalized clean predictions at ±5.897, "
          "a G-derived bound. Noise is recomputed after clipping; observed coordinates are never clipped. "
          "All inference budgets use this same endpoint."),
        p("This amendment is a numerical pilot adaptation. Gaussian initialization at a nonterminal marginal "
          "introduces a train–inference mismatch; it is not a principled zero-terminal-SNR or v-prediction "
          "solution and does not imply exact posterior sampling. Schedule-mismatch concerns [11] limit the "
          "interpretation. Original failures and the amendment are retained, rather than silently replacing them."),
        dict(kind="equation", text="Q = ½ min(NMSEgap / 4, 1) + ½ min(SCgap / 2, 1).                  (2)"),
        p("SC averages gap-only spectral convergence at FFT sizes 64, 128, and 256, periodic Hann windows, "
          "quarter-window hops, and no centered padding. Smaller Q is better; harm means ΔQ = Qcandidate − QB "
          "> 0.02. V freezes at most nine nonpositive score thresholds. For n accepted calibration cases and e "
          "harms, a one-sided Clopper–Pearson upper bound uses level 1 − 0.05/M across M thresholds. A threshold "
          "is eligible only if the bound is ≤0.10; n = 0 gives bound one. If none qualifies, use B everywhere. "
          "The interpretation requires independent, identically distributed group draws and a frozen policy [7], "
          "not arbitrary ocean or site shift.")
    ])
    rows = []
    for key, name, calls in [("B", "Deterministic B", "0"), ("C1", "One draw (C1)", "32"),
                             ("C2", "Four-draw medoid (C2)", "128"), ("C3", "Select, always accept (C3)", "192"),
                             ("full", "Calibrated policy P", "192")]:
        value = policies[key]
        harm = "—" if key == "B" else format(value["all_case_harm_rate"], ".3f")
        rows.append([name, format(value["mean_Q"], ".3f"), format(value["mean_gap_nmse"], ".3f"),
                     format(value["generative_coverage"], ".2f"), harm, calls])
    pages.append([
        h("5  Development results and no-go decision"),
        p("Of 200 V groups, 64 tune B and thresholds, 64 dry-calibrate, 64 dry-test, and eight are unused. "
          "The first 32 tuning groups also monitor training. B is deterministic restoration (mean tuning Q "
          "0.218 versus 0.229 for AR128). Only 5/64 tuning cases (7.8%) contain any candidate improving Q by "
          "at least 0.02, below the frozen 20% feasibility trigger. Increasing to 64 steps leaves 5/64 and "
          "raises mean six-draw latency from 1.28 to 2.48 s. We retain 32 steps and stop the main study."),
        t("Table 2. One case per dry-test V group (n = 64). Harm is an all-case rate relative to B. "
          "Calls count diffusion evaluations; B still has one learned restoration pass.",
          ["Method", "Q ↓", "NMSE ↓", "Coverage", "Harm", "Calls"], rows,
          ["results/pilot_pools_clipped_seed17/pilot_summary.json"]),
        p("All four gate variants freeze only threshold zero, accept 0/64 dry-calibration cases, and fail "
          "the bound (U = 1). Their final threshold is absent: P equals B. Conditional harm among accepted "
          "cases is undefined, not zero. C3 harms 52/64 dry-test cases. P's zero harm is structural fallback, "
          "not a successful risk–coverage tradeoff."),
        f("development_ranking.png", "Fig. 2. Descriptive dry-test score rankings at equal hypothetical coverage. "
          "These counterfactual positive-score acceptances were not used by the calibrated policy and supply "
          "no calibrated guarantee or confirmatory diagnostic comparison.")
    ])
    synthetic_text = (
        "The frozen policy is additionally evaluated on 100 realizations each of tonal continuation, chirp, "
        "onset, offset, colored background, and two paired hidden-event targets (700 cases). Mixtures vary "
        "tones, modulation, SNR, and short echoes; they are controlled signals, not validated ocean propagation. "
        "Each absent/present pair has exactly identical observed samples and uses the same random seeds. "
        + ("All 400 method–pair comparisons produce identical outputs. Each method must therefore make "
           "either a false-event or missed-event decision at the fixed event-projection threshold. " if diagnostics else
           "[LAYOUT PREVIEW: final paired-input check is pending.] ")
        + "This is an identifiability limit: fallback and uncertainty cannot reveal invisible content."
    )
    pages.append([
        h("6  Failure examples and identifiability"),
        f("development_examples.png", "Fig. 3. Explicitly selected extremes of C3-minus-B Q among the 64 dry-test "
          "cases, not representative averages. Gray marks the gap. Spectra use the missing interval only; "
          "B and P coincide because every generative completion is rejected."),
        p("Mean gap NMSE is 0.713 for B/P and 1.460 for C3; full-window NMSE falls to 0.038 and 0.074 "
          "because unchanged context dilutes error. C3's 99th-percentile gap NMSE is 6.944 versus 2.596 for B. "
          "All measured observed-sample drift is zero. A 2,000-draw paired group bootstrap gives descriptive "
          "P-minus-C3 mean Q −0.110 (95% interval −0.130 to −0.089). A date-balanced sensitivity over 63 "
          "dates gives −0.112 (−0.132 to −0.091), a different estimand. Neither is a primary held-out test."),
        p(synthetic_text)
    ])
    if transfer:
        if transfer["completed_cases"]:
            transfer_text = (
                "A separately isolated compatibility run uses Moliner and Välimäki's published general-audio "
                "checkpoint [3] on " + str(transfer["completed_cases"]) + " frozen V groups without fine-tuning. "
                "The pinned implementation and EMA weights retain their 22.05 kHz, 184,184-sample configuration. "
                "Only already-masked observations are resampled; the mask expands by 12 input samples and "
                "unknown padding is excluded from guidance. Inverse resampling is followed by exact observed "
                "projection. The completed run has mean Q " + format(transfer["mean_Q"], ".3f") + " and gap SC "
                + format(transfer["mean_gap_SC"], ".3f") + ". This small domain check is neither an exact benchmark "
                "reproduction nor evidence of superiority over a general-audio model. No adaptation is undertaken "
                "after the pilot no-go decision."
            )
        else:
            transfer_text = ("The published general-audio checkpoint [3] was retrieved and hash-verified, but the "
                             "compatibility run completed no scored case. Its recorded failure is an implementation "
                             "compatibility limit, not evidence of restoration performance. No transfer quality "
                             "score or superiority claim is made.")
    else:
        transfer_text = "[LAYOUT PREVIEW: the published general-audio checkpoint compatibility run is pending; no score is claimed.]"
    diagnostics_rows = []
    if diagnostics:
        for key, name in [("gap_750ms", "750 ms gap"), ("gap_1000ms", "1,000 ms gap"),
                          ("packet_4x50ms", "Four 50 ms gaps"), ("insufficient_context", "Short context"),
                          ("smooth_level_spectrum_shift", "Level/spectrum shift")]:
            # Preserve the exact logged mechanism identifiers in the evidence map.
            item = diagnostics["mechanisms"].get(key)
            if item is not None:
                diagnostics_rows.append([name, str(item["cases"]), format(item["coverage"], ".2f"), format(item["mean_P_Q"], ".3f")])
        if len(diagnostics_rows) != 5:
            diagnostics_rows = [[key.replace("_", " "), str(item["cases"]), format(item["coverage"], ".2f"), format(item["mean_P_Q"], ".3f")]
                                for key, item in diagnostics["mechanisms"].items() if item["cases"] == 20]
    else:
        diagnostics_rows = [[name, "Pending", "—", "—"] for name in
                            ["750 ms gap", "1,000 ms gap", "Four 50 ms gaps", "Short context", "Level/spectrum shift"]]
    pages.append([
        h("7  Shift checks, transfer, and limitations"),
        t("Table 3. Frozen-policy shifts on 20 V sources per condition. P = B; zero coverage "
          "does not validate a transferred risk bound.", ["Condition", "Cases", "Coverage", "Q(P)"],
          diagnostics_rows, ["results/pilot_pools_clipped_seed17_diagnostics/summary.json"]),
        p("Shift tests change gap duration, packet geometry, context, or smooth level/spectrum. Multi-gap and "
          "insufficient-probe-context inputs use a predetermined fallback. No model, score threshold, or "
          "normalization rule is adapted from these outcomes. A zero-acceptance policy is expected to fall "
          "back under every condition; this check establishes behavior, not useful out-of-distribution selection."),
        p(transfer_text),
        p("The results concern one short training run and one seed. Learning remains incomplete, the amended "
          "sampler is approximate, the V subsets are small, and source grouping cannot establish independent "
          "ocean conditions. The planned six-draw medoid (C4), one 192-step draw (C5), waveform RePaint "
          "adaptation (C6) [12], independent classifier, and VLIZ external evaluation were not run. Their rows "
          "are not imputed. Tonal-track and onset structure metrics were not pre-frozen and are not claimed."),
        p("Consequently the five main criteria—held-out coverage ≥25%, relative harm reduction ≥25%, better "
          "diagnostic ranking, Q/classifier noninferiority, and held-out observation consistency—remain "
          "untested as a conjunction. Zero drift passes the development implementation check only. Generic "
          "risk control, average-fidelity superiority, downstream benefit, and cross-archive generalization "
          "are unsupported. A future study needs a separately justified optimization plan and fresh frozen "
          "evaluation, rather than treating this pilot as a successful main experiment.")
    ])
    refs = [
        "Hashemi, C., Stout, T., Hoogs, A., Parham, J.: UniqueShip: Mitigating Data Leakage in Acoustic Ship Classification Benchmark Datasets. arXiv:2609.13659 (2026).",
        "Tembo, D.Z.Q., et al.: Mask-Conditioned Diffusion for Deep-Fade-Aware Inpainting of Underwater Acoustic Spectrograms. ICIGP (2026). doi:10.1109/ICIGP68997.2026.11619767.",
        "Moliner, E., Välimäki, V.: Diffusion-Based Audio Inpainting. J. Audio Eng. Soc. 72(3), 100–113 (2024). doi:10.17743/jaes.2022.0129.",
        "Kong, Z., Ping, W., Huang, J., Zhao, K., Catanzaro, B.: DiffWave: A Versatile Diffusion Model for Audio Synthesis. ICLR (2021). arXiv:2009.09761.",
        "Tashiro, Y., Song, J., Song, Y., Ermon, S.: CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation. NeurIPS (2021). arXiv:2107.03502.",
        "Cohen, K., Wolf, L., Laufer-Goldshtein, B.: Discovering Directions of Uncertainty in Speech Inpainting. Interspeech (2025). doi:10.21437/Interspeech.2025-835.",
        "Angelopoulos, A.N., Bates, S., Candès, E.J., Jordan, M.I., Lei, L.: Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control. arXiv:2110.01052v5 (2022).",
        "Turland, S., Moliner, E., Välimäki, V.: Music Inpainting With Similarity-Guided Latent Diffusion. arXiv:2611.11522v1 (2026).",
        "Song, J., Meng, C., Ermon, S.: Denoising Diffusion Implicit Models. ICLR (2021). arXiv:2010.02502.",
        "Nichol, A.Q., Dhariwal, P.: Improved Denoising Diffusion Probabilistic Models. ICML, PMLR 139, 8162–8171 (2021).",
        "Lin, S., Liu, B., Li, J., Yang, X.: Common Diffusion Noise Schedules and Sample Steps are Flawed. arXiv:2305.08891v4 (2024).",
        "Lugmayr, A., Danelljan, M., Romero, A., Yu, F., Timofte, R., Van Gool, L.: RePaint: Inpainting Using Denoising Diffusion Probabilistic Models. CVPR (2022)."
    ]
    pages.append([
        h("8  Conclusion and reproducibility"),
        p("This pilot does not support a selective underwater-regeneration claim: oracle opportunity is 7.8%, "
          "and dry calibration yields zero acceptance. The practical result is a documented no-go decision, "
          "not a successful generator. Code, source/group hashes, frozen masks and seeds, failed numerical "
          "traces, checkpoints, recording-level metrics, and figure scripts are preserved. Data attribution "
          "and third-party licenses accompany the local artifact package. A single existing RTX 4090 was "
          "used within the user-authorized ten-hour run. No public repository or manuscript submission is "
          "asserted; release and a complete main study remain future work."),
        h("References"),
        *[dict(kind="reference", text=str(index) + ". " + ref) for index, ref in enumerate(refs, 1)]
    ])
    return pages, {"stage": "anonymous_development_draft", "submission_ready": False,
                   "main_study_complete": False, "abstract_words": len(abstract.split()),
                   "layout_preview": diagnostics is None or transfer is None,
                   "literature_search_through": "2026-11-15"}


def disable_numbering(paragraph):
    properties = paragraph._p.get_or_add_pPr()
    numbering = OxmlElement("w:numPr")
    number = OxmlElement("w:numId")
    number.set(qn("w:val"), "0")
    numbering.append(number)
    properties.append(numbering)


def clear_template(document):
    body = document._element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)
    for section in document.sections:
        for tag in ("w:headerReference", "w:footerReference"):
            for child in list(section._sectPr.findall(qn(tag))):
                section._sectPr.remove(child)
    # Remove unused sample assets, identities, comments, and old chart links.
    remove_types = {"header", "footer", "image", "chart", "comments", "customXml", "footnotes", "endnotes", "hyperlink"}
    for key, relationship in list(document.part.rels.items()):
        if relationship.reltype.rsplit("/", 1)[-1] in remove_types:
            del document.part.rels[key]
    props = document.core_properties
    props.author = "Anonymous"
    props.last_modified_by = "Anonymous"
    props.title = "Selective Underwater Waveform Inpainting: A Development Feasibility Audit"
    props.subject = "Anonymous development draft; main study incomplete"
    props.comments = ""
    props.keywords = "underwater acoustics, waveform inpainting, feasibility audit"
    props.category = "Research development draft"
    props.identifier = ""
    props.created = datetime(2026, 11, 15, tzinfo=timezone.utc)
    props.modified = datetime(2026, 11, 15, tzinfo=timezone.utc)
    props.revision = 1


def build(pages, metadata):
    document = Document(OUT / "template/conference_template_macro_free.docx")
    clear_template(document)
    evidence = []
    for page_number, blocks in enumerate(pages, 1):
        if page_number > 1:
            # Break before the first content paragraph, avoiding blank trailing pages.
            page_start = True
        else:
            page_start = False
        for block_index, block in enumerate(blocks):
            kind = block["kind"]
            if kind == "table":
                caption = document.add_paragraph(block["caption"], style="tablecaption")
                caption.paragraph_format.keep_with_next = True
                table = document.add_table(rows=1, cols=len(block["headers"]))
                table.autofit = False
                # Widths fit the template's 120 mm text area.
                widths = ([4.9, 1.1, 1.3, 1.6, 1.4, 1.7] if len(block["headers"]) == 6 else [4.4, 2.1, 2.1, 3.4])
                for col, width in zip(table.columns, widths):
                    col.width = Cm(width)
                for index, text in enumerate(block["headers"]):
                    table.rows[0].cells[index].text = text
                for row in block["rows"]:
                    for index, text in enumerate(row):
                        if index == 0:
                            cells = table.add_row().cells
                        cells[index].text = text
                for row_index, row in enumerate(table.rows):
                    row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
                    for cell_index, cell in enumerate(row.cells):
                        cell.width = Cm(widths[cell_index])
                        for paragraph in cell.paragraphs:
                            paragraph.paragraph_format.space_after = Pt(3)
                            paragraph.paragraph_format.space_before = Pt(3)
                            paragraph.paragraph_format.first_line_indent = Pt(0)
                            paragraph.paragraph_format.line_spacing = Pt(10)
                            for run in paragraph.runs:
                                run.font.size = Pt(8)
                                run.bold = row_index == 0
                        if row_index == 0:
                            shading = OxmlElement("w:shd")
                            shading.set(qn("w:fill"), "EDF2F4")
                            cell._tc.get_or_add_tcPr().append(shading)
                document.add_paragraph().paragraph_format.space_after = Pt(0)
            elif kind == "figure":
                paragraph = document.add_paragraph()
                paragraph.paragraph_format.space_before = Pt(3)
                paragraph.paragraph_format.space_after = Pt(2)
                paragraph.paragraph_format.first_line_indent = Pt(0)
                paragraph.paragraph_format.keep_with_next = True
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                paragraph.add_run().add_picture(str(ROOT / "figures" / block["name"]), width=Cm(block["width_cm"]))
                document.add_paragraph(block["caption"], style="figurecaption")
            else:
                styles = {"heading": "heading1", "title": "papertitle", "author": "author",
                          "abstract": "abstract", "reference": "referenceitem", "equation": "p1a"}
                paragraph = document.add_paragraph(block["text"], style=block.get("style", styles.get(kind, "p1a")))
                if page_start:
                    paragraph.paragraph_format.page_break_before = True
                    page_start = False
                if kind in {"heading", "reference"}:
                    disable_numbering(paragraph)
                if kind == "heading":
                    paragraph.paragraph_format.keep_with_next = True
                    if block_index == 0:
                        paragraph.paragraph_format.space_before = Pt(0)
                elif kind == "paragraph":
                    paragraph.paragraph_format.space_after = Pt(5)
                    paragraph.paragraph_format.first_line_indent = Pt(0)
                elif kind == "reference":
                    paragraph.paragraph_format.space_after = Pt(4)
                    paragraph.paragraph_format.left_indent = Cm(.45)
                    paragraph.paragraph_format.first_line_indent = Cm(-.45)
                    paragraph.paragraph_format.keep_together = True
                elif kind == "equation":
                    paragraph.paragraph_format.space_before = Pt(5)
                    paragraph.paragraph_format.space_after = Pt(7)
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                if "size" in block:
                    for run in paragraph.runs:
                        run.font.size = Pt(block["size"])
            if block.get("evidence"):
                evidence.append({"planned_page": page_number, "block": block_index, "sources": block["evidence"]})
    # Add neutral page numbers; never carry original template author headers.
    for section in document.sections:
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        field = OxmlElement("w:fldSimple")
        field.set(qn("w:instr"), "PAGE")
        footer._p.append(field)
        for run in footer.runs:
            run.font.size = Pt(9)
    target = OUT / "underwater_feasibility_anonymous.docx"
    document.save(target)
    metadata.update(docx_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    planned_pages=len(pages), source_template="template/conference_template_macro_free.docx",
                    table_evidence=evidence, author="Anonymous", macros_executed=False)
    (OUT / "render_plan.json").write_text(json.dumps({"metadata": metadata, "pages": pages}, ensure_ascii=False, indent=2) + "\n")
    (OUT / "build_report.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()
    authors = read("manuscript/author_info.json")
    assert authors["anonymous"] and authors["display_name"] == "Anonymous submission"
    pages, metadata = content(args.allow_pending)
    build(pages, metadata)


if __name__ == "__main__":
    main()
