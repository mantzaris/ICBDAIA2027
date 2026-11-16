# Diagnosing generative underwater acoustic repair

The targeted follow-up **diagnoses and corrects a numerical failure** in generative
underwater audio repair. A matched velocity-prediction model removes the observed
terminal-step blow-up: full-precision gap NMSE on four diagnostic sources falls
from 5.18 million to 1.03. On 64 reserved development cases, first-draw error Q
falls from 0.412 to 0.308, about 25%; the learned baseline remains better at 0.239.
The anonymous paper presents the mechanism, three new equations, controls, and
before/after spectrograms. These are exploratory results on reused development data.

The original development pilot **failed its predeclared feasibility trigger**:
only 5 of 64 tuning groups (7.8%) had a diffusion candidate improving Q by at least
0.02, versus the required 20%. The calibrated development policy accepts no
generated completions and equals its deterministic fallback. This is not evidence
of a successful selective method. The correction raises the tuning opportunity to
7/64 (10.9%), still below 20%; no new release gate is fitted. Actual calibration/test
outcomes remain unopened; the full research study is incomplete.

The user authorized an existing RunPod, requested a run under ten hours, and asked
that the pod remain running. Work stays in this directory and is committed only
on `main`. Nothing is pushed; the user handles the push.

## Deliverables

- [Single LaTeX manuscript](manuscript/mainj.tex), [BibTeX references](manuscript/references.bib),
  and [rendered PDF](manuscript/mainj.pdf): an anonymous development
  diagnosis and targeted correction study, not a completed confirmatory main study.
- [Pilot findings and limits](reports/pilot.md), [claim-to-evidence map](reports/claim_to_evidence.json),
  [table-cell sources](reports/manuscript_audit/table_cells.json), [experiment ledger](reports/experiment_ledger.json),
  and [compute ledger](reports/compute_ledger.json).
- [Frozen protocol](preregistration/protocol.json), [sampler amendment](preregistration/sampler_amendment.json),
  [raw run registry](results/run_registry.jsonl), and [source/data attribution](provenance/DATA_ATTRIBUTION.md).
- [Correction protocol](preregistration/velocity_followup.json),
  [paired results](results/velocity_followup/summary.json),
  [training-control audit](reports/velocity_training_match.json), and
  [independently recomputed metrics](reports/velocity_followup_validation.json).
- All three paper figures are beside `mainj.tex` in `manuscript/`: an analysis-setting
  comparison, a numerical/acoustic correction comparison, and three original
  acoustic case comparisons with plain-language labels.
  PDF/SVG and 450-dpi PNG exports remain in `figures/`, alongside the earlier
  supplemental workflow and waveform/PSD plots. The figure scripts refresh the
  active paper figures; the manuscript builder compiles the editable LaTeX directly.

Completed work includes the 43,208-file UniqueShip audit, a 200-crop metric check,
5,000-update conditional and 2,500-update deterministic pilots, exact CUDA checkpoint
resumption, 392 saved candidate/probe pools, a paired 32/64-step comparison, dry
calibration/testing, 800 synthetic/shift cases, and eight published-checkpoint transfer
cases. The transfer cases were replayed to save complete waveform artifacts; they
remain eight unique cases. Fixed seeds did not yield bitwise transfer reproducibility.
The added correction trains for 5,000 updates, evaluates 192 existing V groups with
two samplers, and independently verifies all 1,536 new candidate gap scores.

The three-seed main matrix, C4/C5/C6 controls, independent classifier, VLIZ audio,
and confirmatory statistical comparisons were not run after the no-go decision.
The 16-source learning diagnostic does not establish near-zero memorization.

## Operator workflow: labels and configuration

The paper connects audio quality to a proposed operator loop: **inspect evidence →
label or retain unknown → recommend settings → approve/configure → measure again**.
Each label needs a source and status: supplied annotation, model hypothesis, or
operator-confirmed identification. A frequency component is an observed feature;
it does not by itself establish object identity.

| Operator task | What this project provides | Capability status |
| --- | --- | --- |
| Distinguish recording from fill | Source records and an explicit missing/estimated-sample mask | Implemented and checked |
| Generate and assess possible gap fills | Diffusion proposals, known-audio checks, and a fixed release rule | Pilot evaluated; no proposals released |
| Diagnose and correct unstable generation | Matched noise/velocity targets, precision controls, and paired acoustic scores | Numerical failure corrected; baseline still wins mean acoustic error |
| Inspect a low-frequency component | [32 ms versus 128 ms analysis windows](manuscript/operator_configuration.pdf), with 4/16 ms hops and 31.25/7.81 Hz frequency bins | Demonstrated offline on the same available input; frames touching gaps excluded |
| Identify an object or vessel class | Existing dataset annotations; an independent labeling model is required | No identification accuracy evaluated |
| Configure physical equipment | Proposed review of acquisition gain/sample rate against instrument capabilities, operator approval, and a new measurement | No hardware commands or configuration benefits evaluated |
| Reason across tools and evidence | Proposed agent that retrieves clips/masks, requests plots, compares labels, and explains setting tradeoffs | Future extension; no language-model agent implemented |

The [configuration manifest](figures/operator_configuration.json) records exactly
which display settings were changed. The [supplemental decision plot](figures/operator_decisions.pdf)
shows all 64 reserved development cases: 52 selected proposals worsen error beyond
the baseline tolerance, and the release rule uses baseline fill throughout. That
baseline is also learned, estimated audio. Numerical error and fluent explanations
are not identification-confidence scores. The paper's operational examples are
illustrative; they are not an operator trial or a deployed sonar system.

## Acoustic examples

The paper's three anecdotes use saved dry-test V outputs: A is the most favorable
C3-minus-B score difference, B the most harmful, and C the lowest observed-context
spectral flatness among the 17 cases with 500 ms gaps. They are explicitly selected
illustrations. The cargo case improves waveform error while worsening spectral
convergence; the other-vessel case adds excess gap energy; the tanker case loses a
prominent low-frequency component. The original policy uses B in all three. The
corrected model's fixed first draw is added to these same cases selected before
the intervention. It reduces excess energy in B; closer level in C still does not
recover the waveform. Original C3 uses a learned selector, so its selection rule
differs from the new first draw. The main results also compare matched first draws.

| Case / source date / gap | Reference | Erased input | B = P | Selected C3 | Velocity first draw |
| --- | --- | --- | --- | --- | --- |
| A: cargo / 2017-12-24 / 50 ms | [WAV](examples/acoustics/A_cargo_reference.wav) | [WAV](examples/acoustics/A_cargo_masked.wav) | [WAV](examples/acoustics/A_cargo_fallback_B_equals_P.wav) | [WAV](examples/acoustics/A_cargo_selected_C3.wav) | [WAV](examples/acoustics/A_cargo_velocity_first.wav) |
| B: other vessel / 2017-01-26 / 50 ms | [WAV](examples/acoustics/B_other_reference.wav) | [WAV](examples/acoustics/B_other_masked.wav) | [WAV](examples/acoustics/B_other_fallback_B_equals_P.wav) | [WAV](examples/acoustics/B_other_selected_C3.wav) | [WAV](examples/acoustics/B_other_velocity_first.wav) |
| C: tanker / 2021-06-11 / 500 ms | [WAV](examples/acoustics/C_tanker_reference.wav) | [WAV](examples/acoustics/C_tanker_masked.wav) | [WAV](examples/acoustics/C_tanker_fallback_B_equals_P.wav) | [WAV](examples/acoustics/C_tanker_selected_C3.wav) | [WAV](examples/acoustics/C_tanker_velocity_first.wav) |

Each clip is 4.096 s at 8 kHz. One shared gain per case preserves relative levels
across its five versions, with no time stretching or independent loudness matching.
The [example manifest](figures/acoustic_example_selection.json) records source
times, gap sample indices, recomputed scores, processing details, and SHA-256 hashes.
Both the floating-point outputs and exported PCM preserve identical observed samples
across variants. These are listening aids, not subjective quality scores or calibrated
sound-pressure measurements. Credit UniqueShip's authors and Ocean Networks Canada
under the source recordings' CC BY 4.0 terms; [attribution and modifications](provenance/DATA_ATTRIBUTION.md)
are retained with the project.

## Inspect and rebuild locally

From this directory:

```bash
python3 -m pytest -q
PYTHONPATH=src python3 scripts/preflight.py
PYTHONPATH=src python3 scripts/posthoc_pilot.py
python3 scripts/plot_pilot.py
PYTHONPATH=src python3 scripts/build_acoustic_examples.py
PYTHONPATH=src python3 scripts/plot_velocity_followup.py
python3 scripts/build_reports.py
python3 scripts/build_manuscript.py
python3 scripts/validate_manuscript.py
```

`build_reports.py` requires locally backed-up candidate and diagnostic arrays,
deliberately excluded from Git. The acoustic example builder uses those frozen
arrays and processed source audio; it verifies their hashes, reproduces saved gap
metrics, and checks PCM export without running any model. Paper compilation needs only the files in
`manuscript/` and a TeX installation with `latexmk`, pdfLaTeX, BibTeX, NewTX,
`booktabs`, `float`, `microtype`, and `hyperref`. The build script keeps
auxiliary files under `.cache/latex/`; it never regenerates or overwrites the text
in `mainj.tex`. You can also compile directly from the manuscript directory:

```bash
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error mainj.tex
```

`mainj.tex` contains all prose, tables, and equations; there are no separate section
files. `references.bib` supplies citations. Unmodified publisher files `llncs.cls`
and `splncs04.bst` are included for portability. The original Word sample establishes
the LNCS template family; retrieval and comparison are recorded in
`provenance/latex_template.json`. The user superseded the original Word-deliverable
requirement with LaTeX only. Prior Word output remains in Git history at `a0e3a17`,
and its audit records are in `reports/manuscript_audit/word_draft/`. PDF author
metadata and the displayed author remain anonymous.

The user requested the upper end of the conference's published “8 to 10 pages”
wording. The compiled paper is **ten total pages including references**; the
organizers have not clarified that ambiguous wording. The paper adds three
equations explaining noise-error amplification, velocity prediction, and the
masked training loss. The [frozen follow-up protocol](preregistration/velocity_followup.json)
specifies one matched 5,000-update correction experiment; it does not reopen the
actual calibration or final-test partitions.

## Environments and GPU reproduction

`requirements.lock` records the CPU verification environment;
`requirements.gpu.lock` records Python 3.12 / PyTorch 2.8.0+cu128 on the RTX 4090;
`requirements.transfer.lock` records the separate legacy general-audio environment;
`reports/manuscript_audit/build_report.json` records the LaTeX engines and hashes;
`provenance/word_environment.lock` retains the former Word/figure environment. Full freezes
include inherited system packages and are environment inventories, not portable
one-command installers. The container image digest was not exposed by SSH and is
not invented. The host driver was not changed.

Exact training and inference commands, hashes, seeds, and statuses are in the run
registry. Training scripts are deliberately restricted to the development pilot;
`configs/essential.yaml` specifies future work and is not an implemented main runner.
The executed sequence was:

1. Download original split archives; join with `zip -s 0`; audit via
   `prepare_data.py <joined.zip> --cache-pilot`. The full native archive was not extracted.
2. Run `smoke_metrics.py`, `resume_check.py`, and `train.py` for seed 17: conditional
   5,000 updates, deterministic 2,500 updates. Model and data configurations are in
   each checkpoint directory. Reduced-width 16-source checks are separate runs.
3. Preserve `diagnose_sampler.py` failures; use the frozen G-only amendment in
   `make_pools.py --sampler-config preregistration/sampler_amendment.json`.
4. For run `pilot_pools_clipped_seed17`, generate `--phase fit`, run
   `analyze_pilot.py --stage fit`, compare steps, generate `--phase calibration`,
   analyze `--stage calibrate`, generate `--phase test`, then analyze `--stage evaluate`.
   These phase names refer only to reserved V subsets, never real C/test.
5. Run the frozen `run_diagnostics.py`. Fetch published transfer artifacts with
   `fetch_transfer.py`, apply `prepare_transfer_compat.py` within `.venv-transfer`,
   and run `transfer_check.py`. The startup failures, compatibility patches, original
   scored run, and waveform replay are all retained.
6. Freeze `preregistration/velocity_followup.json`; train the same conditional
   architecture with `--prediction-type velocity --updates 5000 --seed 17`.
   `verify_velocity_training.py` checks matched training streams;
   `evaluate_velocity_followup.py` evaluates the fixed final checkpoint and
   original frozen cases. `sync_velocity_followup.py` verifies 206 transferred
   files against the remote copies; `plot_velocity_followup.py` independently
   recomputes scores and checks the derived error-propagation identities.

Use new run identifiers for any new experiment. Do not overwrite completed artifacts
or treat a rerun as another independent sample. The dated authorization in
`configs/run_authorization.json` applies to this run only; scripts enforce its deadline.

## Data, preservation, and resource state

UniqueShip/Ocean Networks Canada data are CC BY 4.0 with attribution recorded in
`provenance/`. Source/group manifests precede resampling, crops, and artificial masks.
Missing references are erased at the inference boundary; normalization is recomputed
after each probe mask; exact projection retains observed samples. Repaired positions
keep explicit imputation flags. Original field recordings are not noise-free truth.

Raw archives, processed audio, environments, checkpoints, and candidate arrays are
ignored by Git. Both original archive volumes and twelve checkpoints have verified
local backups; per-case arrays have recorded SHA-256 hashes. Download/retrieval
locations are in `manifests/` and `provenance/`. The remote `/workspace` is a 50 GB
container overlay, not a confirmed persistent volume. Keep the local backups.

The pod is intentionally left running. Research jobs finish within the ten-hour
cap, but leaving the pod running continues billing. The actual booked hourly rate
and storage charges were not supplied, so no dollar total is claimed. To stop it
later, use the existing pod's **Stop** control in RunPod after confirming backups;
do not terminate it expecting the container disk to persist. No shutdown, deletion,
new paid resource, repository publication, email, or submission is performed here.
