# Checking sound and action: generative AI for underwater acoustic analysis

The targeted follow-up **diagnoses and corrects a numerical failure** in generative
underwater audio repair. A matched velocity-prediction model removes the observed
terminal-step blow-up: full-precision gap NMSE on four diagnostic sources falls
from 5.18 million to 1.03. On 64 reserved development cases, first-draw error Q
falls from 0.412 to 0.308, about 25%; the learned baseline remains better at 0.239.
The anonymous paper presents the mechanism, three new equations, controls, and
before/after spectrograms. These are exploratory results on reused development data.

A **GPU agent comparison** evaluates fixed rules, direct Qwen3-8B, Qwen3 reasoning,
reasoning with self-review, and a planner–reviewer pair on 32 recorded acoustic
tasks. Every neural approach uses the same BF16 weights on the user-supplied
RTX PRO 4500 Blackwell. The tasks require a supported acoustic-cue label, a feasible
virtual display setting, and executed verification. A separately frozen follow-up
compares reasoning with and without a configuration feasibility gate on sixteen
additional source groups and dates. These are new agent outcomes on reused acoustic
development data; they do not evaluate vessel identification or physical equipment.
The frozen protocols and all outcomes, including failures, are retained below.
A third, exploratory budget check reuses the first sixteen primary tasks and their
original reasoning outputs, allowing 2,048 thinking tokens per turn and 12,288 total
generated tokens. It addresses frequent truncation under the original cap; these
sixteen paired tasks are not counted as additional independent recordings.

The primary complete-task counts are 10/32 for direct mode, 4/32 for reasoning,
8/32 for self-review, and 9/32 for the planner–reviewer; none of the three predefined
paired contrasts is conclusive after adjustment. Fixed rules complete 32/32.
The gate blocks infeasible applied changes (six to zero), while completion changes
from 2/16 to 3/16 without conclusive evidence of improvement. The larger budget
improves correct settings from 8/16 to 12/16 and completion from 3/16 to 5/16
(exact p=0.6875), while mean tokens rise from 3,044 to 6,278 and median time from
111.6 to 187.2 seconds. These results support checking executed state and reporting
actual cost; they do not establish a general ranking of reasoning ability.

All three phases are complete: **208 scored episodes, including 176 neural episodes,
on 48 unique recording groups**, in **5 hours 24 minutes**. Replay audits verify
1,109 unique tool returns and 1,080 CUDA turns. The [completion record](reports/gpu_agent_remote_completion.json)
confirms that research jobs ended and the pod was left running. The
[backup manifest](manifests/verified_gpu_agent_backup.json) verifies 450 closed remote
output and resource files. No new language-model inference ran on the local computer.

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
  diagnosis, targeted correction, and controlled GPU agent comparison; the original confirmatory restoration study remains incomplete.
- [Pilot findings and limits](reports/pilot.md), [claim-to-evidence map](reports/claim_to_evidence.json),
  [table-cell sources](reports/manuscript_audit/table_cells.json), [experiment ledger](reports/experiment_ledger.json),
  and [compute ledger](reports/compute_ledger.json).
- [Frozen protocol](preregistration/protocol.json), [sampler amendment](preregistration/sampler_amendment.json),
  [raw run registry](results/run_registry.jsonl), and [source/data attribution](provenance/DATA_ATTRIBUTION.md).
- [Correction protocol](preregistration/velocity_followup.json),
  [paired results](results/velocity_followup/summary.json),
  [training-control audit](reports/velocity_training_match.json), and
  [independently recomputed metrics](reports/velocity_followup_validation.json).
- [GPU comparison protocol](preregistration/gpu_agent_benchmark.json),
  [pre-evaluation scoring amendment](preregistration/gpu_agent_scoring_amendment.json),
  [frozen implementation](preregistration/gpu_agent_evaluation_freeze.json), and
  [separate configuration-gate follow-up](preregistration/gpu_agent_guard_followup.json),
  plus the [reasoning-budget check](preregistration/gpu_agent_budget_followup.json).
  Input waveforms, masks, tasks, raw model responses, and executed tool returns are
  under `results/gpu_agent_benchmark/`, `results/gpu_agent_guard/`, and `results/gpu_agent_budget/`.
- All three paper figures are beside `mainj.tex` in `manuscript/`: a recorded acoustic
  example with the comparative agent results, a numerical/acoustic correction comparison, and three original
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
and the original restoration study's confirmatory statistical comparisons were not
run after the no-go decision.
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
| Use tools and audit claimed actions | Four Qwen3-8B workflows and a rule baseline; real measurements, cue labels, and virtual display configuration | Frozen comparison on 32 source groups, plus a separately frozen 16-group feasibility-gate follow-up; bounded task rules and one model/base seed |

The [configuration manifest](figures/operator_configuration.json) records exactly
which display settings were changed. The [supplemental decision plot](figures/operator_decisions.pdf)
shows all 64 reserved development cases: 52 selected proposals worsen error beyond
the baseline tolerance, and the release rule uses baseline fill throughout. That
baseline is also learned, estimated audio. Numerical error and fluent explanations
are not identification-confidence scores. The agent receives numeric tool returns,
without vessel labels, complete-reference gaps, or reconstruction scores. Neural
inference for the new comparison runs on the supplied GPU and changes only a virtual
analysis display. The paper's operational examples are
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
PYTHONPATH=src python3 scripts/analyze_gpu_agent_benchmark.py
PYTHONPATH=src python3 scripts/analyze_gpu_agent_guard_followup.py
PYTHONPATH=src python3 scripts/analyze_gpu_agent_budget_followup.py
PYTHONPATH=src python3 scripts/audit_gpu_agent_failures.py
python3 scripts/audit_gpu_agent_budgets.py
PYTHONPATH=src python3 scripts/plot_gpu_agent_benchmark.py
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

The conference homepage specifies **8–10 pages**. The compiled paper is **ten total
pages including references**; the site does not separately specify reference inclusion. The paper adds three
equations explaining noise-error amplification, velocity prediction, and the
masked training loss. The [frozen follow-up protocol](preregistration/velocity_followup.json)
specifies one matched 5,000-update correction experiment; it does not reopen the
actual calibration or final-test partitions.

The retained CPU agent is an earlier local demonstration after the user reported the pod
unavailable. Its [task and limits](preregistration/cpu_agent_demo.json),
[transport compatibility amendment](preregistration/cpu_agent_transport_amendment.json),
and [post hoc audit continuation](preregistration/cpu_agent_audit_continuation.json)
were committed before their respective model outputs. An initial HTTP 400 before
inference is preserved. The installed Ollama 0.3.2 / Q4_0 model digest is pinned in
`results/cpu_agent_demo/model_environment.json`; the current public model tag differs.
No model download, external API, new GPU, or acoustic retraining was used. Saved
requests and responses allow exact tool-result replay, without assuming language-model
reruns are bitwise reproducible. The plotted evidence card is derived from tools and
is explicitly not the model's unverified prose.

The subsequent GPU comparison changes both model and protocol relative to that
guided CPU example. It does **not** isolate a causal benefit of GPU hardware or
show that GPU execution fixes a CPU reasoning failure. Within the new comparison,
weights and task rules are controlled. Thinking mode, continuation structure, and
the tool gate are the evaluated changes. Equal ceilings (eight turns, sixteen tool
calls, 6,144 generated tokens) do not mean equal actual cost. The 512-token per-turn
thinking cap and forced closing of unfinished reasoning are recorded. Rules encode
the specified policy and provide a task-specific ceiling, not a general acoustic
intelligence system. Cue labels are defined by supplied prominence, frequency, and
relative-level rules; their class balance limits interpretation.

## Environments and GPU reproduction

`requirements.lock` records the CPU verification environment;
`requirements.gpu.lock` records Python 3.12 / PyTorch 2.8.0+cu128 on the RTX 4090;
`requirements.transfer.lock` records the separate legacy general-audio environment;
`provenance/gpu_agent_environment.json` pins the Qwen3 model revision, weight hashes,
CUDA residency, and library versions on the replacement pod;
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
7. On the replacement pod, set up `.venv-agents` with `scripts/setup_gpu_agents.sh`
   and the pinned Qwen3 revision. Prepare frozen observed-only tasks with
   `prepare_gpu_agent_cases.py`. Run two G-only transport cases before the scored
   comparison, preserving the documented scoring corrections. The scored job uses
   `launch_gpu_agent_evaluation.py`; `queue_gpu_agent_guard_followup.py` starts the
   separately frozen sixteen-group follow-up after the primary job exits. Both use
   `configs/gpu_agent_authorization.json`, CUDA-only model weights, and a hard job
   deadline. `sync_gpu_agent_results.py --final` copies closed outputs and verifies
   their hashes against all completed remote phases. The analysis scripts replay
   tool measurements and rescore every report without running a language model.
   `queue_gpu_agent_budget_followup.py` subsequently runs the one larger-budget
   condition if at least three authorized hours remain. It adds sixteen neural
   episodes on reused tasks, preserving the original paired control outputs.

Use new run identifiers for any new experiment. Do not overwrite completed artifacts
or treat a rerun as another independent sample. The dated authorizations in
`configs/run_authorization.json` and `configs/gpu_agent_authorization.json` apply to
their respective GPU sessions; scripts enforce their deadlines.

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

The original RTX 4090 pod was left running after its study, whose jobs finished
within the ten-hour cap; the user subsequently reported it unavailable. The new
GPU experiments use the user's replacement RTX PRO 4500 pod under a separate
ten-hour authorization, with instructions to leave it running. No additional
resource is provisioned. The actual hourly rate and storage charges were not
supplied, so no dollar total is claimed. Hash-verified local outputs preserve the
new results; publicly available pinned model weights remain on the pod. No push
or submission is performed here.

## Readability revision and AI disclosure

The paper now opens with an operator examining a missing tone and an assistant
that reports one display setting while applying another. The abstract explains
one central contribution: checks that locate different failures and test a remedy
for each. Equation explanations and acoustic examples connect the measured
results to decisions about recorded evidence, generated estimates, and executed
settings. The conclusion gives concrete design lessons. Experimental outcomes,
all 110 table values, five equations, and the three figure datasets are unchanged.

The [policy review](reports/ai_disclosure_policy.json) distinguishes the conference
instructions from Springer’s books/proceedings guidance. At the user’s explicit request, the paper includes only a short end declaration;
first-page and caption notices are omitted. The policy review records the difference
from the publisher guidance. Codex’s role includes research planning, code, execution, analysis, and
writing; it is not described as copyediting alone. Evaluated models are described
in the methods. The human authors still need to verify and take responsibility for
the submission. No completed human-review attestation or organizer approval is claimed.
The [audio companion note](examples/acoustics/README.md) identifies the recorded
and estimated clips and the code assistance used to prepare them.
