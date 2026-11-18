# Reading underwater acoustic signals

This project studies bounded inspection of passive underwater acoustic recordings
through linked LOFAR and DEMON views. The original research plan is preserved in
`underwater_acoustic_agent_research_plan.md`. Subsequent decisions and deviations
are recorded in `configs/protocol_amendments.json` and the experiment manifests.

The user explicitly requested AI-generated annotations in place of human
annotators. **Neither annotation checkpoint passed qualification:** the frozen
eligible-type lists are empty in `configs/ai_reference_freeze.json`. Primary real
event scores, the validation-selected fixed comparator, and real H1/H4 tests are
unavailable, not zero accuracy. Raw model annotations and consensus are retained
as unqualified diagnostics; they do not establish physical correctness or
exhaustive negatives. Separate synthetic controls provide exact declared
component truth. Failed annotation and generation attempts remain in the record.

Both final annotation streams are preserved and hash-verified (514 scene records;
`results/ai_annotations/COMPLETED.json`). Across the 245 real/mixture scenes, the
4B checkpoint retained 113 events and the 8B checkpoint none, yielding empty
consensus. The twelve separate audit controls contain three cross-checkpoint
agreements and are excluded from that aggregate. Neither audit promotes a
checkpoint into the qualified scope.

The manuscript uses Alexander V. Mantzaris and James H. Korris with the
affiliations confirmed from the existing paper in this repository. Submission
itself is not automated.

## Paper and evidence

The completed local manuscript uses the actual saved results and figures and fits
10 pages including references. Main figures 1–7 appear in the paper; artifacts
8–9 provide separate robustness/efficiency detail in the local package. All nine
PDF/PNG/SVG figures and their hashed scientific inputs are retained. This does
not presume conference acceptance of supplementary material. Author review and
submission remain outstanding; neither real-reference qualification nor the
original real-grounding success criteria were achieved.

Earlier expanded drafts and clearly labelled layout-only proofs remain under
`provenance/manuscript/` and `reports/pagination_layout_proof/`. Their proxy boxes
are historical page-budget tests, not scientific results. The final manuscript
uses actual recorded figures; final build and page-inspection evidence are in
`reports/manuscript_finalization.json`.

After shutdown, three host-only scripts refined labels without changing
scientific inputs: `scripts/rerender_case23_readability.py`,
`scripts/refine_figures_06_09_presentation.py`, and
`scripts/rerender_figure08_saved.py`. They use archived original SVGs or saved
CUDA display arrays/JSON. Originals remain under
`provenance/presentation_versions/`; the corresponding reports are
`reports/case23_readability_rerender.json`,
`reports/figures06_09_local_presentation.json`, and
`reports/figure08_saved_value_rerender.json`. The central figure manifest binds
the final formats, unchanged numeric inputs, and presentation provenance.

- `manuscript/main.tex`, `references.bib`, and `main.pdf`: paper source and build.
- `figures/` and `manuscript/figure01.pdf` through `figure09.pdf`: measured case
  studies, comparisons, ablations, robustness, and efficiency.
- `viewer/index.html`: local paired-view case gallery and evidence/action records.
- `reports/task_ledger.json`: current execution status; do not infer completion
  from an implementation file or a planned experiment.
- `reports/outline_deliverable_audit.json`: original-plan coverage, documented
  feasibility deviations, reporting limits, and remaining delivery checks.
- `results/`: raw model text, parsed/verified reports, training histories,
  predictions, metric tables, and failed-run records.
- `reports/local_artifact_receipt.json` and `reports/pod_shutdown.json`: verified
  local preservation and the final RunPod control-plane response.

## Data and scope

The dense inventory contains 210 untouched real scenes, 35 declared mixtures of
real excerpts, and 35 synthetic scenes: 280 scenes, 75% untouched real. The main
held-out cohort contains 90 real, 12 mixture, and 12 synthetic scenes. Counts and
hashes are in `data/manifests/`.

The real scenes are contiguous 30-second excerpts: 150 from the accessible
DeepShip subset and 60 from separate MBARI recording days. The real split is
90/30/90 scenes and 26/9/69 recording/day groups. Vessel-disjointness is not
claimed. The 60 MBARI days remain external tests. Short Qiandao examples are
retained separately and are never concatenated into invented continuous scenes.
DeepShip's internal classification test has no Tug reference examples; the
classifier reports both frozen-label and present-reference-class averages.
Paired classifier uncertainty uses the fixed four-label vocabulary; present-class
means and training-seed standard deviations are descriptive only.

The original 80-real-scene ablation design measures report production and cost
because qualified references are unavailable. During real-ablation execution,
after main results were available, the user requested at most 200 new episodes
instead of 480 because runtime was too long. The selected continuation plan uses
exactly the first 33 scenes in the original frozen manifest, across all six new
variants. Completion is now confirmed: **198 retained variant episodes**, comprising
156 preserved from the interrupted run and 42 completed after resumption, with
zero execution failures. The common cohort contains 30 DeepShip scenes and three
MBARI scenes and is therefore domain-imbalanced. The composed real comparison
contains 297 records across nine policies and adds no inference calls. See
`reports/capped_ablation_completion_check.json`; the immutable transfer and
saved-value interpretation are verified in `reports/capped_ablation_transfer_receipt.json`
and `reports/capped_ablation_results_interpretation_audit.json`. Interrupted partial work and
the original inventory remain separately preserved, rather than counted as
completed matched-cohort results. This runtime change was made after main
scoring, during ablations; it was not preregistered. Frozen algorithms/configs
and the original design remain preserved.

The pre-main amendment in `configs/synthetic_ablation_amendment.json` proposed
six new variants on twelve existing synthetic cases (72 new episodes). Those
new synthetic inferences are now **omitted** to honor the overall ablation cap.
The remaining synthetic comparison reuses only F, G4, and F without numerical
filtering from existing main outputs: twelve scenes, three policies, 36 completed
composed records, with no new generation. Both real and synthetic common
evaluations and their verified transfer/interpretation audit are complete.
F without numerical filtering retains the report
schema checks. Figure7 must show separate real and synthetic method inventories;
it must not imply that the six omitted variants or a full nine-policy synthetic
ablation ran. `configs/ablation_user_cap.json` explicitly authorizes the
corresponding presentation branch; the common evaluator and
scientific algorithms stay unchanged. The original 18-scene budget subset remains
six real scenes and twelve synthetic cases, reported separately.

On the 33-scene real prefix (twelve recording/day groups), removing the graph or
component/measurement tables yielded 33/33 empty retained reports, versus 16/33
for F. This is report-production behavior, not a real accuracy result. Retained
emptiness can follow numerical rejection even when the raw model report did not
abstain. On the twelve reused synthetic controls, F without numerical filtering
had TP/FP/FN 12/27/12 and F1 0.3809523880; F had 12/11/12 and F1 0.5106382966.
Both retain the same whole-report schema gate. These saved aggregates do not
establish paired significance, independently replicated gains, or the effects
of the six omitted synthetic variants.

The verified joint budget audit is in
`reports/budget_results_interpretation_audit.json`; both new runs contain 54
completed episodes. On the same twelve synthetic cases and identical sealed
truth, F1 at two/four/six slots is F 0.612903/0.612903/0.510638,
G3 0.612903/0.586207/0.478261, and G4 0.555556/0.454545/0.641509.
These descriptive scores do not show consistent improvement with additional
slots. F's two/four-slot TP/FP/FN aggregates both equal 19/19/5, but their
empty-report counts are four and five; equal aggregate scores do not imply
identical outputs. Schema failures at two/four/six slots are F 0/0/1,
G3 0/1/0, and G4 2/2/0, retained in the denominators. Marginal group-sampling
intervals do not supply paired budget tests. The real two/four-slot runs share
six scenes/five groups and lack qualified accuracy labels. Matched six-scene
slot-six cost accounting awaits the queued CUDA Figure9 export; the full
90-real-scene mean is not a substitute, and no subgroup average is calculated
on CPU.

The original plan's larger benchmark, human validation, and natural-overlap
accuracy claims have not been achieved by this feasibility study. The current
system uses deterministic spectral proposals and a finite inspection menu;
learned classifiers/localizers are separate baselines. Numerical verification
checks component consistency and evidence links, not free-text truth. No
calibrated confidence probabilities or operator-benefit claim is made.
Disjoint synthetic parameter seeds do not establish transfer to unseen generator
families; recording/day split checks cannot exclude pretrained-model corpus overlap.

Acquired audio remains local under `data/raw/`. Provider terms govern any later
redistribution; a source-code license is not treated as an audio license. The
acquisition scripts, provider identifiers, byte hashes, and exact excerpt
coordinates are retained even when audio cannot be redistributed.

## Protocol details accompanying the compact paper

The compact paper removes repeated tables and longer procedural exposition;
the expanded source above, frozen code/configurations, and result exports remain
available. The following details are part of the recorded protocol, not new
experiments or post-result tuning.

LOFAR uses the one-sided Hann STFT PSD
`P[m,k] = c_k |X[m,k]|² / (fs * sum_n w[n]²)`, where `c_k=2` at interior
positive frequencies and `c_k=1` at DC/Nyquist. Values are uncalibrated digital
amplitude squared per hertz. Native PSD supplies measurements; frequency display
pooling does not improve spectral resolution. DEMON uses the analytic envelope
after carrier filtering, with original-recording time coordinates. Carrier
availability depends on sample rate, waveform duration, and filter margins.
The initial carrier is 500–2000 Hz; additional bands are 2000–5000 and
5000–8000 Hz. A modulation peak does not establish blade count or source identity.

The same pinned Qwen3-VL-8B checkpoint and deterministic decoding support all five
policies. G1 reads global LOFAR alone without numerical component verification.
G2 reads paired views and quantitative evidence once. G3 follows a fixed ordering
of available inspections; G4 prioritizes unseen tools/carriers and unresolved
relations. F chooses an inspection or finalization from the current valid menu.
Selection/report limits are 160/2048 tokens. Reports permit six components,
three claims, relations, unresolved questions, and abstention. The JSON grammar
constrains syntax; it cannot establish correct geometry or semantics. The schema
gate is separate from numerical filtering. The no-verifier condition reuses
raw F after report schema checks, not all malformed raw output, and does not
measure avoided verification runtime.

The report filter checks morphology/domain, cited evidence, endpoint agreement
within two native bins, time boundaries, carrier compatibility, and ordered
chirp tracks. It preserves raw reports and rejection reasons. Free-text claims
are not semantically scored, and source identity is not independently validated.
Removing component/measurement tables retains graph relation values, overlap
fractions, and axes metadata. Fixed-resolution ablation affects acoustic LOFAR;
DEMON always uses four-second windows. The synthetic “occlusion” condition gates
tones in time; it is not a physical masking experiment.

P1/P2 classifiers have 155,492/310,340 parameters, use training-only
normalization, weighted cross-entropy, AdamW, validation stopping, and seeds
1729/2718/3141. Their weak-label split is 90/30/30 DeepShip scenes. The separate
localizers use 192/48 auxiliary exact synthetic training/validation scenes and
seeds 17/29/43. The auxiliary pool adds an independent broadband-burst condition;
it does not change the main synthetic controls. Domain-specific grids, weighted
binary cross-entropy plus Dice loss, native-PSD decoding, and validation-selected
thresholds operate under the same six-component cap. No real component labels
train them; P1/P2 are not parameter-matched. Shared feature preparation computes
both LOFAR and DEMON even when P1 consumes only LOFAR.

The 35 within-split real mixtures use 17/6/12 train/validation/test scenes,
30–60-second support, 0/25/50/75/100% parent overlap, and −10/0/+10 dB
excerpt-power ratios. The 35 primary synthetic scenes use the same scene-count
split. Shared-parent scenes are connected into provenance groups, rather than
treated as independent pairs. All twelve test mixtures belong to one such group;
their sparse overlap/level cells are descriptive, not replicated factorial tests.

Exact event matching uses greedy one-to-one assignments, temporal IoU at least
0.3, and narrowband track error within 2 Hz. The unused qualified-AI tolerance is
`max(4 Hz, annotation uncertainty)`, capped at 8 Hz; predicted/display bin widths
cannot relax it. Regions require time–frequency IoU at least 0.3, with IoU 0.5
retained as a separate secondary endpoint. Modulation matches also require
compatible carrier bands. Optional higher beat harmonics receive neither credit
nor penalty. Unenumerated modulation is unscored outside the explicitly
exhaustive synthetic families. Matched geometry errors condition on accepted
matches and must be read with coverage; duplicate candidates are geometric
diagnostics, not confirmed physical duplicates.

The frozen real-reference gate admitted no morphology. Thus primary real H1/H4
tests and the validation-selected fixed comparator remain unavailable; unqualified
checkpoint diagnostics cannot replace them. The full five-policy synthetic
TP/FP/FN, F1 and group intervals, real report-production/schema counts, token and
timing fields remain in `results/main/evaluation/summary.json` and its table and
per-episode exports. Their verified source pointers and interpretation limits are
in `reports/main_results_interpretation_audit.json` and
`reports/manuscript_evidence_map.json`.

Final secondary summaries are bound in
`reports/final_saved_results_interpretation_audit.json`. Per-morphology and
relation outputs remain separate from common event scores. On the twelve main
synthetic cases, F tone/chirp/transient/modulation F1 was
0.485/0.400/0.667/0.800 on only 15/4/2/3 required events; broadband had no required
positives. Explicit temporal-overlap-edge F1 was F/G4 0.077/0.375. Both missed the
single required time–frequency-overlap edge; eleven scenes had no such edge.
No free-text explanation accuracy is supplied.

Mismatched-view report agreement was F/G4 0.947/1.000 on five scored pairs each,
with one both-empty pair per method omitted. These are output-stability metrics,
not correctness or proof of visual use. Real target-attenuation agreement was
0.333/0.571, sham agreement 0.476/0.667, and parent-removal agreement 0/0.049
(F/G4). These compare reports to reports without physical component truth or
causal source-accuracy claims. The pooled exact-synthetic sensitivity F1
0.542/0.612 combines duration and mismatch probes; it is not duration-only
accuracy. Its six base groups also do not establish a real-data population effect.

The final Figure 9 matches the same six real scenes at all budgets. Six-slot
mean recorded CUDA spans are F/G3/G4 113.0/22.9/32.0 seconds. The older 90-real-scene
main means (125.2/24.0/27.4 seconds) use a different cohort; F's corresponding
wall median/p95 are 1403/1779 seconds. These are incomplete, schedule-specific
accounting, as detailed below. The contrast-ranked risk–coverage curve is
unavailable because no real-reference type qualified. No calibration curve or
physical error probability is invented.

Intervention manifests retain exact gain/noise/bandwidth/removal/silence/colormap
settings. Further probes include ten-second excerpts, mismatched initial DEMON
images, 200–400 Hz real attenuation with a 1000–1200 Hz sham (both 10–20 seconds),
and unchanged-gain excerpt removal. Waveform edits regenerate views; image
mismatch retains the target waveform's numerical evidence. Silence requires a
valid empty report. Contrast-ranked risk–coverage groups ties and supplies no
calibrated confidence probability.

Figure 8 separates recovery from scope availability. Nonblank cells with zero
required scoped events display recovery as N/A regardless of the model output,
with scoped false-positive and schema-invalid episode counts visible. This does
not mean silence or physical absence: excluded modulation is unobserved. Blank
waveforms retain the separate fraction of schema-valid empty reports. The plot
input JSON preserves frozen per-episode and aggregate F1/counts, including the
stored zero F1 where no required event is present; the evaluator is unchanged.
Support is checked on CUDA using TP+FN against saved `reference_event_count`.

Intervention truth events, scope booleans and the actual waveform state take
precedence over inherited descriptive metadata. For example, bandwidth-loss
episodes record `final_state.sample_rate_hz=4000`, while inherited generator
truth/manifest sample rates can remain 16000. Inherited
`modulation_evaluation_scope`/`scoring_scope.rule` prose can also be stale after
blank/bandwidth edits; it is not the authority for event eligibility. Exact
bindings are in `reports/robustness_scope_metadata_companion.json`. Added-noise
dB specifies the existing-waveform/added-noise ratio, not achieved clean-signal
SNR; repeated blank conditions are not independent waveforms.

Case selection first seeks qualified tonal/overlap references, then uses opaque
IDs with original/domain diversity; empty eligibility forces explicitly recorded
fallbacks. Adaptive cases prefer two saved F action images. Failure cases follow
the first schema-failure/abstention rule, otherwise the declared disagreement or
unscored fallback. Full choices are frozen in `figures/case_selection_freeze.json`;
model outputs cannot retroactively change the policy. The offline viewer retains
linked cursors, component/evidence highlights, saved action views, and separate
corrections; optional reinspection never overwrites benchmark outputs.

## Execution

Signal transforms/resampling, synthesis, learned-model forward/backpropagation,
matching, scientific metrics, and uncertainty estimates run on the authorized
CUDA GPU. These paths require CUDA rather than falling back to CPU computation.
Tokenization, JSON/control, decoding, transfer, and final rendering use the host.

The pinned vision processors are an explicit exception to the original
all-numerical-on-GPU requirement. `Qwen2VLImageProcessorFast` and InternVL's
`GotOcr2ImageProcessorFast` receive PIL images with their default `device=None`:
they create CPU tensors and perform tiling, resizing, rescaling, normalization,
and patch rearrangement before the returned batch moves to CUDA. This was found
by source audit during early main inference. Frozen processor settings and
completed runs are preserved; the deviation is documented in
`reports/host_input_preprocessing_audit.json`, rather than described as GPU-only
image preparation.

The pod uses an RTX 6000 Ada with about 48GB VRAM, 188GB allocated host memory,
and a 50GB container disk. The recorded service quote is $1.09/hour; this is not
a complete billing statement. Exact package/model revisions are in
`reports/environment.lock.txt`, `reports/model_revision.json`, and the annotator
revision records. Credentials are never stored in project artifacts.

The scripts are run from this directory with `PYTHONPATH=src`. The remote virtual
environment is `.venv/`; its PyTorch 2.8.0 CUDA build comes from the pinned RunPod
image. `requirements.gpu.txt` pins the added runtime packages.

`scripts/prepare_remote.sh` is the historical first-provisioning script: it
queries the model repository's current HEAD and records the resolved revision.
It must not resolve a fresh HEAD when reproducing this study. Use the saved
Qwen revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` from
`reports/model_revision.json` explicitly in `snapshot_download`, for example:

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="Qwen/Qwen3-VL-8B-Instruct",
                  revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                  local_dir="models/qwen3-vl-8b")
```

Use the separately recorded fixed revisions for the annotation checkpoints too.

Paper overview displays redraw PSDs with per-view 5th/99.5th-percentile log-PSD
limits. They are not pixel-identical to model-input displays. Action thumbnails
preserve the recorded tool rasters; plot inputs retain the already GPU-computed
overview limits for later presentation-only reflow.

```bash
# Administrative acquisition and manifest construction:
python3 scripts/acquire_real_data.py
python3 scripts/freeze_manifests.py
python3 scripts/freeze_mechanism_tests.py

# CUDA checks and reproducible inference, using a new output directory:
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*gpu.py' -v
PYTHONPATH=src .venv/bin/python scripts/run_benchmark.py \
  --manifest data/manifests/main_test.jsonl --output results/new_main \
  --methods G1,G2,G3,G4,F --budget 6 \
  --workers 16 --batch-size 8 --batch-wait-ms 2000
```

Do not rerun acquisition or change a frozen manifest inside an existing result
directory. The inference runner checks behavior-source, model, manifest, and
budget invariants before resuming. Raw annotation calibration, eligibility rules,
and held-separate annotation audits must be preserved alongside main references.

Generation CUDA-event timers start after host image preparation and input
transfer; request/batch wall timing includes both. Amortized accelerator elapsed time and batch throughput are
distinct from serial per-scene latency. Serial development measurements and
batch-ledger IDs retain those distinctions. Three trained seeds are genuine
replications of supervised baselines. Identical greedy frozen-model calls are
not relabeled as independent seed experiments.

Mixed-method batching affects request/episode waiting and divides each recorded
CUDA batch span equally among its requests. Final `verify_report` executes on
CUDA under the shared GPU lock but outside dedicated verification timers.
Summed overview/generation/action spans are incomplete and schedule-dependent
accounting; they are not pure model time, standalone method
latency, or marginal/billable GPU cost. Whole-episode wall time includes
verification and waiting. Equal attempted-slot ceilings also do not equalize
tokens or compute. An earlier audit incorrectly attributed overlapping spans to
unlocked verification; the unchanged frozen source already held the lock. That
attribution is withdrawn. No code behavior or score changed.

## Preservation and shutdown

The managed local supervisor, `scripts/finish_gpu_and_stop.py`, waits for the
successful experiment/export queue and an idle GPU before preserving its outputs.
It follows this sequence:

1. Copy the completed generated artifacts, checkpoints, manifests, and source
   snapshot into an isolated timestamped `runpod_archive/` directory. This avoids
   replacing locally revised manuscript text with an older remote copy.
   The archive retains trained checkpoints and pinned public-model revisions;
   public pretrained VLM weight caches are excluded and can be fetched again
   using those exact revisions.
2. Verify every entry in the remote artifact receipt against its local SHA256,
   run the offline viewer's browser checks, and run
   `scripts/verify_final_deliverables.py --gpu-only`. This requires the scientific
   outputs, all nine recorded figures, and the viewer; CPU manuscript finishing
   is checked separately.
3. Recheck the entire remote receipt inventory, launched queue identity, and
   GPU-idle state after transfer and validation. Changed artifacts or active work
   prevent a stop request.
4. Request the authorized pod's **stop** transition through
   `scripts/pod_control.py stop` and record the confirmed control-plane state.
   The supervisor does not terminate or delete the pod, and the pod-scoped
   control credential remains remote.

`reports/hold_gpu_shutdown` prevents the supervisor from stopping the pod when
further GPU work is authorized. After that work finishes, remove the hold and
restart the supervisor; a hold is not silently ignored. Failed or incomplete
queue, receipt, or validation checks also prevent an unverified stop. Progress is
recorded in `reports/automatic_preservation_status.json`; final evidence belongs
in `reports/local_artifact_receipt.json`,
`reports/final_remote_preservation_guard.json` inside the archive, and
`reports/pod_shutdown.json`. The completed run was preserved and verified locally
before the pod was stopped: the control plane confirmed **EXITED at
2026-11-17 21:39:43 UTC**. The local preservation receipt verifies 13,308 files;
the instance was stopped, not terminated or deleted. No further GPU work is
required for this package. `scripts/backup_from_pod.sh` remains as a historical
manual-retrieval utility.

Final CPU manuscript pagination and presentation-only label fixes used preserved
plot inputs after shutdown. No scientific metric was recomputed on CPU. The
final local source snapshot is a separate closing record from the immutable
remote archive.

Compile the paper locally with `latexmk -pdf -interaction=nonstopmode
-halt-on-error main.tex` from `manuscript/`. The checked conference guidance is
in `reports/conference_requirements.json`; the conservative target is at most
10 pages including references. Human authors still need to review the final
paper and submission form.
