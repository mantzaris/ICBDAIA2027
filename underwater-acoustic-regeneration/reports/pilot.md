# Acoustic repair and GPU agent findings

The frozen feasibility requirement was at least 20% of tuning-V cases with one
candidate improving Q over B by 0.02. The observed result was **5/64 = 7.8125%**
with both 32 and 64 DDIM steps. The model trained for 5,000 updates on approximately
two hours of G audio. This outcome rejects proceeding with this pilot configuration;
it does not show that diffusion or selective inpainting is generally ineffective.

On 64 separate development calibration groups, all four gate variants accepted
zero cases at their sole V-frozen threshold. Their binomial upper bound is one,
so no threshold qualified. The resulting policy P uses deterministic B everywhere.
On another 64 V groups, Q is 0.23879 for B/P and 0.34897 for always-accept C3;
C3 harms 52/64 cases relative to B. Zero acceptance makes accepted-case risk
undefined and does not establish successful risk–coverage control. Real C and
test outcomes remain unopened.

## Isolating and correcting the numerical failure

At the user's request, one matched intervention was frozen before its new outcomes
in `preregistration/velocity_followup.json`. It retains the 1,440 G sources,
architecture, seed, optimizer, crop/mask/noise streams, and 5,000-update budget,
while changing the diffusion target to established velocity prediction. This also
changes loss weighting; acoustic gains cannot be attributed only to arithmetic.
The final training sampler and Python/NumPy/CPU/CUDA random states match the
original run (`velocity_training_match.json`). No new hyperparameter search or
main-study evaluation was performed.

On the same four G diagnostic sources, full-precision gap NMSE falls from
5,180,206 with original terminal sampling to 1.028 with terminal velocity sampling,
without clipping. Clipping alone gives 22.549; the old truncated/clipped sampler
gives 1.842. The measured forward-state errors satisfy the derived amplification
identities. Reduced precision gives similar results. This isolates and demonstrates
a fix for the observed numerical failure.

All 192 existing V groups are reused, with the same gaps and candidate seeds.
The fixed first-draw Q decreases from 0.395 to 0.309 on tuning, 0.416 to 0.307 on
practice calibration, and 0.412 to 0.308 on reserved development. On the latter,
the paired difference is -0.103 (exploratory 95% group-bootstrap interval
-0.129 to -0.077); equal weighting of its 63 dates gives -0.104 (-0.132 to -0.077).
The baseline still has lower mean Q in every role. Tuning best-of-four opportunity
rises to 7/64 (10.9%), below the unchanged 20% trigger. No quality predictor or
release gate is transferred or recalibrated for the new model.

The correction is a successful numerical intervention with partial acoustic
improvement, not a successful selective deployment method. Three examples chosen
before the correction now show its first draw alongside the original selected
proposal and baseline. They include reduced excess energy and a case where level
improves but waveform error worsens. Fifteen clips preserve a common gain per case;
the original twelve audio files remain byte-identical.

The added training/evaluation took 29.2/7.3 minutes. GPU work finished at
2026-11-16 05:08:06 UTC, 8.32 hours after the authorized session began; the pod
was left running. New checkpoints and artifacts have verified local backups.

## What was learned

- The original terminal cosine sampler amplified noise-prediction errors into
  enormous waveform errors. The preserved G-only diagnostics motivated a bounded,
  truncated-start amendment before V candidate outcomes. Gaussian initialization
  at that nonterminal time remains approximate. No V endpoint sweep was performed.
- The 16-source learning checks lowered loss but did not establish near-zero
  memorization. Training and sampler limitations remain plausible causes of failure.
- In 100 hidden-event pairs, B, C1, C3, and P each produced exactly identical output
  for identical observations/seeds. All four missed all 100 inserted events at the
  fixed event-projection threshold, with no false-event detection on the absent
  partners. These 400 method comparisons are not 400 independent signal realizations.
- The frozen policy also completed 100 shifted V cases with zero acceptance. This
  verifies fallback behavior; it does not validate a risk bound under distribution shift.
- The released general-audio checkpoint ran on eight V cases after documented
  configuration, NumPy, and mask-edge compatibility corrections. The first completed
  run's Q is 0.65458 and gap SC is 1.25571. Its fixed-seed artifact replay differs
  (Q 0.65366); the largest per-case Q change is 0.02328. This legacy adapter is not
  bitwise reproducible in the recorded environment. Both runs and all failed attempts
  are preserved. Neither run is an exact reproduction or a superiority comparison.

## Scope and unresolved evidence

The planned three-seed training, actual C/test evaluation, C4 six-draw medoid, C5
192-step single draw, C6 waveform RePaint, independent classifier, external VLIZ
audio, and primary Holm-adjusted contrasts were not run. The five main success
criteria remain **untested**; observation consistency passed development checks
only. No classifier or external-generalization claim is supported.

At measured pilot throughput, just the configured conditional and deterministic
training caps across three seeds project to 28.31 hours, before other models,
calibration, inference controls, or external evaluation. That matrix does not fit
the user's ten-hour run. Both the original and corrected configurations remain
below the same feasibility criterion; the targeted correction fits the original
compute boundary.

Source grouping cannot prove independent environmental conditions: 1,332 dates
span partitions and native session/parent identifiers are missing. No exact decoded
or normalized-quantized duplicate or cross-partition timestamp overlap was found,
including unknown-hydrophone comparisons, but exhaustive time-shifted near-duplicate
search was not done. D/V lack pilot-vessel examples; C also lacks dredger; test lacks
dredger. The 8 kHz representation excludes frequencies at or above 4 kHz. Artificial
gaps and undeleted field references are not a paired clean/noisy benchmark.

The closest underwater spectrogram-inpainting publication's metadata were verified,
but its primary full text remains inaccessible. No first-underwater-diffusion or
first-inpainting claim is made. The manuscript is anonymous per user instruction
and targets ten total pages including references, explicitly presenting development
evidence. The user chose the upper end of the venue's published
eight-to-ten-page wording; organizer clarification and deadline timezone remain unresolved.

See `claim_to_evidence.json`, `experiment_ledger.json`, `compute_ledger.json`, and
`manuscript_audit/table_cells.json` for machine-readable evidence. The exact original
registries are retained, including the explicit launch-commit correction in
`provenance_corrections.json`. No results were filled in for unrun experiments.

## Additional local CPU agent demonstration

After the user reported the GPU unavailable, an installed Llama 3.1 8B Q4_0 model
ran one guided tool-use episode on the existing case C. No new GPU, remote API,
model download, or acoustic training was required. A startup HTTP 400 from the
older local tool schema is preserved; the first completed episode is the primary
draft, without selection among reruns.

The model requested a 128 ms analysis, configured the virtual display, and inspected
the recording in one batch. It correctly cited a measured 78.125 Hz bin but claimed
an unexecuted comparison and omitted verification. An explicit post hoc audit
continuation completed the missing 32 ms measurement and verified the 128 ms display.
All seven tool returns replay exactly; four model turns took 466.08 seconds of
CPU execution, with zero reported VRAM. Audio samples and the observation mask
were preserved. These are software display changes, not hardware commands.

The revised prose still omits the requested timing and identity qualifications and
leaves comparison wording unclear. Both drafts are retained. The operator evidence
card reports facts directly from verified tool returns and is not represented as
model-authored prose. This illustrates action checking and its limits on one known
case with explicit feedback, not autonomous reasoning performance or identification
accuracy. An earlier ten-page draft incorporated this trace. The active paper now
uses the replacement-pod GPU comparison below; the CPU episode remains historical
evidence and is not used as a hardware or model-performance control.

## Controlled GPU agent comparison

The user supplied an RTX PRO 4500 Blackwell and authorized additional agent
comparisons within a separate ten-hour run. All new neural inference uses
Qwen3-8B BF16 entirely on CUDA. There is no local language-model inference,
CPU offload, acoustic retraining, or external API. The model revision, complete
prompts, input arrays, masks, task definitions, executed tools, and raw responses
are retained in the GPU protocols and result folders.

Thirty-two V recordings are chosen by frozen hash order, one group and date each,
excluding the three published anecdotes. The four task families each contain
eight cases: frequency detail, timing detail, incompatible constraints, and
fragmented observations. The agent chooses the shortest feasible analysis window,
labels the measured cue, configures a virtual display when allowed, and verifies
the executed state. Cue labels are stipulated rules on measured spectra, not
object-identification annotations. The reference labels contain 16 stable,
15 insufficient, and only one changed case; 20 tasks permit configuration.

| Primary approach, n=32 | Correct setting and state | Correct cue | Complete task | Mean generated tokens | Median seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed rules | 32 | 32 | 32 | 0 | 0.0 |
| Direct agent | 12 | 16 | 10 | 238 | 9.8 |
| Reasoning agent | 18 | 25 | 4 | 3018 | 102.0 |
| Self-review | 16 | 21 | 8 | 3171 | 103.9 |
| Planner–reviewer | 19 | 19 | 9 | 3679 | 134.1 |

The fixed rules implement the supplied task policy and serve as a transparent
ceiling. All four neural workflows share weights, tool access, task instructions,
and total ceilings, but consume different amounts of compute. Thinking is disabled
only in the direct arm. Review begins at a draft or four turns; its planner receives
at most half the shared token allowance. The separate reviewer uses the same
weights with a fresh context. Neither review receives evaluator feedback.

The predefined success differences are reasoning minus direct −18.75 percentage
points (paired bootstrap 95% interval −37.5 to 0; Holm p=0.328125), self-review
minus reasoning +12.5 (0 to 28.125; Holm p=0.4375), and paired agents minus
self-review +3.125 (−15.625 to 21.875; Holm p=1). These small, bounded comparisons
do not establish a general ranking. Better cue labels alone do not establish
complete execution: post-configuration verification occurs in 5/20 feasible
reasoning episodes and 20/20 direct episodes. Both reviewing workflows reach
handoff without a draft in 23/32 cases; ten paired-agent episodes lack a valid
final report. Their effect includes completing unfinished work, rather than only
correcting existing answers.

The descriptive cue confusion counts are retained in the failure audit. Direct
mode labels 11/16 stable cues and 4/15 insufficient-evidence cases correctly;
reasoning labels 16/16 and 8/15, respectively. Both get the sole changed case
right. A single changed case cannot establish reliable detection of acoustic
changes. Cue accuracy includes inability to choose a feasible setting through
the stipulated insufficient-evidence rule; it is not an independent acoustic
recognition endpoint.

The primary audit independently replays all 888 tool returns, checks all 817 CUDA
model turns and their prompts, recomputes every score, and verifies unchanged
input hashes. All 160 episodes, including 128 neural episodes, are retained.
Peak allocated GPU memory is 17,259,933,184 bytes. The primary batch completed
at 2026-11-16 13:10:30 UTC. Both follow-ups below were frozen before their own
outputs, after specified primary-run observations, and are reported separately.

### Configuration-gate and reasoning-budget follow-ups

A first-case discrepancy motivates the configuration gate: the reasoner reports
the required 128 ms window but applies 64 ms. At an 8 Hz bin-spacing limit and
160 ms maximum support, 128 ms gives 7.8125 Hz bins and qualifies; 64 ms gives
15.625 Hz and does not. Sixteen additional V groups and dates, disjoint from the
primary tasks, receive paired original and guarded reasoning agents. The gate
rejects already-measured infeasible settings without supplying the answer, cue
label, or automatic verification. Rejection is guaranteed by construction; any
benefit in complete-task success must be measured separately.

The gate batch completed at 2026-11-16 14:04:41 UTC. Independent replay verifies
159 tool returns and 186 CUDA turns across all 32 paired episodes. The new set
contains three stable, one changed, and twelve insufficient-evidence labels, so
its absolute accuracy should not be pooled with the differently distributed
primary set.

| Gate follow-up, n=16 new groups | Correct setting and state | Correct cue | Complete task | Mean generated tokens | Median seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original reasoning | 8 | 9 | 2 | 3067 | 93.6 |
| Guarded reasoning | 11 | 9 | 3 | 3122 | 92.8 |

Infeasible applied changes fall from six, across six cases, to zero. The guarded
arm rejects nine requests; this protection follows from the tool's construction.
Complete-task success rises by one paired case: +6.25 percentage points, bootstrap
95% interval 0 to 18.75, exact McNemar p=1. There are no paired losses in this
sample, but the evidence does not establish a general completion improvement.
Unsupported structured reports occur in three original and four guarded episodes;
unfinished episodes number two and three, respectively. Rejecting an infeasible
change does not select the shortest feasible window, assign a cue, or complete
verification. The paired prompts and tool descriptions are unchanged, so the
model encounters the gate only through returned errors.

Both arms actually perform post-configuration verification in 4/10 feasible
tasks. The frozen score also rejects all citations to error returns. In guarded
cases 03 and 04, every other success condition passes, but each report cites a
genuine configuration-rejection message alongside its successful measurements.
These are not invented evidence IDs or fabricated numerical measurements. They
remain failures under the unchanged endpoint, and the descriptive audit identifies
them separately. This distinction limits interpreting every citation failure as
an unsupported factual assertion; no relaxed scores replace the frozen results.

The primary cap forces closure on 520 of 585 thinking-enabled model turns.
A second follow-up uses the first sixteen primary tasks, without selection by
their outcomes, and retains their original reasoning outputs as controls. It
increases thinking from 512 to 2,048 tokens per turn and the total allowance from
6,144 to 12,288; all other model, action, tool, sampling, and scoring settings
remain fixed. Both caps change together, and these are reused recordings. No
new independent groups are counted for this sensitivity analysis.

The larger-budget batch completed at 2026-11-16 15:03:01 UTC. It reduces forced
thinking closure from 79/90 turns (87.8%) to 21/77 (27.3%). Nineteen longer-budget
closures reach 2,048 thinking tokens; two reach the remaining total-token allowance
at 585 and 369 tokens. All original 79 closures occur at 512 tokens.

| Budget check, n=16 reused tasks | Correct setting and state | Correct cue | Complete task | Mean generated tokens | Median seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original reasoning | 8 | 13 | 3 | 3044 | 111.6 |
| Longer reasoning | 12 | 13 | 5 | 6278 | 187.2 |

The paired completion difference is +12.5 percentage points, with an exploratory
95% bootstrap interval of −18.75 to 43.75 and exact McNemar p=0.6875: four wins,
two losses, and ten ties. Two longer-budget episodes exhaust their token allowance;
all are retained. Unsupported structured reports decrease from four to three, and
stated/applied configuration mismatches decrease from one to zero. More thinking
reduces truncation and improves settings in this sample, but does not establish a
general completion gain. Both per-turn and total caps change, so the experiment
cannot attribute effects to just one limit.

The first frozen case now reports and applies 128 ms, labels its observed cue
stable, cites measured 39.0625/46.875 Hz peaks, and passes every success check.
The same recording under the original cap reports 128 ms but applies 64 ms.
This is a concrete corrected execution example, not an outcome-selected case or
an additional independent recording. Its spectra and stated/applied comparison
appear in the paper's GPU figure.

The budget audit replays 135 tool returns and 167 CUDA turns, including 73 returns
and 90 turns from the reused controls. Across the three phases there are **208
unique scored episodes**, including **176 neural episodes**, on **48 unique source
groups**. Their 1,109 tool returns and 1,080 CUDA turns are verified without
counting reused controls twice. Two smoke cases (eight neural episodes and two
rule episodes) are excluded from all scored outcomes. The new session finishes in
5.4005 hours from the authorized start; scored neural inference totals 17,946.83
seconds. Peak allocated memory is 17,433,020,416 bytes. All 450 closed remote output
and resource-log files are hash-verified locally. Fixed input and environment hashes
are retained in the phase freezes. The GPU is idle and the pod remains running.
