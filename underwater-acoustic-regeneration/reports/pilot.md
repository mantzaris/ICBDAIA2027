# Completed development pilot: no-go for the main study

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
the user's ten-hour run. The no-go decision, rather than another model search,
preserves the preregistered criterion and actual compute boundary.

Source grouping cannot prove independent environmental conditions: 1,332 dates
span partitions and native session/parent identifiers are missing. No exact decoded
or normalized-quantized duplicate or cross-partition timestamp overlap was found,
including unknown-hydrophone comparisons, but exhaustive time-shifted near-duplicate
search was not done. D/V lack pilot-vessel examples; C also lacks dredger; test lacks
dredger. The 8 kHz representation excludes frequencies at or above 4 kHz. Artificial
gaps and undeleted field references are not a paired clean/noisy benchmark.

The closest underwater spectrogram-inpainting publication's metadata were verified,
but its primary full text remains inaccessible. No first-underwater-diffusion or
first-inpainting claim is made. The manuscript is anonymous per user instruction,
eight total pages including references, and explicitly a development draft. The
venue's exact eight-versus-ten-page maximum and deadline timezone remain unresolved.

See `claim_to_evidence.json`, `experiment_ledger.json`, `compute_ledger.json`, and
`../manuscript/table_cells.json` for machine-readable evidence. The exact original
registries are retained, including the explicit launch-commit correction in
`provenance_corrections.json`. No results were filled in for unrun experiments.
