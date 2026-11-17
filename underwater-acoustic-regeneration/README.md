# Checking sound and action: generative AI for underwater acoustic analysis

This project studies two practical checks for acoustic AI: whether generated audio
preserves useful sound structure, and whether an assistant actually applies and
verifies the analysis setting it reports. The anonymous paper combines a controlled
repair diagnosis, three acoustic examples, and a GPU comparison of four agent
workflows with fixed rules.

The repository keeps the paper sources, code, pinned configurations, frozen
experimental inputs and outcomes, and evidence needed to inspect the results. Large
downloads, model weights, training checkpoints, and full candidate arrays are kept
outside Git. The compiled paper, all figure exports, figure-production code, and
compact inputs needed to redraw the three current figures are included. See
[REPRODUCING.md](REPRODUCING.md) for paper compilation, recorded-agent replay, and
restarting experiments.

## Paper and evidence

- [Single LaTeX source](manuscript/main.tex), [references](manuscript/references.bib),
  and three colocated figure PDFs, plus the [compiled paper](manuscript/main.pdf).
  The paper is anonymous and ten pages including references.
- [Figure inventory and production commands](figures/README.md), including PDF,
  PNG, and SVG exports and lossless plotting inputs for the acoustic examples.
- [Claim-to-evidence map](reports/claim_to_evidence.json),
  [table-cell sources](reports/manuscript_audit/table_cells.json),
  [experiment ledger](reports/experiment_ledger.json), and
  [compute ledger](reports/compute_ledger.json).
- [Repair protocol](preregistration/protocol.json),
  [sampler amendment](preregistration/sampler_amendment.json),
  [correction protocol](preregistration/velocity_followup.json), and
  [paired correction results](results/velocity_followup/summary.json).
- [GPU comparison protocol](preregistration/gpu_agent_benchmark.json),
  [scoring amendment](preregistration/gpu_agent_scoring_amendment.json),
  [configuration-gate follow-up](preregistration/gpu_agent_guard_followup.json), and
  [thinking-budget follow-up](preregistration/gpu_agent_budget_followup.json).
  Frozen tasks, observed inputs, raw responses, executed tool returns, and results
  remain under `results/gpu_agent_benchmark/`, `results/gpu_agent_guard/`, and
  `results/gpu_agent_budget/`, including failed episodes.
- [Fifteen listening clips](examples/acoustics/README.md),
  [example provenance](figures/acoustic_example_selection.json), and
  [data attribution](provenance/DATA_ATTRIBUTION.md).

## What the experiments show

The original repair pilot failed its predeclared feasibility trigger: only 5 of
64 tuning groups had a candidate improving error Q by at least 0.02, against a
required 20%. Its release rule accepts no generated fills. A matched change to
velocity prediction removes a numerical failure: full-precision gap NMSE on four
diagnostic sources falls from 5.18 million to 1.03. On 64 reserved development
cases, first-draw Q falls from 0.412 to 0.308, about 25%, while the learned baseline
remains better at 0.239. Correction raises the tuning opportunity to 7/64, still
below the feasibility threshold.

On 32 acoustic tasks, complete-task counts are 10/32 for direct Qwen3-8B action
selection, 4/32 for reasoning, 8/32 for self-review, and 9/32 for a planner–reviewer
pair; fixed rules complete 32/32. None of the three predefined paired contrasts
is conclusive after adjustment. A tool gate prevents six infeasible applied
changes on sixteen new groups, but completion changes only from 2/16 to 3/16.
On sixteen reused tasks, a larger thinking budget improves setting correctness
from 8/16 to 12/16 and completion from 3/16 to 5/16; the completion gain is
inconclusive and costs more tokens and time.

All three GPU phases finished in 5 hours 24 minutes: 208 scored episodes,
including 176 neural episodes, on 48 unique recording groups. Replay audits verify
1,109 unique tool returns and 1,080 CUDA turns. The
[completion record](reports/gpu_agent_remote_completion.json) records that the
jobs ended and the pod was left running.

These results distinguish sound quality, cue labeling, configuration, and verified
completion. They do not evaluate vessel identification or physical equipment
control. The correction and agents use development recordings; actual calibration
and final-test outcomes remain unopened, and the original confirmatory restoration
study is incomplete.

## Build and check

Paper compilation needs Python and a TeX installation with `latexmk`, pdfLaTeX,
BibTeX, NewTX, `booktabs`, `float`, `microtype`, and `hyperref`. Validation also uses
Poppler's `pdfinfo` and `pdftotext`.

```bash
python3 scripts/build_manuscript.py
python3 scripts/validate_manuscript.py --paper-only
```

To regenerate all three current figures from committed inputs, run
`PYTHONPATH=src python3 scripts/rebuild_paper_figures.py` before building the paper.
This needs Matplotlib and the numerical dependencies, but no GPU, model weights,
or full source archive. Original research and figure-audit scripts remain included.

These commands work from a fresh clone without audio downloads or a GPU. The
paper-only check verifies the manuscript and retained evidence; it does not repeat
the archived waveform audits. The default full validator additionally requires
the locally preserved reconstruction artifacts. Unit tests and recorded-agent
replay are described in [REPRODUCING.md](REPRODUCING.md).

## Storage and future work

The `.gitignore` policy keeps future runs and bulk research artifacts out of Git by
default. Existing frozen results are retained explicitly so that an ordinary
`git add` does not add every new experiment. Manuscript files, all figure exports,
plotting inputs, and the fifteen short WAV examples are intentional deliverables.

Before committing, run `python3 scripts/check_repository_size.py` to check the
tracked snapshot against the 50 MiB total and 2 MiB per-file limits. This checks
the current files, not older Git history.

Files removed from tracking during repository cleanup remain on the current
machine. Raw archives, processed audio, checkpoints, and candidate arrays also
have local preservation records in `manifests/`; those records are not a remote
backup. Keep these local files if exact restoration replay or checkpoint resumption
is needed. Rebuilding the paper and replaying the recorded GPU agents do not require
those large artifacts.

Recorded authorizations apply to the completed GPU sessions. A new experiment
needs a new run identifier, current compute limits, and a separate frozen record;
do not overwrite completed results. No resource is provisioned or job launched by
the paper build. Work is committed on `main`; the user handles pushing.
