# Rebuild, replay, or restart

Run commands from `underwater-acoustic-regeneration/`. Choose the smallest mode
that answers the question: compiling the paper does not require downloading data
or rerunning a model.

| Mode | What is needed | What it establishes |
| --- | --- | --- |
| Compile and check the paper | Tracked manuscript, figures, evidence records; Python, TeX, Poppler | The source builds and agrees with the retained paper records |
| Redraw the three current figures | Tracked plot inputs and production code; Matplotlib and numerical dependencies | The published plots can be regenerated without a GPU or full audio archive |
| Replay recorded agents | Tracked frozen NPZ inputs, tasks, JSON/JSONL traces; analysis dependencies | Tool measurements and scored outcomes can be recomputed without model inference |
| Recheck waveform repair | Preserved processed audio, candidate arrays, and checkpoint evidence in addition to tracked files | Saved reconstruction scores and diagnostic controls can be independently recomputed |
| Start a new experiment | Downloaded data/model weights or preserved checkpoints, a suitable GPU, and a new run record | A new execution, which must remain distinct from the reported study |

## 1. Compile and check the paper

The editable manuscript is `manuscript/main.tex`; `references.bib`, `llncs.cls`,
`splncs04.bst`, and its three figure PDFs are in the same directory. There are no
separate section files. The compiled `main.pdf` is committed alongside all manuscript
figure PDFs. TeX auxiliaries remain ignored.

Install `latexmk`, pdfLaTeX, BibTeX, NewTX, `booktabs`, `float`, `microtype`, and
`hyperref`, plus Poppler tools `pdfinfo` and `pdftotext`, using your TeX/system
package manager. Then run:

```bash
python3 scripts/build_manuscript.py
python3 scripts/validate_manuscript.py --paper-only
```

The builder compiles the existing text and writes `manuscript/main.pdf`; it does
not regenerate prose. Build auxiliaries stay in `.cache/latex/`. Build and
validation reports may change as local engines and PDF metadata change. The
paper-only check does not certify a fresh execution of the experiments or replace
human review before submission.

Direct compilation is also possible from `manuscript/`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

All PDF/PNG/SVG figure exports and their production code are retained. To redraw
the three figures used in `main.tex` from a fresh clone:

```bash
PYTHONPATH=src python3 scripts/rebuild_paper_figures.py
python3 scripts/build_manuscript.py
python3 scripts/validate_manuscript.py --paper-only
```

The compact `figures/data/acoustic_examples.npz` contains lossless reference and
gap samples for the three frozen examples, with provenance and signal hashes in
its JSON companion. Numerical correction plots use the tracked measured results;
the agent plot uses the retained tasks and executed traces. These rendering paths
do not rerun selection, training, or the full archived waveform audit. Original
production/audit scripts remain unchanged. See [the figure inventory](figures/README.md).
The recorded plotting environment is `provenance/word_environment.lock`; different
Matplotlib or font versions can change export bytes without changing the data.

## 2. Check code and replay recorded agents

The project requires Python 3.8 or later. `requirements.lock` records the original
local verification environment; GPU and transfer environments have separate lock
files. Full freezes are historical environment inventories, not universal
installers. Reuse an existing compatible PyTorch environment for the unit tests;
these checks do not launch the GPU experiment or language-model inference.

```bash
python3 -m pip install -e '.[test]'
python3 -m pytest -q
```

PyTorch is required for the model unit tests. On a new machine, install the
appropriate PyTorch build before running the suite. For GPU work, preserve the
pod's compatible CUDA/PyTorch installation and use `requirements.gpu.in` for the
remaining research dependencies.

The recorded agent inputs and traces are small enough to retain in Git. Replay
needs NumPy, SciPy, scikit-learn, and PyYAML, supplied by the base project install;
it needs no model weights or original archive downloads:

```bash
PYTHONPATH=src python3 scripts/analyze_gpu_agent_benchmark.py
PYTHONPATH=src python3 scripts/analyze_gpu_agent_guard_followup.py
PYTHONPATH=src python3 scripts/analyze_gpu_agent_budget_followup.py
PYTHONPATH=src python3 scripts/audit_gpu_agent_failures.py
python3 scripts/audit_gpu_agent_budgets.py
```

These scripts verify saved inputs and events, replay tools, and rewrite derived
summary/audit reports. All model failures remain included. They verify the saved
execution, not that a newly sampled language-model run would be byte-identical.

The frozen records include the earlier CPU demonstration for provenance. It is
not a matched hardware comparison, and replay does not authorize new local
language-model inference.

## 3. Restore full waveform artifacts

Git deliberately omits `data/raw/`, `data/processed/`, `checkpoints/`,
`results/pools/`, caches, and full source-manifest exports. All figure exports and
the compact inputs for the three current paper figures are included.
The current working machine retains these files. Preservation records are:

- `manifests/verified_raw_backup.json` and `manifests/uniqueship_archives.json`:
  original split-volume sizes and SHA-256 hashes.
- `manifests/verified_checkpoint_backup.json` and
  `manifests/verified_velocity_backup.json`: recorded checkpoint/candidate backups.
- `manifests/verified_gpu_agent_backup.json`: the copied GPU agent outputs.
- `figures/*.json`, `reports/velocity_followup_validation.json`, and recorded
  result rows: per-case source, candidate, and figure hashes.

Restore archived files to their recorded relative paths and verify their hashes
before rebuilding scientific evidence. Backup manifests identify files; they are
not downloadable copies of those files. If a local checkpoint or candidate pool
is lost, it must be recovered from another backup or regenerated by a new GPU
execution. A model seed alone is not a promise of bitwise reproduction.

With the full artifacts restored, the reconstruction and paper checks include:

```bash
PYTHONPATH=src python3 scripts/posthoc_pilot.py
PYTHONPATH=src python3 scripts/build_acoustic_examples.py
PYTHONPATH=src python3 scripts/plot_velocity_followup.py
PYTHONPATH=src python3 scripts/plot_gpu_agent_benchmark.py
python3 scripts/build_reports.py
python3 scripts/build_manuscript.py
python3 scripts/validate_manuscript.py
```

Plot regeneration additionally requires Matplotlib and the figure environment
recorded in `provenance/word_environment.lock`; acoustic example export uses
SoundFile. The full validator deliberately requires the archived evidence and
generated exports, so it is not the fresh-clone smoke check. Regenerating figures
can change file bytes with different plotting-library versions even when the
underlying results are unchanged; retain the frozen tracked PDFs for the paper.

## 4. Recover data and restart experiments

Authoritative data identities are in `configs/data.yaml`,
`manifests/uniqueship_archives.json`, and `provenance/DATA_ATTRIBUTION.md`. The
executed source is **UniqueShip v1.0, 12 Class – 5 Hours Each**, available from the
[official release folder](https://drive.google.com/drive/folders/1HxV8AGh8ckZS3JehAU_gY2fuCnscX-eV).
The two original volumes total about 17.8 GB. Joining them also needs space for a
second full archive; budget storage before combining archives and model weights.
The source recordings and distributed derivatives retain their CC BY 4.0
attribution to the dataset authors and Ocean Networks Canada.

On the research machine with the data dependencies installed, recovery starts with:

```bash
python3 scripts/download_uniqueship.py
zip -s 0 data/raw/uniqueship_v1/allclass_5hours.zip --out data/raw/uniqueship_v1/joined.zip
PYTHONPATH=src python3 scripts/prepare_data.py data/raw/uniqueship_v1/joined.zip --cache-pilot
```

`download_uniqueship.py` uses `gdown`; `prepare_data.py` uses SoundFile and the
project's numerical dependencies. The downloader rewrites its archive record;
compare the new hashes with the committed version before accepting a replacement.
Preparation reconstructs `manifests/sources.jsonl.gz`,
`manifests/source_metadata.jsonl.gz`, the CSV extracts under
`provenance/uniqueship_metadata/`, and the processed pilot audio. Those full exports
are reproducible and ignored; `manifests/pilot_sources.jsonl.gz` remains tracked as
the compact record of selected sources. Preparation writes manifests, so preserve
the frozen originals and compare hashes when recreating the historical selection.

Use the exact commands, seeds, configurations, and statuses in
`results/run_registry.jsonl` and `results/local_run_registry.jsonl` to inspect the
completed sequence. The restoration sequence was conditional/deterministic pilot
training, the frozen sampler amendment and candidate-pool evaluation, then the
matched velocity follow-up. Scripts include `train.py`, `make_pools.py`,
`analyze_pilot.py`, and `evaluate_velocity_followup.py`. `--resume` in `train.py`
requires the corresponding preserved checkpoint; downloading source audio does
not recover trained weights.

For the agent model, `provenance/gpu_agent_environment.json` records Qwen3-8B's
pinned revision, weight hashes, libraries, and CUDA residency. The setup recipe is
`scripts/setup_gpu_agents.sh`, which assumes the remote project path
`/workspace/underwater-acoustic-regeneration` and downloads weights under
`.local/models/`. The frozen inputs already in Git can be reused; their preparation
scripts refuse to overwrite existing manifests. Model inference was GPU-only on
the RTX PRO 4500, and new inference should remain on a suitable remote GPU.

Historical job launchers enforce dated deadlines and existing completion markers.
The authorizations in `configs/run_authorization.json` and
`configs/gpu_agent_authorization.json` describe completed sessions; they are not
permission or a working launch configuration for a new run. Create a new run
identifier, current compute limits/deadline, and separate output paths/protocol
before launching new work. Several historical follow-up scripts use fixed paths;
adapt and refreeze them for a new experiment instead of deleting old markers or
overwriting the retained evidence. `configs/essential.yaml` describes future work,
not an implemented full-study launcher.

The legacy general-audio transfer artifacts have separate pinned retrieval URLs
and hashes in `provenance/transfer_artifacts.json`, with `fetch_transfer.py` and
`requirements.transfer.lock`. The VLIZ dataset in `configs/data.yaml` was not used
in the reported experiments and is unnecessary for rebuilding this paper.

## Keep future commits small

New experiment outputs, runtime logs, and model/data downloads are ignored by
default. Paper PDFs, figure exports, plotting inputs, and production code must be
committed. The existing frozen result files and selected
audit logs are explicitly retained; adding another run requires choosing which
compact protocol, input, outcome, and provenance records are worth versioning.
Keep the three active figure PDFs beside `main.tex` and retain attribution with
the short WAV companions. Before committing, run:

```bash
python3 scripts/check_repository_size.py
git status --short
```

The size check limits the tracked snapshot to 50 MiB total and 2 MiB per file;
it does not measure older history. Avoid force adding caches or bulk data.
Removing files from the current tree does not itself remove their older versions
from Git history.

The cleanup record is `reports/repository_storage.json`. A verified complete copy
of the pre-cleanup history is kept locally at
`.local-history/before-repository-cleanup.bundle`, which is ignored by Git. This
bundle includes the earlier versions and commit IDs referenced in the frozen
research records. Keep a separate backup of it if that history is needed; it does
not include data or checkpoints that were never committed. To inspect the old
history without changing the active checkout:

```bash
git clone .local-history/before-repository-cleanup.bundle .cache/original-history
```
