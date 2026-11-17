# Paper figures and production files

All figure exports (PDF, PNG, and SVG) are versioned. The compiled paper, LaTeX
source, bibliography, publisher styles, and colocated PDF figures are in
`../manuscript/`. No GPU is needed to rebuild the paper or redraw its current figures.

From the project directory:

```bash
PYTHONPATH=src python3 scripts/rebuild_paper_figures.py
python3 scripts/build_manuscript.py
python3 scripts/validate_manuscript.py --paper-only
```

| Current paper figure | Original production code | Committed plotting inputs |
| --- | --- | --- |
| `gpu_agent_comparison` | `scripts/plot_gpu_agent_benchmark.py` | Frozen tasks, observed audio, responses, tool traces, and summaries in `results/gpu_agent_*` |
| `velocity_correction` | `scripts/plot_velocity_followup.py` | Measured numerical controls in `reports/` and per-case results in `results/velocity_followup/` |
| `acoustic_case_studies` | `scripts/build_acoustic_examples.py` | `figures/data/acoustic_examples.npz` and its JSON provenance, plus the original selection record |

`rebuild_paper_figures.py` reuses the original acoustic and agent renderers and
reproduces the correction plot from its saved measurements. It runs each figure
in a separate process to preserve its plotting style. The acoustic bundle stores
the original floating-point reference samples and generated gap samples; all full
waveforms are reconstructed losslessly and verified against signal hashes. These
are previously selected examples, not newly generated results. Attribution and
source licenses remain in `provenance/DATA_ATTRIBUTION.md`.

The original scripts remain available for the full waveform and source audit.
`export_paper_plot_inputs.py` can rebuild the compact acoustic bundle from those
locally preserved archives; it is not needed for routine figure regeneration.
The original Matplotlib/NumPy/SciPy versions are recorded in
`provenance/word_environment.lock`. Other versions may alter export bytes.

Earlier presentation figures are also retained: `method`, `development_ranking`,
and `development_examples` come from `scripts/plot_pilot.py`; `acoustic_context`,
`operator_configuration`, and `operator_decisions` from
`scripts/build_acoustic_examples.py`; and `agent_episode` from
`scripts/build_agent_figure.py`. They are historical illustrations rather than
additional figures in the current ten-page paper. Some original audit paths need
the full local waveform/candidate archives described in `REPRODUCING.md`.
