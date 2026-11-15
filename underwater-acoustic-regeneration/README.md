# Selective underwater waveform inpainting

This project tests whether observation-only probes and separately calibrated
acceptance reduce harmful generative gap completions. The task repairs known
missing samples in field recordings; those recordings are not noise-free truth.

The authoritative specification is
`ICBDAIA_2027_Codex_Execution_Prompt.txt`, with the accompanying research-plan PDF.
All project code, data, documentation, and generated artifacts belong inside this
directory. Work is committed only on `main`; the user performs all pushes.

## Current state

Preparation and dataset download are in progress. No real-data experiment,
calibrated policy, or manuscript performance claim is complete. See
`reports/status.json`. The user authorized up to ten hours for this run and asked
that the pod remain running afterward. The connected RTX 4090 has 24 GB VRAM;
`/workspace` is a 50 GB container overlay. Use staged data and verified local
backups rather than assuming that mount survives stopping the pod.

## Local verification

```bash
cd underwater-acoustic-regeneration
python3 -m pytest
PYTHONPATH=src python3 scripts/preflight.py
```

The recorded CPU environment in `requirements.lock` is for reproducibility of
these checks. A separate compatible GPU lock must be captured and tested on the
pod before training. Do not replace or upgrade its host driver.

## Research protocol

Use UniqueShip v1.0 (12 classes, five hours each) for the primary grouped study and
VLIZ v1 only for external reconstruction assessment. Source manifests and connected
vessel/recording groups precede resampling, crops, masks, or derived examples.
Generator training G, diagnostic fitting D, validation V, calibration C, and test
are distinct. Preserve the official test membership and purge conflicting sources.

Resample intact sources to 8 kHz, then crop 32,768 samples and simulate gaps.
Normalize from observed samples only, repeating the calculation for each probe.
The inference interface accepts an `Observation`, never a reference or class label.
Persist repaired audio separately from the observation and imputation mask.

Read `preregistration/protocol.json` before opening held-out outcomes. Calibration
returns an all-fallback policy if none of the validation-frozen thresholds passes
the multiplicity-corrected harm bound. Preserve that negative outcome.

## Outputs and limits

Raw audio, environments, candidate pools, and checkpoints are ignored by Git.
Keep checksums and retrieval manifests under version control; do not commit large
archives. Every reported result must trace to a completed entry in the run registry.
The final editable Word manuscript and rendered PDF depend on completed experiments,
verified template handling, and the author's confirmed author list/affiliations.
