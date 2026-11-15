# Execution decisions

2026-11-15: The user requires all work in this directory, commits only on `main`,
and no push. The repository had no commits and its unborn HEAD was `master`.
HEAD now targets `main`; no additional branch was created.

The user replaced the planning-dollar ceiling requirement with “try to keep under
10 hours for this run”, reported approximately 50 GB storage, and instructed that
the pod be left running. The first run is therefore bounded by time, with a
15-minute margin for saving results. No new paid resources may be provisioned.
The 60–120-hour full-study estimate is not a promise of completion in this run.

The GPU is an RTX 4090 with 24 GB VRAM. `/workspace` resolves to the container
overlay, not an independently persistent volume. Stream native audio from the ZIP
and cache only needed resampled sources. Back up manifests/results/checkpoints
locally with hashes. Do not assume the container survives stopping the pod.

UniqueShip's official website describes 20 kHz audio; the release README says
32 kHz, 32-bit mono, including occasional clips shorter than 4.096 seconds. Actual
headers determine preprocessing, and undersized clips cannot enter 4.096-second
evaluations. Missing background hydrophone IDs require grouping the same date
across all unknown devices, which is more conservative than splitting them.

The closest ICIGP paper's title/authors/DOI were reverified through Crossref. Primary
full text remains unavailable through the checked IEEE page. Do not infer detailed
limitations from its title, or claim generic underwater inpainting as new.

2026-11-15 completion: The user requested anonymous submission authorship. The
eight-page development manuscript uses Anonymous submission and clears identifying
template metadata. No author information remains pending for this draft.

The 5,000-update pilot failed the frozen opportunity trigger: 5/64 rather than
at least 20%, unchanged with 64-step sampling. Dry calibration selected all-fallback.
Main C/test outcomes therefore remain unopened. The G-only sampler correction,
extended learning check, 800 synthetic/shift cases, and eight published-checkpoint
compatibility cases were completed; all failed attempts and the transfer artifact
replay are retained. No additional architecture/training search was launched.

All research processes finished; the existing pod remains running per instruction.
The completed bounded pilot is not a completed main study or submission-ready
efficacy paper. Dollar charges and the container image digest remain unavailable;
neither is inferred. Local backups and the rendered anonymous draft are verified.
