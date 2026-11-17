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

The user subsequently requested LaTeX only: one `mainj.tex`, a separate `.bib`,
and figures in that same directory. The active paper now uses the unmodified
official LNCS class/style with its figures in `manuscript/`. Word output is retired
from the working tree and retained in Git history. The scientific content and
anonymous authorship are preserved; this conversion runs locally without GPU work.

At the user's request, the paper now foregrounds hydrophone acoustics with an
actual waveform/spectrogram overview and three explicitly selected anecdotal
cases: cargo (small aggregate improvement with worse spectral convergence),
other vessel (excess gap energy), and tanker (loss of a low-frequency component).
The manuscript remains eight pages and uses one `mainj.tex` with a separate `.bib`.
Twelve 8 kHz listening clips provide reference/input/B=P/C3 comparisons with a
shared presentation gain per case. Their source and pool hashes, recomputed scores,
gap masks, and PCM observed-sample consistency are verified. The example selection
is post hoc and adds no independent sample, inference, or GPU work. Veirs et al.
(PeerJ, 2016) supplies primary hydrophone-measurement context for the introduction.

The user asked for language and examples that connect with acoustic/sonar operators,
especially correct object labels and equipment configuration. The title, abstract,
keywords, method labels, and anecdotes now use the sequence inspect, label/unknown,
recommend settings, approve/configure, and remeasure. Generative AI describes the
existing diffusion model; agentic review and physical equipment control are explicitly
proposed extensions. A new plot compares 32/4 ms and 128/16 ms window/hop settings
on observed input, excluding every frame touching the artificial gap. It distinguishes
the supplied tanker annotation from the measured frequency component and from any
unperformed identification. A second plot shows the 52/64 worse proposals and the
actual disabled release gate. ReAct is cited for the future tool-use concept.
No new inference, training, object predictions, hardware actions, or operator trials
were performed. All twelve listening clips retain the same audio samples.

The user subsequently requested a ten-page version, one to three additional
equations, and a measured correction of the identified failure. A single
velocity-prediction training intervention was frozen before its outcomes in
`preregistration/velocity_followup.json` and committed on main. It retains the
original architecture, data, seed, stochastic training streams, optimizer,
batch size, and 5,000-update budget. One terminal sampler and one sampler matching
the old amendment are compared on the same 192 V groups. Actual C/test outcomes
remain unopened, and the original learned release gate is not reused for the
new generator. The original no-go result and all failed sampler controls remain
part of the record. No additional pod or paid service was provisioned; the
existing ten-hour deadline and instruction to leave the pod running still apply.

2026-11-16 follow-up completion: Matched training and 192-group evaluation finished
at 05:08:06 UTC, 8.32 hours after the authorized session start. The added training
and evaluation took 29.2 and 7.3 minutes. Velocity prediction reduced full-precision
gap NMSE on the four original G diagnostic sources from 5.18 million to 1.03,
without truncation or clipping. Reserved-development first-draw Q fell from 0.412
to 0.308; baseline Q remains 0.239. The unchanged tuning feasibility trigger still
fails (7/64 versus 20%), and no new gate is calibrated. The original no-go evidence
is preserved alongside the demonstrated numerical correction.

The paper now has ten pages including references, three added equations (five
total), a before/after correction figure, and corrected outputs in the same three
previously selected acoustic cases. Fifteen listening clips use a shared gain per
case; the original twelve are byte-identical. All 1,536 new velocity candidate
metrics were independently recomputed, 206 backup files matched remote SHA-256
checks, and the training-control audit passed. The GPU is idle and the pod remains
running. No main-study, object-identification, equipment-control, or reasoning-agent
capability is claimed.

2026-11-16 CPU follow-up: The user reported the GPU unavailable and asked whether
the agent demonstration required another pod. The existing local Llama 3.1 8B
Q4_0 model and masked recordings were sufficient. No remote connection, new model
download, paid API, or acoustic retraining was used. The task, model digest, tools,
and limits were frozen on main. Ollama 0.3.2 initially rejected numeric schema
enum values before inference; the error and transport-only amendment are retained.

The first completed episode used two model turns and three batched tools, selected
128 ms, and measured a 78.125 Hz peak bin at 7.8125 Hz spacing. Its draft falsely
claimed a comparison without measuring 32 ms and omitted verification. A separately
frozen post hoc audit continuation added four tool calls, including the missing
comparison and verification. Total model/tool execution took 466.08 seconds on CPU
with zero reported VRAM. The revised prose still lacks requested qualifications
and is not released as a verified operator report. A structured evidence card is
assembled directly from tool returns. All seven tool results replay exactly.

Figure 1 now shows this actual agent/audit sequence and the two observed-only
spectrograms. The paper remains ten pages, with sixteen references and five
equations. The one-case, guided nature and explicit corrective feedback are disclosed;
no autonomous success rate, object-identification accuracy, or hardware-control
claim is added. Original GPU study completion and its ten-hour boundary are unchanged.

2026-11-16 replacement-pod GPU completion: The user authorized comparisons on an
RTX PRO 4500 Blackwell and excluded new local language-model inference. A frozen
32-recording comparison evaluates fixed rules and four Qwen3-8B workflows: direct,
reasoning, self-review, and a planner–reviewer pair. The three predefined completion
contrasts are inconclusive after Holm adjustment. Separate protocols test a
configuration feasibility gate on sixteen new groups/dates and a larger thinking
budget on the first sixteen primary tasks. No favorable reruns or omitted failures
replace the original scores. The gate prevents six infeasible applied changes;
completion changes from 2/16 to 3/16. Longer thinking improves setting correctness
from 8/16 to 12/16, but the 3/16 to 5/16 completion change is inconclusive and costs
more tokens and time. The first frozen case's stated/applied discrepancy is corrected
under the larger budget and every success check passes for that case.

All 208 scored episodes (176 neural; 48 unique groups) finish in 5.4005 hours.
Independent audits replay 1,109 unique tool returns and verify 1,080 CUDA turns;
450 closed remote outputs/resource files are backed up with matching hashes.
The GPU is idle; the pod remains running. The paper replaces its historical CPU
illustration with the controlled GPU results, retains the numerical correction and
three acoustic anecdotes, and uses nineteen references. The frozen citation rule
continues to reject two genuine guard-error citations; this limitation is disclosed,
not silently rescored. A figure metadata variable was corrected during final review
to retain the first case's cue label after plotting other cases; raw results and
scores were unchanged. All code, data, and manuscript work stays in this project;
commits remain on main and nothing is pushed.

2026-11-16 readability and disclosure revision: Rewrote the abstract, opening,
methods explanations, acoustic anecdotes, and conclusion around the operator’s
need to distinguish plausible output from checked evidence. The contribution is
controlled diagnosis and targeted intervention in these acoustic applications;
established algorithms and inconclusive comparisons remain explicit. All 110
table values and five equation blocks are unchanged, and no experiment was run.
The anonymous paper remains ten pages with nineteen references. Current conference
and Springer policy sources were checked; a first-page note, end declaration,
figure/table caption notices, and an accompanying audio note disclose substantial
Codex assistance. The publisher guidance PDF is retained with its checksum.
The homepage’s explicit 8–10-page range replaces the earlier uncertainty about
whether ten pages were allowed. Human author review is still required before
submission; no such review, organizer approval, or acceptance is claimed.

2026-11-16 end-only disclosure placement: At the user's explicit request, removed
the title footnote and all six figure/table preparation notices. The paper now
contains only one short AI Use Declaration after the conclusion and before the
references, kept together on page ten. Scientific prose, equations, results, and
figure data are unchanged. The single-source LaTeX paper remains anonymous and
ten pages; compilation, manuscript validation, and visual inspection passed.
Policy records distinguish this chosen placement from the publisher guidance;
no claim of publisher approval or submission is made.

2026-11-16 repository preservation cleanup: Untracked 39 rebuildable files (32.14 MB)
without deleting working copies. Retained the paper sources and three active
figures, code/configurations/tests, compact audit records, all frozen agent inputs
and traces, and acoustic companions. New results, bulk data/models, generated
exports, and build outputs are ignored by default. Added a restart guide and a
tracked-file size check. Updated build/validation scripts for the user's main.tex
rename and added paper-only validation for fresh clones. An isolated copy containing
only retained files passed 59 tests, built the ten-page paper, and reproduced all
three agent analyses and both failure/resource audits byte-for-byte. Full local
artifact validation also passed. A verified ignored bundle preserves the original
Git history. No data or model inference was run and nothing was pushed.

2026-11-16 complete paper asset preservation: At the user's explicit request,
restored the compiled manuscript, all thirty PDF/PNG/SVG figure exports, and the
previous manuscript figure PDFs to tracking. All original production scripts
remain unchanged. Added a lossless 0.81 MB acoustic plotting bundle and a portable
renderer for the three current figures, so a fresh clone needs neither a GPU nor
the full audio/candidate archives to redraw them. All nine regenerated current
figure exports match the existing files byte-for-byte. The isolated paper build
and paper-only validation pass; full local artifact validation also passes. The
scientific text is unchanged; only its build-command comment was corrected. The
tracked-size budget is 50 MiB with a 2 MiB per-file limit, and the size checker now
requires every inventoried paper asset to remain tracked. Bulk models/data and
TeX intermediate files remain ignored.
