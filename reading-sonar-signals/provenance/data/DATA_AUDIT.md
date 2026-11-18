# Real recording acquisition and limitations

Acquisition completed November 17, 2026 UTC (November 16 America/New_York).
The frozen continuous benchmark contains **210 untouched 30-second real scenes**:
150 DeepShip excerpts and 60 MBARI excerpts. DeepShip contributes 90 training,
30 validation, and 30 internal-test scenes. All 60 MBARI scenes are external test.
There are 26 training, 9 validation, and 69 test recording/day groups.

The frozen manifest is `data/manifests/real_scenes.jsonl`, SHA-256
`611c242772f0225a024bc93e686681b2313b7ec500359103aeb748e0649e2bad`.
Every file was decoded; every excerpt is a bit-exact copy of source sample values,
without normalization, resampling, mixing, or concatenation. Source and excerpt
hashes and offsets are in the manifests. Exact duplicate excerpt PCM count is
zero. A GPU near-duplicate audit is a separate required analysis.

## DeepShip

The [author repository](https://github.com/irfankamboh/DeepShip) was downloaded at
revision `77725726a33d6aef4ce1d0fe70f67c0bbee4b5b6`. Its 63 public WAV files contain
5,723 seconds of mono 32 kHz float32 audio. Only 44 original files contain at least
one full 30-second scene. Together they support 162 nonoverlapping scenes;
150 were selected using a fixed parent-file split and a duration-only sampling
rule. This public subset is not the complete 47-hour corpus described by the
authors. Full access requires contacting the provider; no contact was made.

The public subset does not supply verified per-file vessel IDs, session IDs, or
recording dates. Splits are **original-file-disjoint**, not vessel-disjoint.
Grouping at a stronger unobserved level cannot be claimed. The public repository
does not provide an explicit audio redistribution license. Acquired audio remains
outside the Git release; this audit makes no redistribution-rights claim.

No source files were copied from the sibling regeneration project. Because both
DeepShip and that project's UniqueShip data originate from Ocean Networks Canada,
underlying recording overlap has not been ruled out. The studies' raw provenance
must not be described as wholly independent merely because their folder names
differ. MBARI is the genuinely separate marine recording domain in this study.

## MBARI Pacific Sound

The [provider page](https://www.mbari.org/project/open-acoustic-data/) documents
continuous acoustic recordings and provider-produced lower-rate releases. The
[AWS Open Data registry](https://registry.opendata.aws/pacific-sound/) specifies
CC BY 4.0 and the `pacific-sound-16khz` bucket. Attribution: MBARI Pacific Ocean
Sound Recordings, accessed November 17, 2026, via that registry; provider header
requests citation of DOI `10.1109/OCEANS.2016.7761363`.

Sixty days were spread across available 2016 daily objects by a date-only rule.
One 30-second interval per day was selected by a key-hash rule before inspecting
audio. HTTP 206 ranges retrieved the exact 24-bit PCM payload; source ETags,
headers, byte ranges, source offsets, and excerpt hashes are retained. Full daily
objects were not downloaded, so no full-object SHA-256 is claimed.

Two initial excerpts were all-zero PCM. A fixed hourly advance (wrapping within
the same day) selected the first nonzero interval. June 6 required 15 rejected
offsets before 66,540 s; November 19 required one before 65,820 s. These integrity
exclusions and all rejected offsets are stored in `sources.jsonl`. No class,
event, or model score was used to choose replacement content.

All MBARI days belong to the external-test domain. Day-level grouping does not
claim different hydrophones or deployments. Native 16 kHz audio supports the
0.5–2 and 2–5 kHz carrier bands; a full 5–8 kHz carrier is unavailable with the
required transition margin. The experiment uses unchanged digital amplitudes,
not pressure-calibrated SPL.

## QiandaoEar22 short-window audit

The [authors' repository](https://github.com/xiaoyangdu22/QiandaoEar22) links an
anonymously accessible SharePoint release. Byte access was verified. Twelve
bounded clips, four from each target subdataset, were downloaded and decoded;
they are 3 seconds, mono 52,734 Hz integer32 WAVs. Exact original IDs, filenames,
provider metadata, and hashes are stored separately in
`data/manifests/qiandao_short_windows.jsonl`.

These clips do not satisfy the 30-second continuous-scene requirement. Segment
indices were not verified as original sample offsets, so no clips were
concatenated. Label-bearing filenames are withheld from models. Provider target
labels are not component boxes or proof of each physical sound source. Explicit
audio redistribution terms were not found; the clips stay outside Git.

## Protocol changes and annotation status

The original 360-real-scene feasible plan was reduced to 210 rather than weaken
recording independence or construct fictitious continuous scenes. Whole DeepShip
access and provable Qiandao reconstruction were unavailable in this execution.
Naturally overlapping real components have not been independently verified.

The user subsequently explicitly required AI-generated annotations because human
annotators will not be available. The annotation package and code therefore use
separate blinded AI reference passes, disclose their origins, preserve
disagreements, and evaluate reference errors on exact synthetic controls.
Real-data agreement with these labels is not independently verified physical
accuracy. AI consensus cannot certify exhaustive negative regions. The original
research outline remains unmodified as the historical proposed protocol.
