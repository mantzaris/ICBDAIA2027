# Data attribution and audit scope

Primary release: **UniqueShip v1.0, 12 Class – 5 Hours Each**. Authors: Connor
Hashemi, Trevor Stout, Anthony Hoogs, and Jason Parham. Cite *UniqueShip: Mitigating
Data Leakage in Acoustic Ship Classification Benchmark Datasets*,
[arXiv:2609.13659](https://arxiv.org/abs/2609.13659), 2026, and
[Ocean Networks Canada](https://www.oceannetworks.ca/data/data-policy/).
The [official release site](https://uniqueshipdata.org/) states CC BY 4.0 for the
ONC-owned recordings. Archive identities and computed checksums are in
`manifests/uniqueship_archives.json`; the release README is preserved here.

All 43,208 source WAV headers and ZIP CRCs were checked. The files are mono
32 kHz PCM_32, despite the website's 20 kHz description. Twelve sources are shorter
than 4.096 seconds and are excluded from the pilot cache. File SHA-256 retains the
exact source bytes. The decoded-content hash uses little-endian float32 mono;
gain/offset-normalized waveforms are additionally quantized at 1e-4 and hashed.
These checks found no exact decoded duplicates. They do not exhaustively exclude
time-shifted or overlapping recordings.

The fifteen clips in `examples/acoustics/` are short derivatives of three UniqueShip
recordings owned by Ocean Networks Canada, under the release's CC BY 4.0 terms.
Credit the dataset authors above and ONC when sharing these audio examples.
Changes include resampling to 8 kHz, a 4.096 s crop, observed-context normalization,
and a shared presentation gain per case. Erased-input clips additionally zero the
gap; reconstructed clips replace that gap with frozen original or velocity-follow-up
outputs. All five versions of each case use the same presentation gain. They are
16-bit PCM listening aids, not calibrated sound-pressure measurements.
`figures/acoustic_example_selection.json` identifies original archive members,
recording times, hydrophone, gap samples, transformations, and checksums. These
derivatives retain their source-data attribution; the project-code license does
not replace it.

The release omits native parent recording/session IDs. Grouping uses supplied MMSI,
clip identity, available duplicate hashes, and background hydrophone/day. Missing
background device IDs are grouped together across devices for a given date.
Recorded source-group counts are not proof of statistical independence; shared
recording conditions remain a limitation and require date-block sensitivity analysis.

External release, not yet downloaded: **VLIZ AIS-annotated Hydrophone Recordings
for Vessel Classification, v1**, [Zenodo 17233667](https://zenodo.org/records/17233667),
data DOI [10.14284/723](https://doi.org/10.14284/723). The record metadata confirms
CC BY 4.0, 26,005,892,324 archive bytes, and MD5
`8098056c952ff706bdf1adaab7147096`. External evaluation remains unperformed.
