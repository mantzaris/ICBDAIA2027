Dataset Version = 1.0

# UniqueShip — packaged bundle for the "12 Class - 5 Hours" split, where all classes are represented with 5 hours each.

5-second hydrophone recordings from Ocean Networks Canada for the UniqueShip dataset, labelled by vessel
class from co-located AIS, with K-fold train/validation/test splits.

Please visit website (https://uniqueshipdata.org/) for more information, paper, and how to cite. Also, please visit Oceans 3.0 Data Portal (https://data.oceannetworks.ca/) for access to the raw data from ONC. The paper contains and cites the exact deployments used in UniqueShip.

## Layout

```
<bundle>/
    audio/
        tug/  cargo/  tanker/  passengership/  background/   # one dir per class
            20220412T200340.000Z_20220412T200345.000Z.wav    # <start>_<end>, UTC
    fold_1/  …  fold_5/
        metadata_5s_train.csv
        metadata_5s_validation.csv
        metadata_5s_test.csv
```

Audio is stored once under `audio/` and shared by all folds. Each fold is
just three CSVs selecting rows from it.

## Audio

Mono WAV, 32 kHz, 32-bit, 5.0 s per clip (a few are short, min 4.0 s).
`background` = hydrophone segments with no vessel present.

## Metadata CSVs

One row per clip, 42 columns. `path` is relative to the CSV's own directory:

```python
df = pd.read_csv("fold_1/metadata_5s_train.csv")
wav = os.path.join("fold_1", df.loc[0, "path"])   # ../audio/tug/2022….wav
```

Core columns:

| column | meaning |
| --- | --- |
| `label` | class name, matches the `audio/` subdirectory |
| `path` | relative path to the WAV |
| `duration_sec` | clip length (≈5.0) |
| `sample_rate` | 32000 |
| `date` | `YYYYMMDD` (UTC) |
| `MMSI` | vessel identity; **0 for background** |
| `hydrophone` | recording device id |

The rest are optional context, carried through from AIS and CTD and **often
missing** — check for nulls before relying on them. Background rows have `MMSI = 0` and no AIS or hydrophone fields.

## Splits

5 folds, roughly 80/10/10 train/validation/test by duration.

Splits are grouped, not random: every clip from one vessel (`MMSI`) stays in a
single split, and background is split by whole day. No vessel appears in more
than one fold's test set, so the folds can be treated as independent evaluations
of unseen ships. Each vessel is assigned one dominant class, so a ship never
contributes to two labels.


## Unzipping
To unzip files into one folder, please use following command:

```
zip -s 0 allclass_5hours.zip --out allclass_5hours_joined.zip && unzip -q allclass_5hours_joined.zip -d stage/
```
