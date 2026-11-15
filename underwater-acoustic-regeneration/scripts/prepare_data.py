"""Audit official UniqueShip WAVs and assign groups before processing audio.

Run --inspect first to review actual CSV columns. The joined ZIP is read directly;
native audio is never fully expanded to disk. Test references are not evaluated.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import posixpath
import threading
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from acoustic_regeneration.data.grouping import assign_partitions, audit_partitions
from acoustic_regeneration.data.preprocessing import resample_source
from acoustic_regeneration.registry import file_hash

ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    path.write_bytes(gzip.compress(payload, mtime=0) if path.suffix == ".gz" else payload)


def metadata_rows(archive: zipfile.ZipFile) -> list:
    entries = set(archive.namelist())
    rows = []
    for partition in ("train", "validation", "test"):
        matches = [name for name in entries if name.endswith("fold_1/metadata_5s_" + partition + ".csv")]
        if len(matches) != 1:
            raise ValueError("Expected exactly one fold-1 CSV for " + partition)
        name = matches[0]
        payload = archive.read(name)
        destination = ROOT / "provenance" / "uniqueship_metadata" / Path(name).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.with_suffix(".csv.gz").write_bytes(gzip.compress(payload, mtime=0))
        for original in csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))):
            member = posixpath.normpath(posixpath.join(posixpath.dirname(name), original["path"]))
            if member not in entries:
                raise ValueError("Metadata path absent from archive: " + member)
            source_id = hashlib.sha256(member.encode()).hexdigest()
            mmsi = original.get("MMSI", "").strip()
            if mmsi.endswith(".0"):
                mmsi = mmsi[:-2]
            if mmsi in {"", "nan", "0"}:
                mmsi = None
            hydrophone = original.get("hydrophone", "").strip()
            if not hydrophone or hydrophone.lower() == "nan":
                hydrophone = "unknown_device_grouped_across_devices"
            date = original.get("date", "").strip()
            if not date:
                raise ValueError("Source date missing; grouping cannot proceed")
            rows.append({"dataset": "UniqueShip-12class-5hours-v1", "archive_member": member,
                         "original_file_id": source_id, "mmsi": mmsi,
                         "recording_session_id": original.get("recording_session_id") or None,
                         "native_parent_file": original.get("original_file") or None,
                         "date": date, "timestamp_interval": Path(member).stem,
                         "site": original.get("site") or None, "hydrophone": hydrophone,
                         "class": original["label"], "official_partition": partition,
                         "declared_sample_rate": original.get("sample_rate"),
                         "source_metadata": original})
    if len({row["original_file_id"] for row in rows}) != len(rows):
        raise ValueError("Fold contains repeated file paths; audit before proceeding")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-pilot", action="store_true")
    args = parser.parse_args()
    manifests = ROOT / "manifests"
    manifests.mkdir(exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        rows = metadata_rows(archive)
        if args.inspect:
            print(json.dumps({"rows": len(rows), "classes": dict(Counter(row['class'] for row in rows)),
                              "metadata_columns": list(rows[0]["source_metadata"]),
                              "first_row": rows[0]}, indent=2))
            return
        # This source manifest exists before any decoded or resampled derivative.
        write_jsonl(manifests / "source_metadata.jsonl.gz", rows)
        archive_read_lock = threading.Lock()

        def audit(row):
            # Keep each shared archive read atomic; decoding/hashing can proceed
            # concurrently. Retain ZIP overlap and CRC checks without bypasses.
            with archive_read_lock:
                content = archive.read(row["archive_member"])
            with sf.SoundFile(io.BytesIO(content)) as audio:
                rate, channels, frames, subtype = audio.samplerate, audio.channels, len(audio), audio.subtype
                decoded = audio.read(dtype="float32", always_2d=True)
            if channels != 1 or not np.isfinite(decoded).all():
                raise ValueError("Non-mono or nonfinite source: " + row["archive_member"])
            # Audit gain/offset variants before assigning final groups. This is
            # not the observation normalization later used by the restorer.
            centered = decoded.astype(np.float64) - decoded.mean(dtype=np.float64)
            deviation = max(float(np.sqrt(np.mean(centered ** 2))), 1e-12)
            quantized = np.rint(centered / deviation * 10000).astype("<i4")
            return dict(row, native_rate=rate, channels=channels, native_frames=frames, subtype=subtype,
                        duration_seconds=frames / rate, source_sha256=hashlib.sha256(content).hexdigest(),
                        normalized_sha256=hashlib.sha256(quantized.tobytes()).hexdigest(),
                        decoded_sha256=hashlib.sha256(decoded.astype("<f4").tobytes()).hexdigest())

        audited = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for index, row in enumerate(executor.map(audit, rows)):
                audited.append(row)
                if (index + 1) % 1000 == 0:
                    print("audited", index + 1, "of", len(rows), flush=True)
        assigned = assign_partitions(audited)
        report = audit_partitions(assigned)
        report["native_rates"] = dict(Counter(row["native_rate"] for row in assigned))
        report["subtypes"] = dict(Counter(row["subtype"] for row in assigned))
        report["too_short_for_4096_ms"] = sum(row["duration_seconds"] < 4.096 for row in assigned)
        report["missing_native_parent_file"] = sum(row["native_parent_file"] is None for row in assigned)
        report["background_grouping"] = "hydrophone-day; unknown hydrophones merged across devices for each date"
        counts = Counter(row["decoded_sha256"] for row in assigned)
        report["exact_duplicate_copies_beyond_first"] = sum(value - 1 for value in counts.values())
        report["near_duplicate_audit"] = "gain/offset-normalized waveform hashes quantized at 1e-4; not exhaustive for time-shifted/overlapping sources"
        manifest = manifests / "sources.jsonl.gz"
        write_jsonl(manifest, assigned)
        report["manifest_sha256"] = file_hash(manifest)
        (manifests / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if args.cache_pilot:
            # Deterministic, group-spread selection; no calibration/test outcomes.
            eligible = [row for row in assigned if row["duration_seconds"] >= 4.096]
            selected = []
            for partition, limit in (("G", 1440), ("D", 200), ("V", 200)):
                candidates = sorted((row for row in eligible if row["partition"] == partition),
                                    key=lambda row: hashlib.sha256(("pilot-17:" + row["original_file_id"]).encode()).hexdigest())
                by_group = {}
                remainder = []
                for row in candidates:
                    if row["group_id"] not in by_group:
                        by_group[row["group_id"]] = row
                    else:
                        remainder.append(row)
                selected.extend((list(by_group.values()) + remainder)[:limit])
            cache = ROOT / "data" / "processed" / "pilot_8k"
            cache.mkdir(parents=True, exist_ok=True)
            for row in selected:
                audio, rate = sf.read(io.BytesIO(archive.read(row["archive_member"])), dtype="float64")
                processed = resample_source(audio, rate).astype(np.float32)
                path = cache / (row["original_file_id"] + ".npy")
                np.save(path, processed, allow_pickle=False)
                row["processed_path"] = str(path.relative_to(ROOT))
                row["processed_sha256"] = file_hash(path)
            write_jsonl(manifests / "pilot_sources.jsonl.gz", selected)
            print("pilot cache sources", len(selected), flush=True)


if __name__ == "__main__":
    main()
