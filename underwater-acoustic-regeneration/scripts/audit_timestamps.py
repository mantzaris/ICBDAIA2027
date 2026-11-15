"""Check temporal overlap and shared recording days without reading test outcomes."""
from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    with gzip.open(ROOT / "manifests/sources.jsonl.gz", "rt") as stream:
        rows = list(map(json.loads, stream))
    by_device = defaultdict(list)
    dates = defaultdict(set)
    for row in rows:
        start, end = row["timestamp_interval"].split("_")
        row = dict(row, start=datetime.strptime(start, "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=timezone.utc).timestamp(),
                   end=datetime.strptime(end, "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=timezone.utc).timestamp())
        by_device[row["hydrophone"]].append(row)
        dates[row["date"]].add(row["partition"])
    overlaps, close_pairs = [], []
    for device, sources in by_device.items():
        sources.sort(key=lambda row: row["start"])
        active = []
        for row in sources:
            active = [other for other in active if other["end"] + 60 >= row["start"]]
            for other in active:
                if row["group_id"] == other["group_id"]:
                    continue
                separation = row["start"] - other["end"]
                record = {"left_source": other["original_file_id"], "right_source": row["original_file_id"],
                          "left_partition": other["partition"], "right_partition": row["partition"],
                          "hydrophone": device, "separation_seconds": separation}
                if separation < 0:
                    overlaps.append(record)
                elif other["partition"] != row["partition"]:
                    close_pairs.append(record)
            active.append(row)
    cross_overlap = [row for row in overlaps if row["left_partition"] != row["right_partition"]]
    report = {"sources": len(rows), "hydrophones": {key: len(value) for key, value in by_device.items()},
              "unique_dates": len(dates), "dates_shared_across_partitions": sum(len(value) > 1 for value in dates.values()),
              "cross_group_overlapping_intervals": len(overlaps), "cross_partition_overlapping_intervals": len(cross_overlap),
              "cross_partition_pairs_within_60_seconds": len(close_pairs),
              "cross_partition_pairs_within_5_seconds": sum(row["separation_seconds"] <= 5 for row in close_pairs),
              "independence_proven": False,
              "implication": "Vessel/source grouping does not establish environmental independence. Shared days require date-block sensitivity; no unconditional calibration guarantee follows from group counts.",
              "cross_partition_overlap_examples": cross_overlap[:20], "nearby_examples": close_pairs[:10]}
    (ROOT / "reports/temporal_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
