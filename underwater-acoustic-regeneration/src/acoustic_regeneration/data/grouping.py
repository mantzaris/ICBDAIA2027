"""Connected source ancestry prevents derived samples from crossing partitions."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, List, Mapping, Sequence


def identity_keys(row: Mapping) -> List[str]:
    namespace = str(row.get("dataset", "unknown"))
    keys = ["file:" + namespace + ":" + str(row["original_file_id"])]
    for field in ("mmsi", "recording_session_id", "encounter_id", "native_parent_file"):
        value = row.get(field)
        if value not in (None, "", "unknown", "0", 0):
            keys.append(field + ":" + str(value))
    # Decoded-content hashes are intentionally global across archives.
    if row.get("decoded_sha256"):
        keys.append("audio:" + row["decoded_sha256"])
    if row.get("normalized_sha256"):
        keys.append("normalized_audio:" + row["normalized_sha256"])
    if str(row.get("class", "")).lower() == "background":
        if not row.get("hydrophone") or not row.get("date"):
            raise ValueError("Background sources require hydrophone and date")
        keys.append("background:" + namespace + ":" + row["hydrophone"] + ":" + row["date"])
    return keys


def connected_groups(rows: Sequence[Mapping]) -> List[str]:
    parent = list(range(len(rows)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen: Dict[str, int] = {}
    for index, row in enumerate(rows):
        for key in identity_keys(row):
            if key in seen:
                parent[root(index)] = root(seen[key])
            else:
                seen[key] = index
    members = defaultdict(set)
    for index, row in enumerate(rows):
        members[root(index)].update(identity_keys(row))
    names = {key: hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()
             for key, values in members.items()}
    return [names[root(index)] for index in range(len(rows))]


def assign_partitions(rows: Sequence[Mapping], seed: int = 20261115) -> List[dict]:
    """Preserve test and validation; hash remaining groups in a 6:1:1 G:D:C ratio.

    Conflicting non-test rows are purged instead of promoted into the benchmark
    test set. Validation-linked non-validation rows are also purged. Actual
    fractions/group counts must be audited; the algorithm never splits a group.
    """
    names = connected_groups(rows)
    official = defaultdict(set)
    for group, row in zip(names, rows):
        official[group].add(str(row.get("official_partition", "train")).lower())
    assigned = []
    for group, row in zip(names, rows):
        fold_partition = str(row.get("official_partition", "train")).lower()
        tags = official[group]
        if "test" in tags:
            partition = "test" if fold_partition == "test" else "purged"
        elif tags & {"val", "validation", "v"}:
            partition = "V" if fold_partition in {"val", "validation", "v"} else "purged"
        else:
            digest = hashlib.sha256((str(seed) + ":" + group).encode()).digest()
            unit = int.from_bytes(digest[:8], "big") / 2 ** 64
            partition = "G" if unit < 0.75 else ("D" if unit < 0.875 else "C")
        assigned.append(dict(row, group_id=group, partition=partition))
    audit_partitions(assigned)
    return assigned


def audit_partitions(rows: Sequence[Mapping]) -> dict:
    groups = defaultdict(set)
    identities = defaultdict(set)
    counts = defaultdict(int)
    source_partitions = defaultdict(set)
    for row in rows:
        partition = row["partition"]
        counts[partition] += 1
        if partition == "purged":
            continue
        if partition not in {"G", "D", "V", "C", "test", "external"}:
            raise ValueError("Unknown partition: " + partition)
        groups[row["group_id"]].add(partition)
        source_partitions[row["original_file_id"]].add(partition)
        for key in identity_keys(row):
            identities[key].add(partition)
    collisions = [key for mapping in (groups, identities, source_partitions)
                  for key, partitions in mapping.items() if len(partitions) > 1]
    if collisions:
        raise ValueError("Source/descendant leakage across partitions: " + repr(collisions[:5]))
    group_counts = {partition: sum(partition in values for values in groups.values())
                    for partition in counts if partition != "purged"}
    return {"sources": dict(counts), "independent_groups": group_counts}
