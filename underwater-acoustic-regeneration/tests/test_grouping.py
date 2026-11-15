import pytest

from acoustic_regeneration.data.grouping import assign_partitions, audit_partitions, connected_groups


def test_transitive_source_vessel_links_and_official_test_purge():
    rows = [
        {"original_file_id": "a", "mmsi": "123", "recording_session_id": "s1", "official_partition": "test"},
        {"original_file_id": "b", "mmsi": "123", "recording_session_id": "s2", "official_partition": "train"},
        {"original_file_id": "c", "mmsi": "456", "recording_session_id": "s2", "official_partition": "train"},
    ]
    assert len(set(connected_groups(rows))) == 1
    result = assign_partitions(rows)
    assert [row["partition"] for row in result] == ["test", "purged", "purged"]
    assert audit_partitions(result)["independent_groups"] == {"test": 1}


def test_descendant_relabeling_does_not_bypass_leakage_check():
    rows = [
        {"original_file_id": "same", "group_id": "invented-a", "partition": "G"},
        {"original_file_id": "same", "group_id": "invented-b", "partition": "test"},
    ]
    with pytest.raises(ValueError, match="leakage"):
        audit_partitions(rows)


def test_duplicate_audio_cannot_cross_groups_even_if_identity_changes():
    rows = [{"original_file_id": str(i), "decoded_sha256": "same-audio", "mmsi": str(i + 1)} for i in range(3)]
    assert len(set(connected_groups(rows))) == 1


def test_partition_assignment_is_order_independent():
    rows = [{"original_file_id": str(i), "mmsi": str(i + 1)} for i in range(100)]
    first = {row["original_file_id"]: row["partition"] for row in assign_partitions(rows)}
    second = {row["original_file_id"]: row["partition"] for row in assign_partitions(rows[::-1])}
    assert first == second


def test_background_sources_group_by_device_day():
    rows = [{"original_file_id": str(i), "class": "background", "hydrophone": "h", "date": "2026-01-01", "mmsi": 0}
            for i in range(3)]
    assert len(set(connected_groups(rows))) == 1
