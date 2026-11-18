"""CPU-only safeguards against stale labels and held-out control leakage."""
from copy import deepcopy
import json
import subprocess
import sys

import pytest

from sonar.reference_protocol import (
    ELIGIBILITY_PATH, REPOSITORY_ROOT, SOURCE_PATHS, object_hash,
    protocol_identity, select_controls, source_hashes, verify_frozen, verify_record,
)


POLICY = {"calibration_seeds": [101, 102], "held_out_audit_seeds": [201, 202]}


def controls():
    return [
        {"scene_id": f"control-{seed}", "seed": seed, "annotation_control": True,
         "control_role": role, "stratum": "synthetic", "split": "pilot"}
        for role, seeds in [("annotation_calibration", POLICY["calibration_seeds"]),
                            ("reference_error_audit", POLICY["held_out_audit_seeds"])]
        for seed in seeds
    ]


def test_reordered_controls_preserve_declared_membership():
    records = list(reversed(controls()))
    assert [r["seed"] for r in select_controls(records, POLICY, "calibration")] == [101, 102]
    assert [r["seed"] for r in select_controls(records, POLICY, "reference_error_audit")] == [201, 202]
    assert [r["seed"] for r in records] == [202, 201, 102, 101]


def test_audit_record_cannot_fill_a_missing_calibration_slot():
    with pytest.raises(ValueError, match="Missing declared"):
        select_controls(controls()[1:], POLICY, "calibration")
    records = controls()
    records[0]["seed"] = 201
    records[0]["scene_id"] = "relabeled-audit"
    with pytest.raises(ValueError, match="role disagrees"):
        select_controls(records, POLICY, "calibration")


@pytest.mark.parametrize("field,value,error", [
    ("scene_id", "control-101", "scene IDs"),
    ("seed", 101, "Duplicate control seed"),
    ("seed", 999, "Undeclared control seed"),
    ("annotation_control", False, "annotation control"),
    ("split", "train", "pilot generator split"),
    ("stratum", "real", "pilot generator split"),
])
def test_control_provenance_is_required(field, value, error):
    records = controls()
    records[1][field] = value
    with pytest.raises(ValueError, match=error):
        select_controls(records, POLICY, "calibration")


def test_declared_roles_cannot_overlap_or_repeat_seeds():
    policy = deepcopy(POLICY)
    policy["held_out_audit_seeds"][0] = 101
    with pytest.raises(ValueError, match="disjoint"):
        select_controls(controls(), policy, "calibration")
    policy = deepcopy(POLICY)
    policy["calibration_seeds"] = [101, 101]
    with pytest.raises(ValueError, match="unique integer"):
        select_controls(controls(), policy, "calibration")


def test_frozen_identity_rejects_changed_sources_and_policy():
    identity = protocol_identity("configs/ai_annotations.json")
    freeze = {"source_hashes": deepcopy(identity["source_hashes"]),
              "eligibility_protocol_sha256": identity["eligibility_protocol_sha256"]}
    verify_frozen(freeze, identity)
    stale = deepcopy(freeze)
    stale["source_hashes"]["src/sonar/synthetic.py"] = "older-generator"
    with pytest.raises(RuntimeError, match="synthetic.py"):
        verify_frozen(stale, identity)
    stale = deepcopy(freeze)
    stale["eligibility_protocol_sha256"] = "older-policy"
    with pytest.raises(RuntimeError, match="eligibility policy"):
        verify_frozen(stale, identity)
    stale = deepcopy(freeze)
    del stale["source_hashes"]["src/sonar/data.py"]
    with pytest.raises(RuntimeError, match="data.py"):
        verify_frozen(stale, identity)


def test_cached_label_requires_exact_input_and_protocol():
    identity = protocol_identity("configs/ai_annotations.json")
    record = controls()[0]
    stored = {"scene_id": record["scene_id"], "protocol_identity": deepcopy(identity),
              "input_record_sha256": object_hash(record)}
    verify_record(stored, identity, record)
    changed = dict(record, split="test")
    with pytest.raises(RuntimeError, match="input record"):
        verify_record(stored, identity, changed)
    stale = deepcopy(stored)
    stale["protocol_identity"]["config_sha256"] = "old-config"
    with pytest.raises(RuntimeError, match="protocol identity"):
        verify_record(stale, identity, record)
    stale = deepcopy(stored)
    del stale["protocol_identity"]
    with pytest.raises(RuntimeError, match="protocol identity"):
        verify_record(stale, identity, record)
    stale = dict(stored, scene_id="another-scene")
    with pytest.raises(RuntimeError, match="scene ID"):
        verify_record(stale, identity, record)


def test_identity_covers_both_models_and_all_protocol_sources():
    hashes = source_hashes()
    assert set(hashes) == set(SOURCE_PATHS)
    required = {"scripts/run_reference_labels.py", "scripts/select_ai_references.py",
                "src/sonar/annotators.py", "src/sonar/reference_protocol.py",
                "src/sonar/signal.py", "src/sonar/metrics.py", "src/sonar/synthetic.py",
                "src/sonar/data.py", "src/sonar/json_protocol.py", ELIGIBILITY_PATH,
                "configs/ai_annotations.json", "configs/ai_annotations_internvl8b_pilot.json"}
    assert required <= set(hashes)
    small = protocol_identity("configs/ai_annotations.json")
    large = protocol_identity(REPOSITORY_ROOT / "configs/ai_annotations_internvl8b_pilot.json")
    assert small["source_hashes"] == large["source_hashes"]
    assert small["config_path"] != large["config_path"]
    assert small["config_sha256"] != large["config_sha256"]


def test_object_hash_is_canonical_and_rejects_nonfinite_data():
    assert object_hash({"a": 1, "b": [2, 3]}) == object_hash({"b": [2, 3], "a": 1})
    assert object_hash({"b": [2, 3]}) != object_hash({"b": [3, 2]})
    with pytest.raises(ValueError):
        object_hash({"seed": float("nan")})


def test_protocol_helpers_do_not_import_torch_or_initialize_cuda():
    code = (
        "import sys; "
        f"sys.path.insert(0, {json.dumps(str(REPOSITORY_ROOT / 'src'))}); "
        "from sonar.reference_protocol import protocol_identity; "
        "protocol_identity('configs/ai_annotations.json'); "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
