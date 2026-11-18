"""CPU-only identity checks for frozen AI reference construction.

Control membership comes from declared seeds, never manifest position. Hashes
bind cached labels to the complete input record and the code/configuration used
to generate them; they do not certify the physical correctness of AI labels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ELIGIBILITY_PATH = "configs/ai_reference_eligibility.json"
SOURCE_PATHS = (
    "scripts/run_reference_labels.py",
    "scripts/select_ai_references.py",
    "src/sonar/annotators.py",
    "src/sonar/reference_protocol.py",
    "src/sonar/signal.py",
    "src/sonar/metrics.py",
    "src/sonar/synthetic.py",
    "src/sonar/data.py",
    "src/sonar/json_protocol.py",
    "src/sonar/annotation_batching.py",
    "configs/ai_reference_execution.json",
    "configs/ai_annotations.json",
    "configs/ai_annotations_internvl8b_pilot.json",
    ELIGIBILITY_PATH,
)


def object_hash(value) -> str:
    """Hash JSON data independently of dictionary insertion order."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    """Fingerprint every source/configuration that defines this protocol."""
    return {path: _file_hash(REPOSITORY_ROOT / path) for path in SOURCE_PATHS}


def _declared_seeds(policy, key):
    seeds = policy.get(key)
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"Policy {key} must be a nonempty list")
    if any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError(f"Policy {key} must contain unique integer seeds")
    return seeds


def select_controls(records, policy, role) -> list[dict]:
    """Validate control provenance, then select an exact declared seed set.

    Accept a complete control manifest or just the requested role's controls.
    Every supplied record must be a valid declared control, including controls
    from the other role. The result follows policy seed order, not input order.
    """
    roles = {
        "calibration": "annotation_calibration",
        "annotation_calibration": "annotation_calibration",
        "audit": "reference_error_audit",
        "reference_error_audit": "reference_error_audit",
    }
    if role not in roles:
        raise ValueError(f"Unknown control role: {role}")
    role = roles[role]
    calibration = _declared_seeds(policy, "calibration_seeds")
    audit = _declared_seeds(policy, "held_out_audit_seeds")
    if set(calibration) & set(audit):
        raise ValueError("Calibration and held-out audit seeds must be disjoint")
    expected_roles = {seed: "annotation_calibration" for seed in calibration}
    expected_roles.update({seed: "reference_error_audit" for seed in audit})
    by_seed = {}
    scene_ids = set()
    for record in records:
        if not isinstance(record, dict) or record.get("annotation_control") is not True:
            raise ValueError("Every supplied record must be an annotation control")
        scene_id, seed = record.get("scene_id"), record.get("seed")
        if not isinstance(scene_id, str) or not scene_id or scene_id in scene_ids:
            raise ValueError("Control scene IDs must be nonempty and unique")
        if type(seed) is not int or seed not in expected_roles:
            raise ValueError(f"Undeclared control seed: {seed}")
        if seed in by_seed:
            raise ValueError(f"Duplicate control seed: {seed}")
        if record.get("control_role") != expected_roles[seed]:
            raise ValueError(f"Control role disagrees with declared seed membership: {seed}")
        if record.get("stratum") != "synthetic" or record.get("split") != "pilot":
            raise ValueError(f"Control must use the synthetic pilot generator split: {seed}")
        if record.get("generator_split", "pilot") != "pilot":
            raise ValueError(f"Control generator_split must be pilot: {seed}")
        by_seed[seed] = record
        scene_ids.add(scene_id)
    requested = calibration if role == "annotation_calibration" else audit
    missing = sorted(set(requested) - by_seed.keys())
    if missing:
        raise ValueError(f"Missing declared {role} control seeds: {missing}")
    return [by_seed[seed] for seed in requested]


def protocol_identity(config_path) -> dict:
    """Return source and selected configuration hashes without loading a model."""
    path = Path(config_path)
    path = (REPOSITORY_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("Reference configuration must be inside the repository") from exc
    hashes = source_hashes()
    if relative not in hashes:
        raise ValueError("Reference configuration must be included in protocol source hashes")
    return {
        "source_hashes": hashes,
        "config_path": relative,
        "config_sha256": hashes[relative],
        "eligibility_protocol_sha256": hashes[ELIGIBILITY_PATH],
    }


def verify_frozen(freeze, identity) -> None:
    """Reject changed sources/policy before using the frozen reference protocol."""
    expected = identity.get("source_hashes")
    if not isinstance(expected, dict) or set(expected) != set(SOURCE_PATHS):
        raise RuntimeError("Current protocol identity has incomplete source hashes")
    frozen = freeze.get("source_hashes")
    if frozen != expected:
        frozen = frozen if isinstance(frozen, dict) else {}
        changed = sorted(path for path in set(expected) | set(frozen)
                         if expected.get(path) != frozen.get(path))
        raise RuntimeError("Frozen reference source hashes changed: " + ", ".join(changed))
    policy_hash = expected[ELIGIBILITY_PATH]
    if identity.get("eligibility_protocol_sha256") != policy_hash:
        raise RuntimeError("Current protocol identity has inconsistent eligibility policy hash")
    if freeze.get("eligibility_protocol_sha256") != policy_hash:
        raise RuntimeError("Frozen reference eligibility policy hash changed")
    config_path = identity.get("config_path")
    if config_path not in expected or identity.get("config_sha256") != expected[config_path]:
        raise RuntimeError("Current protocol identity has inconsistent configuration hash")


def verify_record(stored, expected_identity, input_record) -> None:
    """Require a cache record to match current protocol and exact input metadata."""
    if stored.get("protocol_identity") != expected_identity:
        raise RuntimeError("Cached reference protocol identity changed or is missing")
    if stored.get("input_record_sha256") != object_hash(input_record):
        raise RuntimeError("Cached reference input record changed or its hash is missing")
    if stored.get("scene_id") != input_record.get("scene_id"):
        raise RuntimeError("Cached reference scene ID does not match its input record")
