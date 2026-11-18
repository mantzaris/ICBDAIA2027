#!/usr/bin/env python3
"""Audit saved metric citations without changing exports or computing metrics.

Completed common-evaluator directories below results/ are discovered by default.
Both table_cells.json and claim_to_evidence.json are checked. Explicitly null
event_metrics containers receive additive navigation companions, never invented
metric values. A malformed pointer, missing field, or non-null mismatch fails.

Example: python3 scripts/audit_metric_table_pointers.py --project-root .
Repeat --evaluation-dir for a narrower audit. The report is safe to regenerate
after later study phases finish; no scientific file is written.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


EXPORTS = ("table_cells.json", "claim_to_evidence.json")
METRICS = ("precision", "recall", "f1")


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def strict_json(value):
    def reject(token):
        raise ValueError("Nonstandard JSON numeric token: " + token)
    return json.loads(value, parse_constant=reject)


def encode_pointer(tokens):
    return "".join("/" + str(token).replace("~", "~0").replace("/", "~1") for token in tokens)


def resolve_pointer(document, pointer):
    """Return the value or the closest resolvable parent and failure category."""
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise ValueError("Invalid JSON pointer")
    encoded = pointer.split("/")[1:] if pointer else []
    if any(re.search(r"~(?:[^01]|$)", token) for token in encoded):
        raise ValueError("Invalid JSON pointer escape")
    tokens = [token.replace("~1", "/").replace("~0", "~") for token in encoded]
    value = document
    for index, token in enumerate(tokens):
        failure = None
        if value is None:
            failure = "null_parent"
        elif isinstance(value, dict):
            if token not in value:
                failure = "missing_field"
        elif isinstance(value, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", token) or int(token) >= len(value):
                failure = "invalid_array_index"
        else:
            failure = "non_container_parent"
        if failure:
            return dict(resolved=False, reason=failure, parent_pointer=encode_pointer(tokens[:index]),
                        parent_value=value, remaining_tokens=tokens[index:])
        value = value[int(token)] if isinstance(value, list) else value[token]
    return dict(resolved=True, value=value)


def cell_check(summary, cell, index):
    """Compare existing values exactly, with one explicitly bounded null case."""
    if not isinstance(cell, dict):
        raise ValueError("Export entry is not an object")
    for key in ("stratum", "domain", "method", "metric", "value", "source", "json_pointer", "reference_scored_episodes"):
        if key not in cell:
            raise ValueError("Missing export field: " + key)
    if any(not isinstance(cell[key], str) for key in ("stratum", "domain", "method", "metric")):
        raise ValueError("Invalid export cell identity")
    if cell["metric"] not in METRICS:
        raise ValueError("Unknown exported metric: " + cell["metric"])
    expected = encode_pointer(["results", cell["stratum"], cell["domain"], cell["method"], "event_metrics", cell["metric"]])
    if cell["json_pointer"] != expected:
        raise ValueError("Pointer differs from the export's declared stratum/domain/method/metric")
    resolved = resolve_pointer(summary, cell["json_pointer"])
    if resolved["resolved"]:
        # Comparing canonical JSON avoids Python's True == 1 equivalence.
        if json.dumps(resolved["value"], sort_keys=True, allow_nan=False) != json.dumps(cell["value"], sort_keys=True, allow_nan=False):
            raise ValueError("Saved export value differs from its resolved source value")
        return dict(status="resolved", export_entry_pointer="/" + str(index), cell_index=index,
                    original_pointer=cell["json_pointer"], requested_metric=cell["metric"],
                    unchanged_export_value=cell["value"], source_value=resolved["value"], claim_id=cell.get("claim_id"))
    if not (resolved["reason"] == "null_parent" and cell["value"] is None
            and resolved["remaining_tokens"] == [cell["metric"]]
            and resolved["parent_pointer"].endswith("/event_metrics")):
        raise ValueError("Unexplained broken source pointer: " + resolved["reason"] + " at " + resolved["parent_pointer"])
    group_pointer = resolved["parent_pointer"].rsplit("/", 1)[0]
    group = resolve_pointer(summary, group_pointer)["value"]
    # The producer's zero scored-reference count explains why its whole metric
    # container is unavailable. A missing field is never treated as this case.
    if type(group.get("reference_scored_episodes")) is not int or group["reference_scored_episodes"] != 0:
        raise ValueError("Null metric container lacks explicit zero reference-scored support")
    if type(cell.get("reference_scored_episodes")) is not int or cell["reference_scored_episodes"] != 0:
        raise ValueError("Null export does not agree with source zero reference-scored support")
    return dict(status="explicitly_unavailable_null_parent", export_entry_pointer="/" + str(index), cell_index=index,
                original_pointer=cell["json_pointer"], resolvable_parent_pointer=resolved["parent_pointer"],
                parent_value=None, requested_metric=cell["metric"], unchanged_export_value=None, claim_id=cell.get("claim_id"),
                unavailable_reason="Source event_metrics is explicitly null and both saved reference_scored_episodes fields are zero; no leaf metric is asserted.",
                source_support_pointer=group_pointer + "/reference_scored_episodes", source_support_value=group["reference_scored_episodes"],
                saved_reference_statuses=group.get("reference_statuses"),
                saved_reference_scopes=group.get("reference_scopes"),
                saved_partial_reference_diagnostic_episodes=group.get("partial_reference_diagnostic_episodes"))


def confined(root, value):
    path = (root / value).resolve()
    if root != path and root not in path.parents:
        raise ValueError("Path lies outside project root: " + str(value))
    return path


def expected_keys(summary):
    """Inventory identities only; never aggregate their scientific values."""
    keys = set()
    try:
        for stratum, domains in summary["results"].items():
            for domain, methods in domains.items():
                for method in methods:
                    keys.update((stratum, domain, method, metric) for metric in METRICS)
    except (KeyError, AttributeError, TypeError) as error:
        raise ValueError("Invalid completed-summary result inventory") from error
    if not keys:
        raise ValueError("Completed evaluation has no exported result groups")
    return keys


def audit(root, directories=None):
    root = Path(root).resolve()
    if directories is not None:
        directories = [confined(root, item) for item in directories]
    else:
        directories = {p.parent for name in EXPORTS for p in (root / "results").rglob(name)}
        # Include completed directories even if either or both exports vanished.
        # Unrelated training/annotation summaries are outside this audit's scope.
        for path in (root / "results").rglob("summary.json"):
            try:
                value = strict_json(path.read_bytes())
            except (OSError, ValueError):
                continue  # Known export directories are diagnosed below.
            if isinstance(value, dict) and value.get("status") == "complete_evaluation":
                directories.add(path.parent)
    records, companions, skipped, failures, sources = [], [], [], [], {}
    for folder in sorted(set(directories)):
        relative = folder.relative_to(root).as_posix()
        summary_path = folder / "summary.json"
        try:
            summary_bytes = summary_path.read_bytes()
            summary = strict_json(summary_bytes)
            if summary.get("status") != "complete_evaluation":
                skipped.append(dict(directory=relative, reason="summary is not a completed evaluation", saved_status=summary.get("status")))
                continue
            source_name = summary_path.relative_to(root).as_posix()
            source_hash = digest_bytes(summary_bytes)
            sources[source_name] = source_hash
            table_keys = expected_keys(summary)
        except (OSError, ValueError, AttributeError) as error:
            failures.append(dict(directory=relative, error=str(error)))
            continue
        for filename in EXPORTS:
            export_path = folder / filename
            export_name = export_path.relative_to(root).as_posix()
            record = dict(export_path=export_name, source_summary_path=source_name, source_summary_sha256=source_hash,
                          status="passed", checked_cells=[], failures=[])
            mappings = []
            try:
                export_bytes = export_path.read_bytes()
                record["export_sha256"] = digest_bytes(export_bytes)
                sources[export_name] = record["export_sha256"]
                cells = strict_json(export_bytes)
                if not isinstance(cells, list):
                    raise ValueError("Metric export must be a list")
                observed = Counter()
                for index, cell in enumerate(cells):
                    try:
                        if not isinstance(cell, dict) or not isinstance(cell.get("source"), str):
                            raise ValueError("Missing/invalid source summary reference")
                        identity = tuple(cell.get(key) for key in ("stratum", "domain", "method", "metric"))
                        if all(isinstance(value, str) for value in identity):
                            observed[identity] += 1
                        if filename == "claim_to_evidence.json" and cell.get("claim_id") != "{}_{}_{}".format(cell.get("stratum"), cell.get("method"), cell.get("metric")):
                            raise ValueError("Missing or mismatched claim_id")
                        if confined(root, folder / cell["source"]) != summary_path:
                            raise ValueError("Cell does not reference its evaluation summary")
                        result = cell_check(summary, cell, index)
                        record["checked_cells"].append(result)
                        if result["status"] == "explicitly_unavailable_null_parent":
                            mappings.append(result)
                    except (ValueError, KeyError, TypeError) as error:
                        record["failures"].append(dict(cell_index=index, export_entry_pointer="/" + str(index),
                                                       original_cell=cell, error=str(error)))
                expected = table_keys if filename == "table_cells.json" else {key for key in table_keys if key[1] == "all"}
                missing, extra = sorted(expected - set(observed)), sorted(set(observed) - expected)
                duplicate = sorted(key for key, count in observed.items() if count != 1)
                if missing or extra or duplicate:
                    record["failures"].append(dict(error="Export row inventory mismatch", missing_cell_keys=missing,
                                                    unexpected_cell_keys=extra, duplicate_cell_keys=duplicate))
                if digest_bytes(export_path.read_bytes()) != record["export_sha256"]:
                    raise ValueError("Export changed during the read-only audit")
            except (OSError, ValueError, TypeError) as error:
                record["failures"].append(dict(error=str(error)))
            if record["failures"]:
                record["status"] = "integrity_failure"
                failures.append(dict(export_path=export_name, errors=record["failures"]))
            records.append(record)
            if mappings:
                companions.append(dict(export_path=export_name, export_sha256=record["export_sha256"],
                    source_summary_path=source_name, source_summary_sha256=source_hash,
                    meaning="Additive navigation companions only; original exports and values are unchanged. The original leaf is unavailable because its event_metrics parent is null.",
                    mappings=mappings))
        try:
            if digest_bytes(summary_path.read_bytes()) != source_hash:
                failures.append(dict(directory=relative, error="Source summary changed during the read-only audit"))
        except OSError as error:
            failures.append(dict(directory=relative, error="Source summary unavailable at final integrity check: " + str(error)))
    if not records:
        failures.append(dict(error="No completed evaluation exports were audited"))
    return dict(audited_at_utc=datetime.now(timezone.utc).isoformat(), status="passed" if not failures else "integrity_failure",
                project_root=str(root), audit_script_sha256=digest_bytes(Path(__file__).read_bytes()),
                scope="Completed common-evaluation exports only; this audit does not certify that the overall study is finished.",
                numerical_policy="CPU-only JSON navigation, exact saved-value comparisons, file hashing, and administrative inventory counts. No scientific values computed or aggregated.",
                scientific_sources_or_outputs_modified=False, remote_writes=False,
                source_sha256=sources, export_audits=records, null_pointer_companions=companions,
                skipped_directories=skipped, failures=failures)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--evaluation-dir", action="append", help="Project-relative completed evaluation directory; repeat as needed")
    parser.add_argument("--output", type=Path, default=Path("reports/metric_table_pointer_audit.json"))
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    output = confined(root, args.output)
    if root / "reports" not in output.parents or output.suffix != ".json":
        parser.error("Additive audit output must be a JSON file below project reports/")
    report = audit(root, args.evaluation_dir)
    if str(output.relative_to(root)) in report["source_sha256"]:
        parser.error("Output would overwrite a source export")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=report["status"], report=str(output), exports=len(report["export_audits"]),
                          checked_cells=sum(len(r["checked_cells"]) for r in report["export_audits"]),
                          unavailable_companions=sum(len(r["mappings"]) for r in report["null_pointer_companions"]),
                          failures=report["failures"]), indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
