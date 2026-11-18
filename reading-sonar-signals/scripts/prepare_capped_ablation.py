"""Prepare an explicitly authorized, method-balanced ablation prefix resume.

Administrative file/JSON/hash operations only. The frozen runner, controller,
original manifest and interrupted run remain unchanged. No model is loaded.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

METHODS = ["no_demon", "no_graph", "no_quantitative", "forced_exclusivity", "fixed_resolution", "random"]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inside(root, path):
    value = (root / path).resolve()
    if value == root or root not in value.parents:
        raise ValueError("Path must be inside the project: " + str(path))
    return value


def relative(root, path):
    return Path(path).relative_to(root).as_posix()


def inventory(folder):
    rows = []
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise ValueError("Refusing a symlink in the original run: " + str(path))
        if path.is_file():
            rows.append({"path": path.relative_to(folder).as_posix(), "bytes": path.stat().st_size,
                         "sha256": digest(path)})
    return rows


def selection(root, original_manifest, freeze_path, cap):
    if cap < len(METHODS):
        raise ValueError("Cap must permit at least one complete six-method scene")
    freeze = read(freeze_path)
    for name, expected in freeze["sha256"].items():
        if digest(inside(root, name)) != expected:
            raise ValueError("Frozen source changed: " + name)
    name = relative(root, original_manifest)
    if freeze["sha256"].get(name) != digest(original_manifest):
        raise ValueError("Original manifest must be present in the unchanged freeze")
    lines = [line for line in original_manifest.read_bytes().splitlines(keepends=True) if line.strip()]
    records = [json.loads(line) for line in lines]
    if len({row["scene_id"] for row in records}) != len(records):
        raise ValueError("Original manifest contains duplicate scene IDs")
    count = cap // len(METHODS)
    if count >= len(records):
        raise ValueError("This preparation requires a strict prefix of the original run")
    if any(row.get("stratum") != "real" for row in records):
        raise ValueError("This authorized cap applies only to the original real ablation cohort")
    return records, records[:count], b"".join(lines[:count]), freeze


def publish_manifest(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("Existing capped manifest differs; refusing overwrite")
        return
    with path.open("xb") as stream:
        stream.write(payload)


def prepare(root, source, output, original_manifest, capped_manifest, freeze_path, cap=200,
            writers_stopped=False, writer_pid=None, manifest_only=False):
    root = Path(root).resolve()
    source, output, original_manifest, capped_manifest, freeze_path = [
        inside(root, value) for value in (source, output, original_manifest, capped_manifest, freeze_path)]
    if original_manifest == capped_manifest or source == output or source in output.parents or output in source.parents:
        raise ValueError("Original and derived destinations must be separate")
    records, chosen, payload, freeze = selection(root, original_manifest, freeze_path, cap)
    selected_ids = [row["scene_id"] for row in chosen]
    if manifest_only:
        publish_manifest(capped_manifest, payload)
        return {"status": "capped_manifest_prepared", "manifest": relative(root, capped_manifest),
                "sha256": digest(capped_manifest), "selected_scene_ids": selected_ids,
                "selected_scene_count": len(chosen), "methods": METHODS,
                "requested_cap": cap, "expected_inference_episodes": len(chosen) * len(METHODS),
                "selection_rule": "Original frozen manifest prefix, chosen without reading episode outcomes"}
    if not writers_stopped:
        raise ValueError("Require --writers-stopped after the coordinator stops the original writer")
    if writer_pid is not None:
        stat = Path("/proc") / str(writer_pid) / "stat"
        if stat.exists() and stat.read_text().split(") ", 1)[1].split()[0] != "Z":
            raise ValueError("The identified original writer is still alive")
    if output.exists():
        raise ValueError("Derived output already exists; resume it with the unchanged runner")
    if (source / "COMPLETED.json").exists():
        raise ValueError("Expected an interrupted original run, not a completed run")
    original_config = read(source / "run.json")
    if (inside(root, original_config["manifest"]) != original_manifest or
            original_config["manifest_sha256"] != digest(original_manifest) or
            original_config["scene_count"] != len(records) or original_config["methods"] != METHODS):
        raise ValueError("Original configuration does not match the declared real ablation grid")
    for name, expected in original_config["source_sha256"].items():
        if freeze["sha256"].get(name) != expected or digest(inside(root, name)) != expected:
            raise ValueError("Original runner/behavior fingerprint differs from frozen source: " + name)
    before = inventory(source)
    reusable, incomplete, outside = [], [], []
    required = {"scene_id", "method", "report", "raw_report", "schema_valid", "generations", "actions", "final_state"}
    for record in records:
        sid = record["scene_id"]
        for method in METHODS:
            folder = source / sid / method
            episode = folder / "episode.json"
            complete, reason = False, "episode_missing"
            if episode.exists():
                try:
                    value = read(episode)
                    if not isinstance(value, dict) or not required <= set(value):
                        reason = "incomplete_episode_structure"
                    elif value["scene_id"] != sid or value["method"] != method:
                        raise ValueError("Original episode identity mismatch: " + str(episode))
                    else:
                        complete = True
                except (json.JSONDecodeError, UnicodeDecodeError):
                    reason = "interrupted_episode_json"
            item = {"scene_id": sid, "method": method, "source_episode": relative(root, episode)}
            if sid in selected_ids:
                if complete:
                    reusable.append(item)
                else:
                    incomplete.append(dict(item, reason=reason, partial_directory_retained=folder.exists()))
            elif complete:
                outside.append(dict(item, sha256=digest(episode)))
    if len(chosen) * len(METHODS) + len(outside) > cap:
        raise ValueError("Prefix plus already completed outside-prefix jobs would exceed the authorized cap")
    publish_manifest(capped_manifest, payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".capped-ablation-preparing-", dir=str(output.parent)))
    try:
        config = dict(original_config, manifest=relative(root, capped_manifest),
                      manifest_sha256=digest(capped_manifest), scene_count=len(chosen), started_at_unix=time.time())
        write(stage / "run.json", config)
        reused = []
        for item in reusable:
            src = source / item["scene_id"] / item["method"]
            dst = stage / item["scene_id"] / item["method"]
            shutil.copytree(src, dst, copy_function=shutil.copy2)
            files = []
            for entry in inventory(src):
                local = Path(entry["path"])
                if digest(dst / local) != entry["sha256"]:
                    raise ValueError("Reuse copy differs: " + str(src / local))
                files.append({"source": relative(root, src / local),
                              "destination": relative(root, output / item["scene_id"] / item["method"] / local),
                              "sha256": entry["sha256"], "bytes": entry["bytes"]})
            reused.append(dict(item, destination_episode=relative(root, output / item["scene_id"] / item["method"] / "episode.json"),
                               sha256=digest(src / "episode.json"), files=files))
        if inventory(source) != before:
            raise ValueError("Original run changed during preparation; derived output was not published")
        write(stage / "original_run_inventory.json", {"root": relative(root, source), "files": before})
        provenance = {"status": "prepared_capped_resume", "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": digest(Path(__file__)), "requested_cap": cap, "selected_scene_count": len(chosen),
            "expected_inference_episodes": len(chosen) * len(METHODS), "methods": METHODS,
            "selected_scene_ids": selected_ids, "selection_rule": "First floor(cap/6) scenes in unchanged frozen manifest order; no outcome selection; balanced by method, not by domain",
            "original_run": {"path": relative(root, source), "configuration_path": relative(root, source / "run.json"),
                "configuration_sha256": digest(source / "run.json"), "manifest_path": relative(root, original_manifest),
                "manifest_sha256": digest(original_manifest)},
            "capped_manifest": {"path": relative(root, capped_manifest), "sha256": digest(capped_manifest)},
            "derived_configuration_sha256": digest(stage / "run.json"),
            "configuration_fields_changed": ["manifest", "manifest_sha256", "scene_count", "started_at_unix"],
            "original_inventory": {"path": relative(root, output / "original_run_inventory.json"),
                                   "sha256": digest(stage / "original_run_inventory.json")},
            "source_unchanged_after_copy": True, "reused_episodes": reused,
            "incomplete_selected_jobs": incomplete, "completed_outside_selected": outside,
            "writers_stopped_explicitly_confirmed": True, "writer_pid_checked": writer_pid,
            "retained_scope_note": "Original partial files, failures, batch ledger and configuration are retained unchanged. Only complete episodes are copied byte-for-byte. Unfinished selected jobs will restart; discarded partial computation is not hidden. Reused episode costs belong to original execution; the new runner ledger covers only resumed work.",
            "resume_command": [".venv/bin/python", "scripts/run_benchmark.py", "--manifest", relative(root, capped_manifest),
                "--output", relative(root, output), "--methods", ",".join(METHODS), "--budget", str(config["budget"]),
                "--model", config["model"], "--workers", str(config["workers"]),
                "--batch-size", str(config["max_batch_size"]), "--batch-wait-ms", str(config["batch_wait_ms"])]}
        write(stage / "reuse_provenance.json", provenance)
        if output.exists():
            raise ValueError("Derived output appeared during preparation")
        stage.rename(output)
        return provenance
    except BaseException:
        shutil.rmtree(stage)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", default="results/ablation_inference")
    parser.add_argument("--output", default="results/ablation_inference_capped")
    parser.add_argument("--input-manifest", default="data/manifests/ablation_test.jsonl")
    parser.add_argument("--manifest", default="data/manifests/ablation_test_capped.jsonl")
    parser.add_argument("--freeze", default="configs/final_protocol_freeze.json")
    parser.add_argument("--cap", type=int, default=200)
    parser.add_argument("--writers-stopped", action="store_true")
    parser.add_argument("--writer-pid", type=int)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    result = prepare(args.project_root, args.source, args.output, args.input_manifest, args.manifest,
                     args.freeze, args.cap, args.writers_stopped, args.writer_pid, args.manifest_only)
    print(json.dumps({key: result[key] for key in ("status", "requested_cap", "selected_scene_count", "expected_inference_episodes")}, indent=2))


if __name__ == "__main__":
    main()
