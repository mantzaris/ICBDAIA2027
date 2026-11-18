#!/usr/bin/env python3
"""CPU-only final artifact verification and explicitly authorized source snapshot.

This administrative checker does not import the scientific package, infer labels,
recompute scores, edit frozen sources, or update the artifact receipt. Run from
either a preserved local copy or the remote project. Model/schema failures count
as represented episodes; missing episodes do not. Empty AI eligibility is valid.

  python3 scripts/verify_final_deliverables.py
  python3 scripts/verify_final_deliverables.py --gpu-only
  python3 scripts/verify_final_deliverables.py --snapshot --writers-finished

The second command is for the coordinator only after all writers have stopped.
It requires all checks to pass and refuses to overwrite an existing snapshot.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ABLATIONS = ("no_demon", "no_graph", "no_quantitative", "forced_exclusivity", "fixed_resolution", "random")
# name, manifest stem, scenes, methods, common evaluation required
RUNS = (
    ("development", "development", 42, ("G3", "G4"), True),
    ("main", "main_test", 114, ("G1", "G2", "G3", "G4", "F"), True),
    ("ablation_inference", "ablation_test", 80, ABLATIONS, False),
    ("ablations", "ablation_test", 80, ("F", "G4") + ABLATIONS + ("no_verifier",), True),
    ("synthetic_ablation_inference", "synthetic_ablation_test", 12, ABLATIONS, False),
    ("synthetic_ablations", "synthetic_ablation_test", 12, ("F", "G4") + ABLATIONS + ("no_verifier",), True),
    ("budget2", "budget_test", 18, ("F", "G3", "G4"), True),
    ("budget4", "budget_test", 18, ("F", "G3", "G4"), True),
    ("robustness", "robustness_test", 54, ("F", "G4"), True),
    ("sensitivity", "sensitivity_test", 30, ("F", "G4"), True),
    ("localization/predictions", "main_test", 114,
     tuple("%s_localizer_seed%d" % (method, seed) for seed in (17, 29, 43) for method in ("P1", "P2")), True),
)

# The user-authorized runtime cap changes cohort accounting only. The original
# frozen grids remain the default whenever this explicit amendment is absent.
CAPPED_RUNS = {
    "ablation_inference": ("ablation_inference_capped", "ablation_test_capped", 33, ABLATIONS, False),
    "ablations": ("ablations", "ablation_test_capped", 33, ("F", "G4") + ABLATIONS + ("no_verifier",), True),
    "synthetic_ablation_inference": ("synthetic_ablation_inference", "synthetic_ablation_test", 12, (), False),
    "synthetic_ablations": ("synthetic_ablations", "synthetic_ablation_test", 12, ("F", "G4", "no_verifier"), True),
}
USER_CAP = "configs/ablation_user_cap.json"


class VerificationError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def strip_tex_comments(text):
    # A percent preceded by an odd number of backslashes is a literal percent.
    lines = []
    for line in text.splitlines():
        for index, char in enumerate(line):
            if char == "%":
                prefix = line[:index]
                backslashes = len(prefix) - len(prefix.rstrip("\\"))
                if backslashes % 2 == 0:
                    line = prefix
                    break
        lines.append(line)
    return "\n".join(lines)


def manuscript_text_checks(text):
    clean = strip_tex_comments(text)
    switches = re.findall(r"\\finalresults(true|false)\b", clean)
    require(switches and switches[-1] == "true", "Manuscript must enable \\finalresultstrue")
    require("\\begin{document}" in clean, "Missing manuscript document body")
    body = clean.split("\\begin{document}", 1)[1]
    require(not re.search(r"\\pending\s*\{", body), "Manuscript still contains an actual \\pending{...} call")
    return body


class Verifier:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.sections = []
        self.checked_hashes = {}
        self.run_rows = {}
        self.user_cap = None

    def path(self, value):
        """Map project-relative and known local/remote project-absolute paths."""
        value = Path(value)
        if value.is_absolute():
            prefixes = (self.root, Path("/workspace/reading-sonar-signals"),
                        Path("/home/resort/Documents/repos/ICBDAIA2027/reading-sonar-signals"))
            for prefix in prefixes:
                try:
                    value = value.relative_to(prefix)
                    break
                except ValueError:
                    pass
            else:
                raise VerificationError("Source path lies outside known project roots: " + str(value))
        result = (self.root / value).resolve()
        require(result == self.root or self.root in result.parents, "Path escapes project: " + str(value))
        return result

    def hash(self, path, expected):
        path = self.path(path)
        require(isinstance(expected, str) and bool(re.fullmatch(r"[a-f0-9]{64}", expected)),
                "Missing/invalid expected SHA256: " + str(path))
        actual = sha256(path)
        require(actual == expected, "SHA256 mismatch: " + str(path))
        self.checked_hashes[str(path.relative_to(self.root))] = actual

    def hashes(self, mapping):
        require(isinstance(mapping, dict), "Expected path-to-SHA256 mapping")
        for path, expected in mapping.items():
            self.hash(path, expected)

    def embedded_hashes(self, value):
        """Verify saved numerical-array/image references without decoding them."""
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and "sha256" in value:
                self.hash(value["path"], value["sha256"])
            for child in value.values():
                self.embedded_hashes(child)
        elif isinstance(value, list):
            for child in value:
                self.embedded_hashes(child)

    def section(self, name, callback):
        try:
            evidence = callback()
            self.sections.append(dict(name=name, status="passed", evidence=evidence))
        except (OSError, ValueError, KeyError, TypeError, VerificationError, subprocess.SubprocessError) as exc:
            self.sections.append(dict(name=name, status="failed", error=str(exc)))

    def pairs(self, rows, expected, label):
        counts = Counter((row["scene_id"], row["method"]) for row in rows)
        missing = sorted(expected - set(counts))
        extra = sorted(set(counts) - expected)
        duplicate = sorted(key for key, count in counts.items() if count != 1)
        require(not (missing or extra or duplicate),
                "%s grid mismatch: expected=%d actual=%d missing=%d extra=%d duplicates=%d; examples=%s" %
                (label, len(expected), len(rows), len(missing), len(extra), len(duplicate),
                 repr((missing + extra + duplicate)[:5])))

    def protocol(self):
        for filename in ("final_protocol_freeze", "morphology_supplement", "relation_supplement"):
            self.hashes(read_json(self.path("configs/" + filename + ".json"))["sha256"])
        return {"frozen_file_hashes_checked": len(self.checked_hashes)}

    def run_specs(self):
        if self.path(USER_CAP).is_file():
            return tuple(CAPPED_RUNS.get(spec[0], spec) for spec in RUNS)
        return RUNS

    def cap_amendment(self):
        """Validate a declared runtime amendment, never infer it from missing work."""
        amendment = read_json(self.path(USER_CAP))
        require(amendment.get("schema_version") == 1 and
                amendment.get("status") == "user_authorized_runtime_cap", "Unknown ablation cap amendment")
        require(amendment["max_new_ablation_episodes"] == 200 and
                amendment["planned_new_real_ablation_episodes"] == 198, "Unexpected authorized ablation cap")
        original, capped, synthetic = (amendment[key] for key in ("original", "capped", "synthetic_reuse"))
        declarations = (
            (original, dict(manifest="data/manifests/ablation_test.jsonl", run="results/ablation_inference",
                            scene_count=80, methods=list(ABLATIONS), episodes=480)),
            (capped, dict(manifest="data/manifests/ablation_test_capped.jsonl", run="results/ablation_inference_capped",
                          scene_count=33, methods=list(ABLATIONS), episodes=198, composed_run="results/ablations",
                          composed_methods=["F", "G4", *ABLATIONS, "no_verifier"], composed_episodes=297,
                          reuse_provenance="results/ablation_inference_capped/reuse_provenance.json",
                          original_inventory="results/ablation_inference_capped/original_run_inventory.json")),
            (synthetic, dict(manifest="data/manifests/synthetic_ablation_test.jsonl", source_run="results/main",
                             stub_run="results/synthetic_ablation_inference", composed_run="results/synthetic_ablations",
                             proof="results/synthetic_ablation_inference/reuse_only_provenance.json",
                             scene_count=12, new_inference_episodes=0, methods=["F", "G4", "no_verifier"],
                             composed_episodes=36, omitted_methods=list(ABLATIONS))),
        )
        for actual, expected in declarations:
            require(all(actual.get(key) == value for key, value in expected.items()),
                    "Ablation cap declaration differs from the authorized grids")
            self.hash(actual["manifest"], actual["manifest_sha256"])
        require(amendment["queue_script"] == "scripts/run_capped_remaining_study.py", "Unexpected capped queue script")
        old_manifest = self.path(original["manifest"])
        new_manifest = self.path(capped["manifest"])
        original_rows, selected = read_jsonl(old_manifest), read_jsonl(new_manifest)
        require(len(original_rows) == 80 and len(selected) == 33 and selected == original_rows[:33],
                "Capped cohort must be the first 33 frozen manifest rows")
        prefix = b"".join(line for line in old_manifest.read_bytes().splitlines(keepends=True) if line.strip())
        prefix = b"".join(prefix.splitlines(keepends=True)[:33])
        require(new_manifest.read_bytes() == prefix, "Capped manifest is not the byte-preserved frozen prefix")
        require(dict(Counter(row["domain"] for row in selected)) == capped["domain_counts"],
                "Declared capped domain counts differ from the selected manifest")
        require(all(row.get("stratum") == "real" for row in selected), "Real cap includes a non-real scene")
        old_run, new_run = self.path(original["run"]), self.path(capped["run"])
        require(not (old_run / "COMPLETED.json").exists(), "Interrupted original ablations must not claim completion")
        self.hash(old_run / "run.json", original["run_configuration_sha256"])
        old_config, new_config = read_json(old_run / "run.json"), read_json(new_run / "run.json")
        require(old_config["scene_count"] == 80 and old_config["methods"] == list(ABLATIONS),
                "Original ablation configuration was changed")
        require(self.path(old_config["manifest"]) == old_manifest and
                old_config["manifest_sha256"] == original["manifest_sha256"], "Original manifest binding changed")
        changed = {"manifest", "manifest_sha256", "scene_count", "started_at_unix"}
        require({key: value for key, value in old_config.items() if key not in changed} ==
                {key: value for key, value in new_config.items() if key not in changed},
                "Capped run changed frozen inference settings")
        require(new_config["scene_count"] == 33 and self.path(new_config["manifest"]) == new_manifest and
                new_config["manifest_sha256"] == capped["manifest_sha256"], "Capped configuration is misbound")
        interruption = amendment["original_interruption"]
        self.hash(interruption["path"], interruption["sha256"])
        interrupted = read_json(self.path(interruption["path"]))
        require(interrupted["planned_cap"] == 198 and
                interrupted["original_run_sha256"] == original["run_configuration_sha256"] and
                interrupted["scientific_hash_mismatches"] == 0, "Interruption proof does not match the original run")
        proof = read_json(self.path(capped["reuse_provenance"]))
        require(proof["status"] == "prepared_capped_resume" and proof["requested_cap"] == 200 and
                proof["selected_scene_count"] == 33 and proof["expected_inference_episodes"] == 198 and
                proof["methods"] == list(ABLATIONS) and proof["selected_scene_ids"] == [r["scene_id"] for r in selected],
                "Capped reuse proof has incorrect scope")
        require(proof["source_unchanged_after_copy"] is True and proof["writers_stopped_explicitly_confirmed"] is True,
                "Capped copy lacks explicit stopped-writer/preservation evidence")
        require(set(proof["configuration_fields_changed"]) == changed, "Unexpected capped configuration changes")
        self.hash("scripts/prepare_capped_ablation.py", proof["script_sha256"])
        self.hash(new_run / "run.json", proof["derived_configuration_sha256"])
        for key, path in (("path", old_run), ("configuration_path", old_run / "run.json"), ("manifest_path", old_manifest)):
            require(self.path(proof["original_run"][key]) == path, "Reuse proof points at another original run")
        self.hash(old_run / "run.json", proof["original_run"]["configuration_sha256"])
        self.hash(old_manifest, proof["original_run"]["manifest_sha256"])
        require(self.path(proof["capped_manifest"]["path"]) == new_manifest, "Reuse proof points at another capped manifest")
        self.hash(new_manifest, proof["capped_manifest"]["sha256"])
        require(self.path(proof["original_inventory"]["path"]) == self.path(capped["original_inventory"]),
                "Original inventory path differs from amendment")
        self.hash(capped["original_inventory"], proof["original_inventory"]["sha256"])
        inventory = read_json(self.path(capped["original_inventory"]))
        require(self.path(inventory["root"]) == old_run, "Original inventory points at another run")
        inventoried = set()
        for record in inventory["files"]:
            path = self.path(old_run / record["path"])
            require(old_run in path.parents and not path.is_symlink(), "Original inventory escapes its run")
            require(path not in inventoried and path.stat().st_size == record["bytes"], "Invalid original file inventory")
            inventoried.add(path)
            self.hash(path, record["sha256"])
        require(inventoried == {p for p in old_run.rglob("*") if p.is_file()},
                "Original interrupted artifacts were added or removed after preservation")
        expected = {(row["scene_id"], method) for row in selected for method in ABLATIONS}
        self.pairs(proof["reused_episodes"] + proof["incomplete_selected_jobs"], expected, "capped preparation")
        for record in proof["reused_episodes"]:
            sid, method = record["scene_id"], record["method"]
            source, destination = old_run / sid / method, new_run / sid / method
            require(self.path(record["source_episode"]) == source / "episode.json" and
                    self.path(record["destination_episode"]) == destination / "episode.json", "Reused episode is misidentified")
            self.hash(source / "episode.json", record["sha256"])
            self.hash(destination / "episode.json", record["sha256"])
            copied = set()
            for entry in record["files"]:
                src, dst = self.path(entry["source"]), self.path(entry["destination"])
                require(source in src.parents and destination in dst.parents and
                        src.relative_to(source) == dst.relative_to(destination) and src not in copied,
                        "Reused file has inconsistent source/destination")
                copied.add(src)
                require(src.stat().st_size == dst.stat().st_size == entry["bytes"], "Reused file size changed")
                self.hash(src, entry["sha256"])
                self.hash(dst, entry["sha256"])
            require(copied == {p for p in source.rglob("*") if p.is_file()}, "Reuse inventory omits source episode files")
        outside = {(p.parent.parent.name, p.parent.name) for filename in ("episode.json", "failure.json")
                   for p in old_run.glob("*/*/" + filename)} - expected
        require(198 + len(outside) <= 200, "Preserved outside-prefix outcomes exceed the authorized total cap")
        self.pairs(proof["completed_outside_selected"], outside, "preserved outside-prefix outcomes")
        for record in proof["completed_outside_selected"]:
            self.hash(record["source_episode"], record["sha256"])
        self.user_cap = amendment
        for path in (USER_CAP, capped["reuse_provenance"]):
            self.hash(path, sha256(self.path(path)))
        return dict(requested_cap=200, real_inference_episodes=198, composed_real_episodes=297,
                    reused_original_episodes=len(proof["reused_episodes"]), preserved_outside_prefix=len(outside),
                    original_interrupted_run_preserved=True, additional_synthetic_inference_episodes=0)

    def synthetic_reuse(self):
        require(self.user_cap is not None, "User-cap amendment did not pass verification")
        scope = self.user_cap["synthetic_reuse"]
        source, stub, output = (self.path(scope[key]) for key in ("source_run", "stub_run", "composed_run"))
        config, marker = read_json(stub / "run.json"), read_json(stub / "COMPLETED.json")
        derivation, proof = config["derivation"], read_json(self.path(scope["proof"]))
        require(config["methods"] == [] and derivation["kind"] == "reuse_only_no_inference" and
                derivation["additional_inference_calls"] == marker["additional_inference_calls"] == 0,
                "Synthetic stub is not a zero-inference reuse record")
        require(proof["status"] == "complete_reuse_only" and proof["additional_inference_calls"] == 0 and
                proof["new_inference_episodes"] == 0 and proof["scene_count"] == 12 and
                proof["composed_episodes"] == 36 and proof["composed_methods"] == ["F", "G4", "no_verifier"] and
                proof["omitted_methods"] == list(ABLATIONS), "Synthetic reuse proof has an incorrect scope")
        require(self.path(proof["source_run"]) == source and
                self.path(derivation["source_run_configuration"]) == source / "run.json", "Synthetic reuse source differs")
        for record in (derivation, proof):
            for key, path in (("source_run_configuration_sha256", source / "run.json"),
                              ("source_completion_sha256", source / "COMPLETED.json"),
                              ("amendment_sha256", USER_CAP),
                              ("administrative_script_sha256", "scripts/run_synthetic_reuse_only.py")):
                self.hash(path, record[key])
        permitted = {"manifest", "manifest_sha256", "scene_count", "methods", "started_at_unix", "derivation"}
        main_config = read_json(source / "run.json")
        require({k: v for k, v in config.items() if k not in permitted} ==
                {k: v for k, v in main_config.items() if k not in permitted}, "Synthetic reuse altered source settings")
        scenes = read_jsonl(self.path(scope["manifest"]))
        expected = {row["scene_id"] for row in scenes}
        require(len(scenes) == len(expected) == 12, "Synthetic reuse cohort is incomplete")
        truth = read_json(output / "exact_truth_provenance.json")
        require(Counter(row["scene_id"] for row in truth) == Counter(expected), "Synthetic exact-truth reuse grid differs")
        for row in truth:
            filename = row["scene_id"] + ".json"
            require(self.path(row["source"]) == source / "sealed_truth" / filename and
                    self.path(row["destination"]) == output / "sealed_truth" / filename, "Exact-truth reuse paths differ")
            self.hash(row["source"], row["sha256"])
            self.hash(row["destination"], row["sha256"])
        for row in read_json(output / "composition_provenance.json"):
            method = "F" if row["method"] == "no_verifier" else row["method"]
            path = self.path(row["source"])
            require(path.parent == source / row["scene_id"] / method, "Synthetic composition used new inference")
        for path in (scope["proof"], output / "exact_truth_provenance.json"):
            self.hash(path, sha256(self.path(path)))
        return dict(additional_inference_calls=0, new_inference_episodes=0, reused_composed_episodes=36,
                    omitted_methods=list(ABLATIONS), byte_identical_exact_truth_files=12)

    def evaluation(self, run, destination, manifest, expected):
        summary = read_json(destination / "summary.json")
        require(summary["status"] == "complete_evaluation", "Incomplete common evaluation: " + str(destination))
        for key, path in (("run_configuration_sha256", run / "run.json"), ("manifest_sha256", manifest),
                          ("evaluator_code_sha256", self.path("scripts/evaluate_benchmark.py")),
                          ("metrics_code_sha256", self.path("src/sonar/metrics.py"))):
            self.hash(path, summary[key])
        if summary.get("reference_source"):
            self.hash(summary["reference_source"], summary["reference_source_sha256"])
        rows = read_jsonl(destination / "per_episode.jsonl")
        self.pairs(rows, expected, str(destination))
        for filename in ("failed_runs.json", "table_cells.json", "claim_to_evidence.json"):
            read_json(destination / filename)
        for row in rows:
            require(row["execution_status"] in ("completed", "failed"),
                    "Missing/unrecognized execution status: " + row["scene_id"] + "/" + row["method"])
            if row.get("episode_path"):
                self.hash(row["episode_path"], row["episode_sha256"])
            else:
                # A retained failure is an outcome, but its source must exist.
                failure = read_json(run / row["scene_id"] / row["method"] / "failure.json")
                require(row.get("failure") == failure, "Failure record differs from common evaluation")
        return rows

    def run(self, spec):
        name, stem, count, methods, evaluated = spec
        run = self.path("results/" + name)
        manifest = self.path("data/manifests/" + stem + ".jsonl")
        scenes = read_jsonl(manifest)
        ids = {row["scene_id"] for row in scenes}
        require(len(scenes) == count and len(ids) == count, "Unexpected manifest count/duplicate IDs: " + stem)
        expected = {(sid, method) for sid in ids for method in methods}
        config = read_json(run / "run.json")
        require(config["scene_count"] == count and Counter(config["methods"]) == Counter(methods),
                "Run configuration has wrong scene/method counts: " + name)
        self.hash(manifest, config["manifest_sha256"])
        self.hashes(config.get("source_sha256", {}))
        marker_path = run.parent / "COMPLETED.json" if name == "localization/predictions" else run / "COMPLETED.json"
        marker = read_json(marker_path)
        require(bool(marker), "Empty completion marker: " + name)
        if name != "localization/predictions" and not config.get("composition"):
            require(all(marker.get(key) == value for key, value in config.items()), "Completion marker/config differ: " + name)
            require(0 <= marker["execution_failures"] <= marker["new_completed_episodes"] <= len(expected),
                    "Invalid resume completion counters: " + name)
            require(marker.get("finished_at_unix") is not None, "Completion marker lacks finish time: " + name)
        artifacts = {}
        for filename in ("episode.json", "failure.json"):
            for path in run.glob("*/*/" + filename):
                key = (path.parent.parent.name, path.parent.name)
                artifacts.setdefault(key, []).append(path)
        self.pairs([dict(scene_id=k[0], method=k[1]) for k in artifacts], expected, name)
        counts = Counter()
        for key, paths in artifacts.items():
            # Stale failure.json can coexist with a successfully resumed episode.
            chosen = next((p for p in paths if p.name == "episode.json"), paths[0])
            payload = read_json(chosen)
            require(isinstance(payload, dict) and bool(payload), "Empty/invalid retained episode: " + str(chosen))
            require(payload.get("scene_id", key[0]) == key[0] and payload.get("method", key[1]) == key[1],
                    "Episode identity mismatch: " + str(chosen))
            counts["completed" if chosen.name == "episode.json" else "failed"] += 1
        if config.get("composition"):
            provenance = read_json(run / "composition_provenance.json")
            self.pairs(provenance, expected, name + " composition")
            require(marker.get("episodes") == len(expected), "Incomplete composition marker: " + name)
            for record in provenance:
                self.hash(record["source"], record["sha256"])
            self.hashes(config["composition_sources"])
        if evaluated:
            self.run_rows[name] = self.evaluation(run, run / "evaluation", manifest, expected)
            for scene in scenes:
                if scene.get("stratum", "").startswith("synthetic"):
                    truth_path = run / "sealed_truth" / (scene["scene_id"] + ".json")
                    truth = read_json(truth_path)
                    require(truth.get("scene_id") == scene["scene_id"], "Missing/misaligned sealed synthetic truth: " + str(truth_path))
        if name == "development":
            # An unavailable primary comparator is a legitimate frozen decision.
            read_json(run / "evaluation/fixed_selection.json")
        if name == "main":
            for model in ("internvl4b", "internvl8b"):
                self.evaluation(run, run / ("diagnostic_" + model), manifest, expected)
        if name == "sensitivity":
            sensitivity = read_json(run / "evaluation/output_sensitivity.json")
            self.pairs(sensitivity["rows"], expected, "output sensitivity")
            self.hash(manifest, sensitivity["input_manifest_sha256"])
            self.hash(run / "run.json", sensitivity["run_sha256"])
        return dict(scenes=count, methods=len(methods), episodes=len(expected), execution_status_counts=dict(counts))

    def perception(self):
        folder = self.path("results/perception")
        summary = read_json(folder / "summary.json")
        require(summary["status"] == "complete", "Perception training incomplete")
        expected_models = {(method, seed) for method in ("P1", "P2") for seed in (1729, 2718, 3141)}
        records = summary["runs"]
        require(Counter((r["method"], r["seed"]) for r in records) == Counter(expected_models),
                "Expected six perception checkpoints")
        for record in records:
            self.hash(folder / record["checkpoint"], record["checkpoint_sha256"])
        self.hash(folder / "predictions.jsonl", summary["predictions_sha256"])
        scenes = read_jsonl(self.path("data/manifests/real_scenes.jsonl"))
        ids = {r["scene_id"] for r in scenes if r["domain"] == "deepship" and r["split"] == "test"}
        predicted = read_jsonl(folder / "predictions.jsonl")
        require(Counter((r["scene_id"], r["method"], r["seed"]) for r in predicted) ==
                Counter((sid, method, seed) for sid in ids for method, seed in expected_models),
                "Incomplete perception test predictions")
        return dict(checkpoints=len(records), test_predictions=len(predicted))

    def localization(self):
        folder = self.path("results/localization")
        marker = read_json(folder / "COMPLETED.json")
        require(marker["models"] == 6 and marker["evaluation_scenes"] == 114, "Incomplete localizer training/evaluation")
        frozen = read_json(folder / "FROZEN_MODELS.json")
        records = frozen["models"]
        require(records == read_json(folder / "training_records.json"), "Localizer training/frozen checkpoint records differ")
        require(Counter((r["method"], r["seed"]) for r in records) ==
                Counter((method, seed) for method in ("P1_localizer", "P2_localizer") for seed in (17, 29, 43)),
                "Expected six frozen localizer checkpoints")
        self.hash("data/manifests/main_test.jsonl", frozen["evaluation_manifest_sha256"])
        for record in records:
            self.hash(record["checkpoint"], record["checkpoint_sha256"])
        return dict(checkpoints=len(records))

    def annotations(self):
        folder = self.path("results/ai_annotations")
        marker = read_json(folder / "COMPLETED.json")
        require(marker["raw_annotation_records"] == 514 and marker["final_scene_count"] == 257,
                "Incomplete AI annotation marker")
        mapping = {"primary_reference_sha256": "primary_reference.jsonl", "secondary_consensus_sha256": "secondary_consensus.jsonl",
                   "checkpoint_agreement_sha256": "checkpoint_agreement.json", "raw_inventory_sha256": "raw_record_inventory.jsonl",
                   "image_inventory_sha256": "image_inventory.json"}
        for key, filename in mapping.items():
            self.hash(folder / filename, marker[key])
        self.hash("configs/ai_reference_freeze.json", marker["frozen_protocol_sha256"])
        raw = read_jsonl(folder / "raw_record_inventory.jsonl")
        require(len(raw) == 514 and len({r["path"] for r in raw}) == 514, "Incomplete raw annotation inventory")
        for row in raw:
            self.hash(row["path"], row["sha256"])
        self.hashes(read_json(folder / "image_inventory.json"))
        for model, expected_hash in marker["unqualified_diagnostic_reference_sha256"].items():
            self.hash(folder / ("diagnostic_" + model + ".jsonl"), expected_hash)
            self.hash(folder / (model + "_completion_and_heldout_audit.json"), marker["models"][model]["heldout_summary_sha256"])
        status = read_json(self.path("annotations/status.json"))
        require(status["final_raw_annotation_status"] == "completed", "Annotation status is stale")
        self.hash(folder / "COMPLETED.json", status["completion_marker_sha256"])
        return dict(raw_records=len(raw), eligible_types=marker["eligible_types"], independent_ground_truth=False)

    def supplement(self, name, kind):
        run = self.path("results/" + name)
        folder = run / (kind + "_supplement")
        summary = read_json(folder / "summary.json")
        common = self.run_rows.get(name)
        require(common is not None, "Common evaluation did not pass: " + name)
        if kind == "morphology":
            require(summary["status"] == "complete", "Incomplete morphology supplement")
            expected = {(r["scene_id"], r["method"]) for r in common
                        if r["reference_kind"] == "exact_synthetic_component_truth" and r["event_metrics"] is not None}
            self.hash("configs/morphology_supplement.json", summary["protocol_sha256"])
            self.hash(run / "evaluation/per_episode.jsonl", summary["frozen_evaluation_sha256"])
        else:
            require(summary["status"] == "secondary_exact_synthetic_assessment" and summary["primary_scores_modified"] is False,
                    "Incomplete/incorrectly scoped relation supplement")
            expected = {(r["scene_id"], r["method"]) for r in common if r["stratum"].startswith("synthetic")}
            for key, path in (("config_sha256", "configs/relation_supplement.json"),
                              ("source_sha256", "scripts/evaluate_relation_supplement.py"),
                              ("common_summary_sha256", run / "evaluation/summary.json"),
                              ("common_per_episode_sha256", run / "evaluation/per_episode.jsonl"),
                              ("run_configuration_sha256", run / "run.json")):
                self.hash(path, summary[key])
        rows = read_jsonl(folder / "per_episode.jsonl")
        self.pairs(rows, expected, name + " " + kind)
        for row in rows:
            self.hash(run / "sealed_truth" / (row["scene_id"] + ".json"),
                      row["truth_sha256"] if kind == "morphology" else row["exact_truth_sha256"])
        return dict(episodes=len(rows))

    def queue(self):
        folder = self.path("reports/remaining_study")
        marker = read_json(folder / "COMPLETED.json")
        require(marker["status"] == "all_queued_experiments_and_exports_finished", "Study queue not complete")
        if self.path(USER_CAP).is_file():
            require(self.user_cap is not None, "User-cap amendment did not pass verification")
        script = self.user_cap["queue_script"] if self.user_cap is not None else "scripts/run_remaining_study.py"
        self.hash(script, marker["script_sha256"])
        ledger = read_jsonl(folder / "ledger.jsonl")
        successes = {row["phase"]: row for row in ledger if row["returncode"] == 0}
        require(set(marker["phases"]) <= set(successes), "Queue lacks successful ledger entries")
        for phase in marker["phases"]:
            self.hash(successes[phase]["log"], successes[phase]["log_sha256"])
        if self.user_cap is not None:
            self.hash(USER_CAP, marker["amendment_sha256"])
            needed = {"capped_ablation_inference", "capped_ablation_composition", "capped_ablation_evaluation", "synthetic_reuse_only"}
            require(needed <= set(marker["phases"]), "Queue omitted capped inference/composition/evaluation or synthetic reuse")
            for phase in needed:
                self.hash(USER_CAP, successes[phase]["amendment_sha256"])
        return dict(successful_phases=len(marker["phases"]))

    def figures(self, include_manuscript=True):
        manifest = read_json(self.path("figures/manifest.json"))
        require(manifest["no_placeholder_figures"] is True and not manifest["unavailable"], "Some figures are unavailable/placeholders")
        require(Counter(row["figure"] for row in manifest["artifacts"]) == Counter(range(1, 10)), "Expected all nine figures")
        self.hash("scripts/make_figures.py", manifest["code_sha256"])
        self.hashes(manifest["source_files"])
        for artifact in manifest["artifacts"]:
            number = artifact["figure"]
            self.hash(artifact["plot_inputs"], artifact["plot_inputs_sha256"])
            inputs = read_json(self.path(artifact["plot_inputs"]))
            self.hashes(inputs["source_files"])
            self.embedded_hashes(inputs)
            expected = {"figures/figure%02d.%s" % (number, ext) for ext in ("pdf", "png", "svg")}
            require({str(self.path(r["path"]).relative_to(self.root)) for r in artifact["outputs"]} == expected,
                    "Figure lacks PDF/PNG/SVG outputs: " + str(number))
            for output in artifact["outputs"]:
                self.hash(output["path"], output["sha256"])
                path = self.path(output["path"])
                with path.open("rb") as stream:
                    prefix = stream.read(4096)
                signature = {".pdf": prefix.startswith(b"%PDF"), ".png": prefix.startswith(b"\x89PNG\r\n\x1a\n"),
                             ".svg": b"<svg" in prefix}
                require(signature[path.suffix], "Invalid figure file signature: " + str(path))
                if path.suffix == ".pdf" and include_manuscript and number <= 7:
                    self.hash("manuscript/" + path.name, output["sha256"])
        return dict(figures=9, figure_files=27, required_manuscript_pdf_copies=7 if include_manuscript else "not_checked_gpu_only")

    def viewer(self):
        folder = self.path("viewer")
        manifest = read_json(folder / "case_manifest.json")
        bundle = read_json(folder / "viewer_bundle.json")
        require(bundle["cases"] and bundle["offline"] is True and bundle["annotation_completion_marker_present"] is True,
                "Final viewer is empty or lacks completed annotation provenance")
        require(manifest["complete_coordinate_mapping"] is True, "Viewer coordinate exports incomplete")
        required = {"index.html", "app.js", "style.css", "README.txt", "viewer_bundle.json"}
        require(required <= set(manifest["files_sha256"]), "Missing standalone viewer files")
        for name, expected in manifest["files_sha256"].items():
            path = (folder / name).resolve()
            require(folder in path.parents, "Viewer inventory path escapes bundle")
            self.hash(path, expected)
        self.hashes(manifest["sources"])
        self.hashes(bundle["source_files"])
        self.hash("scripts/build_interactive_viewer.py", manifest["builder_sha256"])
        require("__CASE_DATA__" not in (folder / "index.html").read_text(), "Unfilled viewer HTML template")
        def assets(value):
            if isinstance(value, dict):
                if "url" in value and "sha256" in value:
                    require(not re.match(r"[A-Za-z][A-Za-z0-9+.-]*:", value["url"]), "Viewer asset is not offline")
                    path = (folder / value["url"]).resolve()
                    require(folder in path.parents, "Viewer asset path escapes bundle")
                    self.hash(path, value["sha256"])
                for child in value.values():
                    assets(child)
            elif isinstance(value, list):
                for child in value:
                    assets(child)
        assets(bundle["cases"])
        smoke = read_json(folder / "browser_check.json")
        require(smoke["status"] == "passed" and smoke["coordinateMapping"] == "tested" and smoke["runtimeExceptions"] == 0,
                "Final viewer browser smoke did not pass")
        require(smoke["sourceScene"] in {case["scene_id"] for case in bundle["cases"]},
                "Browser smoke inspected a different viewer case")
        require(all((folder / "browser_check.json").stat().st_mtime + 2 >= (folder / filename).stat().st_mtime
                    for filename in ("viewer_bundle.json", "index.html", "app.js", "style.css", "case_manifest.json")),
                "Browser smoke predates the final viewer; rerun it on this preserved bundle")
        return dict(cases=len(bundle["cases"]), inventoried_files=len(manifest["files_sha256"]), browser_smoke="passed")

    def manuscript(self):
        main = self.path("manuscript/main.tex")
        text = main.read_text()
        visited = {main}
        def expand(match):
            path = main.parent / match.group(1)
            if not path.suffix:
                path = path.with_suffix(".tex")
            path = self.path(path)
            require(path not in visited, "Recursive/duplicate manuscript input: " + str(path))
            visited.add(path)
            return re.sub(r"\\(?:input|include)\s*\{([^}]+)\}", expand, strip_tex_comments(path.read_text()))
        text = re.sub(r"\\(?:input|include)\s*\{([^}]+)\}", expand, strip_tex_comments(text))
        body = manuscript_text_checks(text)
        require({"figure%02d.pdf" % i for i in range(1, 8)} <= set(re.findall(r"\\researchfigure\s*\{([^}]+)\}", body)),
                "Manuscript does not include its seven main research figures")
        pdf = main.with_suffix(".pdf")
        result = subprocess.run(["pdfinfo", str(pdf)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, flags=re.M)
        require(match is not None and 8 <= int(match.group(1)) <= 10, "Final manuscript PDF must have 8–10 pages")
        sources = set(visited) | set(main.parent.glob("*.bib")) | set(main.parent.glob("figure*.pdf"))
        require(all(pdf.stat().st_mtime + 2 >= p.stat().st_mtime for p in sources), "PDF predates manuscript source/figure edits; compile again")
        return dict(pages=int(match.group(1)), pdf_sha256=sha256(pdf), finalresults=True, unresolved_pending_calls=0)

    def verify(self, gpu_only=False):
        self.section("frozen_protocol_hashes", self.protocol)
        if self.path(USER_CAP).is_file():
            self.section("user_authorized_ablation_cap", self.cap_amendment)
        self.section("ai_annotation_completion", self.annotations)
        self.section("perception_training", self.perception)
        self.section("localization_training", self.localization)
        for spec in self.run_specs():
            self.section("run:" + spec[0], lambda spec=spec: self.run(spec))
        if self.path(USER_CAP).is_file():
            self.section("synthetic_ablation_reuse_only", self.synthetic_reuse)
        for kind, field in (("morphology", "scope"), ("relation", "eligible_run_names")):
            try:
                names = read_json(self.path("configs/" + kind + "_supplement.json"))[field]
            except (OSError, ValueError, KeyError) as exc:
                self.sections.append(dict(name=kind + "_scope", status="failed", error=str(exc)))
                continue
            for name in names:
                self.section(kind + ":" + name, lambda name=name, kind=kind: self.supplement(name, kind))
        final_checks = [("remaining_study_queue", self.queue), ("nine_figures", lambda: self.figures(include_manuscript=not gpu_only)),
                        ("standalone_final_viewer", self.viewer)]
        if not gpu_only:
            final_checks.append(("final_manuscript", self.manuscript))
        for name, callback in final_checks:
            self.section(name, callback)
        return dict(status="passed" if all(s["status"] == "passed" for s in self.sections) else "unfinished_or_inconsistent",
                    verified_at_utc=datetime.now(timezone.utc).isoformat(), root=str(self.root),
                    verifier_sha256=sha256(Path(__file__)), scope="gpu_deliverables" if gpu_only else "all_final_deliverables",
                    manuscript_check_skipped=gpu_only, execution="CPU administrative file/schema/hash checks only; no score recomputation",
                    retained_model_failures_allowed=True, empty_ai_eligibility_allowed=True,
                    sections=self.sections, verified_file_sha256=self.checked_hashes)


def snapshot_sources(root, verification_path):
    """Called only after successful verification and explicit writers-finished flag."""
    root = Path(root).resolve()
    verification = read_json(verification_path)
    require(verification.get("status") == "passed" and verification.get("scope") == "all_final_deliverables",
            "Final snapshot requires a successful full verification report")
    Verifier(root).hashes(verification["verified_file_sha256"])
    final = root / "provenance/final_source"
    require(not final.exists(), "Refusing to overwrite existing final_source snapshot")
    paths = set()
    for directory in ("src", "scripts", "tests", "configs"):
        require((root / directory).is_dir(), "Missing snapshot source directory: " + directory)
        paths.update(p for p in (root / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo"))
    # Post-GPU presentation repairs depend on preserved vector layouts and
    # already computed display arrays. Keep these and their audit receipts in
    # the final package so the revised PDFs remain reproducible after pod stop.
    for directory in ("figures", "reports", "provenance/presentation_versions", "provenance/final_administrative_review"):
        if (root / directory).is_dir():
            paths.update(p for p in (root / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo"))
    for pattern in ("requirements*.txt", "README*", "pyproject.toml", "underwater_acoustic_agent_research_plan.md",
                    "manuscript/*.tex", "manuscript/*.bib", "manuscript/*.cls", "manuscript/*.bst", "manuscript/*.sty",
                    "manuscript/main.pdf", "manuscript/figure*.pdf", "figures/figure*.pdf", "figures/manifest.json",
                    "figures/inputs/*.json", "figures/*selection*.json"):
        paths.update(root.glob(pattern))
    for required in ("README.md", "underwater_acoustic_agent_research_plan.md", "manuscript/main.tex", "manuscript/main.pdf"):
        require(root / required in paths, "Missing required snapshot file: " + required)
    require(paths, "No source files selected")
    final.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".final_source.building-", dir=str(final.parent)))
    try:
        inventory = []
        for source in sorted(paths):
            require(not source.is_symlink(), "Snapshot source is a symlink: " + str(source))
            relative = source.relative_to(root)
            before = sha256(source)
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(destination))
            require(before == sha256(destination) == sha256(source), "Source changed during snapshot: " + str(relative))
            inventory.append(dict(path=relative.as_posix(), bytes=destination.stat().st_size, sha256=before))
        # Recheck every source at the end to catch writes after its own copy.
        for row in inventory:
            require(sha256(root / row["path"]) == row["sha256"], "Source changed while snapshot was assembling: " + row["path"])
        manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), kind="final_source_and_manuscript_snapshot",
                        source_root=str(root), writer_completion_explicitly_confirmed=True,
                        verification_report_sha256=sha256(verification_path), files=inventory,
                        scope="Current source, tests, configs, requirements, README, original plan, final manuscript, all figure formats/display inputs, administrative audit reports and preserved presentation versions; raw results preserved separately")
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        shutil.copy2(str(verification_path), str(stage / "verification_report.json"))
        require(not final.exists(), "Snapshot destination appeared during copy")
        stage.rename(final)
    except BaseException:
        shutil.rmtree(str(stage))
        raise
    return dict(status="complete", snapshot="provenance/final_source", files=len(inventory),
                total_bytes=sum(r["bytes"] for r in inventory), manifest_sha256=sha256(final / "manifest.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("reports/final_deliverables_verification.json"))
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--gpu-only", action="store_true", help="Verify GPU outputs and final viewer; skip manuscript finalization/page count")
    parser.add_argument("--writers-finished", action="store_true", help="Coordinator explicitly confirms no source/manuscript writers remain")
    args = parser.parse_args()
    if args.snapshot and not args.writers_finished:
        parser.error("--snapshot requires explicit --writers-finished; wait for all writers")
    if args.snapshot and args.gpu_only:
        parser.error("Final source/manuscript snapshot requires the full verification, without --gpu-only")
    verifier = Verifier(args.project_root)
    output = verifier.path(args.output)
    require("artifact_receipt" not in output.name and "provenance/final_source" not in output.as_posix(),
            "Verification output cannot overwrite the receipt or final snapshot")
    report = verifier.verify(gpu_only=args.gpu_only)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    failed = [section for section in report["sections"] if section["status"] != "passed"]
    print(json.dumps(dict(status=report["status"], report=str(output), failed_sections=failed), indent=2))
    if failed:
        return 1
    if args.snapshot:
        snapshot = snapshot_sources(verifier.root, output)
        (verifier.root / "reports/final_source_snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
        print(json.dumps(snapshot, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
