"""CPU administrative fixtures for the explicit user-authorized runtime cap."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


v = module("verify_cap_fixture", "verify_final_deliverables.py")
prepare = module("prepare_cap_fixture", "prepare_capped_ablation.py")


class UserCapVerification(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sonar-cap-verifier-", dir="/tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checker = v.Verifier(self.root)
        self.amendment = v.read_json(ROOT / v.USER_CAP)
        self.records = [dict(scene_id="fixture%02d" % i, domain="deepship" if i < 30 else "mbari", stratum="real")
                        for i in range(80)]
        original = self.amendment["original"]
        manifest = self.write_lines(original["manifest"], self.records)
        runner = self.write("scripts/run_benchmark.py", {"inert_fixture": True})
        script = self.root / "scripts/prepare_capped_ablation.py"
        script.write_bytes((ROOT / "scripts/prepare_capped_ablation.py").read_bytes())
        frozen = {original["manifest"]: v.sha256(manifest), "scripts/run_benchmark.py": v.sha256(runner)}
        self.write("configs/final_protocol_freeze.json", {"sha256": frozen})
        config = dict(manifest=original["manifest"], manifest_sha256=v.sha256(manifest), scene_count=80,
                      methods=list(v.ABLATIONS), source_sha256={"scripts/run_benchmark.py": v.sha256(runner)},
                      started_at_unix=1, budget=6, model="inert-fixture", workers=16, max_batch_size=8, batch_wait_ms=2000)
        old_config = self.write(original["run"] + "/run.json", config)
        for sid, method in (("fixture00", v.ABLATIONS[0]), ("fixture01", v.ABLATIONS[1])):
            self.write(original["run"] + "/" + sid + "/" + method + "/episode.json",
                       dict(scene_id=sid, method=method, schema_valid=False, report=None, raw_report=None,
                            generations=[], actions=[], final_state={}))
        self.write(original["run"] + "/fixture02/" + v.ABLATIONS[0] + "/partial.json", {"partial": True})
        cap = self.amendment["capped"]
        self.proof = prepare.prepare(self.root, original["run"], cap["run"], original["manifest"],
                                     cap["manifest"], "configs/final_protocol_freeze.json", writers_stopped=True)
        original["manifest_sha256"] = v.sha256(manifest)
        original["run_configuration_sha256"] = v.sha256(old_config)
        cap["manifest_sha256"] = v.sha256(self.root / cap["manifest"])
        interruption = self.write(self.amendment["original_interruption"]["path"],
                                  dict(planned_cap=198, original_run_sha256=v.sha256(old_config), scientific_hash_mismatches=0))
        self.amendment["original_interruption"]["sha256"] = v.sha256(interruption)
        synthetic = self.amendment["synthetic_reuse"]
        self.synthetic_scenes = [dict(scene_id="synthetic%02d" % i, stratum="synthetic") for i in range(12)]
        synthetic["manifest_sha256"] = v.sha256(self.write_lines(synthetic["manifest"], self.synthetic_scenes))
        self.write(v.USER_CAP, self.amendment)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n")
        return path

    def write_lines(self, name, rows):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def synthetic_fixture(self):
        scope = self.amendment["synthetic_reuse"]
        source = scope["source_run"]
        config = dict(v.read_json(self.root / self.amendment["original"]["run"] / "run.json"),
                      methods=["F", "G4"], scene_count=114)
        original = self.write(source + "/run.json", config)
        completed = self.write(source + "/COMPLETED.json", {"complete": True})
        script = self.write("scripts/run_synthetic_reuse_only.py", {"inert_administrative_fixture": True})
        derivation = dict(kind="reuse_only_no_inference", additional_inference_calls=0,
                          source_run_configuration=source + "/run.json", source_run_configuration_sha256=v.sha256(original),
                          source_completion_sha256=v.sha256(completed), amendment_sha256=v.sha256(self.root / v.USER_CAP),
                          administrative_script_sha256=v.sha256(script))
        stub = dict(config, methods=[], scene_count=12, manifest=scope["manifest"],
                    manifest_sha256=scope["manifest_sha256"], derivation=derivation)
        self.write(scope["stub_run"] + "/run.json", stub)
        self.write(scope["stub_run"] + "/COMPLETED.json", dict(stub, new_completed_episodes=0,
                   execution_failures=0, finished_at_unix=2, additional_inference_calls=0))
        proof = dict(derivation, status="complete_reuse_only", new_inference_episodes=0,
                     source_run=source, scene_count=12, composed_episodes=36,
                     composed_methods=["F", "G4", "no_verifier"], omitted_methods=list(v.ABLATIONS))
        self.write(scope["proof"], proof)
        truth, composition = [], []
        for scene in self.synthetic_scenes:
            sid = scene["scene_id"]
            src = self.write(source + "/sealed_truth/" + sid + ".json", {"scene_id": sid, "events": []})
            dst = self.write(scope["composed_run"] + "/sealed_truth/" + sid + ".json", {"scene_id": sid, "events": []})
            truth.append(dict(scene_id=sid, source=str(src), destination=str(dst), sha256=v.sha256(src)))
            for method in scope["methods"]:
                composition.append(dict(scene_id=sid, method=method,
                                        source=source + "/" + sid + "/" + ("F" if method == "no_verifier" else method) + "/episode.json"))
        self.write(scope["composed_run"] + "/exact_truth_provenance.json", truth)
        self.write(scope["composed_run"] + "/composition_provenance.json", composition)

    def test_explicit_amendment_and_all_unchanged_legacy_grids(self):
        result = self.checker.cap_amendment()
        self.assertEqual(result["reused_original_episodes"], 2)
        self.assertEqual(result["real_inference_episodes"], 198)
        amended = {spec[0]: spec for spec in self.checker.run_specs()}
        self.assertEqual(amended["synthetic_ablations"][3], ("F", "G4", "no_verifier"))
        for spec in v.RUNS:
            if spec[0] not in v.CAPPED_RUNS:
                self.assertEqual(amended[spec[0]], spec)
        (self.root / v.USER_CAP).unlink()
        self.assertEqual(self.checker.run_specs(), v.RUNS)

    def test_completed_original_is_rejected(self):
        self.write("results/ablation_inference/COMPLETED.json", {"incorrect": True})
        with self.assertRaisesRegex(v.VerificationError, "must not claim completion"):
            self.checker.cap_amendment()

    def test_changed_inference_behavior_is_rejected(self):
        path = self.root / "results/ablation_inference_capped/run.json"
        config = v.read_json(path)
        config["budget"] = 7
        self.write(str(path), config)
        with self.assertRaisesRegex(v.VerificationError, "frozen inference settings"):
            self.checker.cap_amendment()

    def test_reordered_prefix_cannot_be_authorized_by_rehashing(self):
        selected = self.records[:33]
        selected[0], selected[1] = selected[1], selected[0]
        path = self.write_lines(self.amendment["capped"]["manifest"], selected)
        self.amendment["capped"]["manifest_sha256"] = v.sha256(path)
        self.write(v.USER_CAP, self.amendment)
        with self.assertRaisesRegex(v.VerificationError, "first 33 frozen"):
            self.checker.cap_amendment()

    def test_added_original_file_is_rejected(self):
        self.write("results/ablation_inference/late_partial.json", {"late": True})
        with self.assertRaisesRegex(v.VerificationError, "added or removed"):
            self.checker.cap_amendment()

    def test_reused_schema_rejection_is_preserved_and_tampering_fails(self):
        episode = self.root / self.proof["reused_episodes"][0]["destination_episode"]
        self.assertIs(v.read_json(episode)["schema_valid"], False)
        self.write(str(episode), {"schema_valid": True})
        with self.assertRaisesRegex(v.VerificationError, "SHA256 mismatch"):
            self.checker.cap_amendment()

    def test_duplicate_or_missing_preparation_pairs_are_rejected(self):
        self.proof["incomplete_selected_jobs"].append(self.proof["incomplete_selected_jobs"][0])
        self.write(self.amendment["capped"]["reuse_provenance"], self.proof)
        with self.assertRaisesRegex(v.VerificationError, "duplicates=1"):
            self.checker.cap_amendment()

    def test_synthetic_reuse_has_zero_inference_and_complete_truth(self):
        self.checker.cap_amendment()
        self.synthetic_fixture()
        result = self.checker.synthetic_reuse()
        self.assertEqual(result["byte_identical_exact_truth_files"], 12)
        self.checker.run(v.CAPPED_RUNS["synthetic_ablation_inference"])

    def test_synthetic_new_inference_is_rejected(self):
        self.checker.cap_amendment()
        self.synthetic_fixture()
        path = "results/synthetic_ablation_inference/run.json"
        config = v.read_json(self.root / path)
        config["methods"] = ["no_demon"]
        self.write(path, config)
        with self.assertRaisesRegex(v.VerificationError, "not a zero-inference"):
            self.checker.synthetic_reuse()

    def test_synthetic_changed_truth_is_rejected(self):
        self.checker.cap_amendment()
        self.synthetic_fixture()
        self.write("results/synthetic_ablations/sealed_truth/synthetic00.json", {"scene_id": "synthetic00", "changed": True})
        with self.assertRaisesRegex(v.VerificationError, "SHA256 mismatch"):
            self.checker.synthetic_reuse()


if __name__ == "__main__":
    unittest.main()
