"""Isolated CPU/file fixtures; run the real composer, mock only GPU evaluation."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class SyntheticReuseOnly(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sonar-synthetic-reuse-admin-", dir="/tmp")
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        for name in ("run_synthetic_reuse_only.py", "compose_ablations.py"):
            shutil.copy2(ROOT / "scripts" / name, self.root / "scripts" / name)
        spec = importlib.util.spec_from_file_location("synthetic_reuse_fixture", self.root / "scripts/run_synthetic_reuse_only.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.source = self.root / "results/main"
        self.stub = self.root / "results/synthetic_ablation_inference"
        self.output = self.root / "results/synthetic_ablations"
        self.scenes = ["fixture_s%02d" % i for i in range(12)]
        self.manifest = "data/manifests/synthetic_ablation_test.jsonl"
        path = self.root / self.manifest
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps({"scene_id": sid, "stratum": "synthetic", "fixture_only": True}) + "\n" for sid in self.scenes))
        self.write(self.source / "run.json", {"manifest": self.manifest, "manifest_sha256": self.sha(path),
            "scene_count": 12, "methods": ["F", "G4"], "started_at_unix": 1,
            "source_sha256": {"scripts/compose_ablations.py": self.sha(self.root / "scripts/compose_ablations.py")},
            "budget": 6, "numerical_device": "cuda:0", "decoding": "greedy"})
        self.write(self.source / "COMPLETED.json", {"fixture_only": True, "completed": True})
        for index, sid in enumerate(self.scenes):
            for method in ("F", "G4"):
                valid = not (index == 0 and method == "F")
                raw = {"components": [{"id": "fixture_component"}], "relations": [], "claims": [],
                       "unresolved_questions": [], "abstained": False}
                self.write(self.source / sid / method / "episode.json", {"scene_id": sid, "method": method,
                    "schema_valid": valid, "schema_errors": [] if valid else ["fixture_invalid"],
                    "raw_report": raw, "report": raw if valid else None,
                    "model_gpu_seconds": 12.5, "tool_gpu_seconds": 2.5,
                    "generations": [{"fixture_only": True}], "verification_audit": [{"status": "fixture_only"}]})
            truth = self.source / "sealed_truth" / (sid + ".json")
            truth.parent.mkdir(exist_ok=True)
            truth.write_text(json.dumps({"scene_id": sid, "fixture_only": True, "events": []}, indent=3) + "\n")
        self.reuse = {"manifest": self.manifest, "manifest_sha256": self.sha(path),
            "source_run": "results/main", "stub_run": "results/synthetic_ablation_inference",
            "composed_run": "results/synthetic_ablations", "proof": "results/synthetic_ablation_inference/reuse_only_provenance.json",
            "scene_count": 12, "new_inference_episodes": 0, "methods": ["F", "G4", "no_verifier"],
            "composed_episodes": 36, "omitted_methods": ["no_demon", "no_graph", "no_quantitative", "forced_exclusivity", "fixed_resolution", "random"]}
        self.write(self.root / "configs/ablation_user_cap.json", {"synthetic_reuse": self.reuse})
        self.write(self.root / "configs/final_protocol_freeze.json", {"sha256": {
            self.manifest: self.sha(path), "scripts/compose_ablations.py": self.sha(self.root / "scripts/compose_ablations.py")}})
        self.calls = []
        self.real_run = subprocess.run

    def tearDown(self):
        self.temp.cleanup()

    def sha(self, path):
        return self.module.sha(path)

    @staticmethod
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n")

    @staticmethod
    def read(path):
        return json.loads(path.read_text())

    def files(self, folder):
        return {p.relative_to(folder).as_posix(): self.sha(p) for p in sorted(folder.rglob("*")) if p.is_file()}

    def invoke(self):
        previous_cwd = Path.cwd()
        previous_pythonpath = os.environ.get("PYTHONPATH")
        def dispatch(argv, **kwargs):
            self.calls.append(list(argv))
            if argv[1] == "scripts/compose_ablations.py":
                # Execute the actual unchanged script in a separate CPU process.
                return self.real_run(argv, cwd=str(self.root), stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, timeout=20, **kwargs)
            if argv[1] == "scripts/evaluate_benchmark.py":
                self.assertEqual(argv, [sys.executable, "scripts/evaluate_benchmark.py", "--run",
                    self.reuse["composed_run"], "--references", "results/ai_annotations/primary_reference.jsonl"])
                return subprocess.CompletedProcess(argv, 0)
            self.fail("Unexpected external invocation, including any inference: " + repr(argv))
        try:
            with patch.object(self.module.subprocess, "run", side_effect=dispatch):
                self.module.main()
        finally:
            os.chdir(previous_cwd)
            if previous_pythonpath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = previous_pythonpath

    def test_36_reports_schema_gate_truth_hashes_and_zero_inference(self):
        source_before = self.files(self.source)
        self.invoke()
        self.assertEqual(source_before, self.files(self.source))
        self.assertEqual(len(list(self.output.glob("*/*/episode.json"))), 36)
        stub, proof = self.read(self.stub / "run.json"), self.read(self.stub / "reuse_only_provenance.json")
        self.assertEqual(stub["methods"], [])
        self.assertEqual(stub["derivation"]["kind"], "reuse_only_no_inference")
        self.assertEqual(stub["derivation"]["additional_inference_calls"], 0)
        self.assertEqual(proof["additional_inference_calls"], 0)
        self.assertEqual(proof["new_inference_episodes"], 0)
        self.assertEqual(proof["composed_episodes"], 36)
        marker = self.read(self.stub / "COMPLETED.json")
        self.assertEqual(marker["new_completed_episodes"], 0)
        self.assertEqual(marker["additional_inference_calls"], 0)
        self.assertFalse(list(self.stub.glob("*/*/episode.json")))
        for index, sid in enumerate(self.scenes):
            for method in ("F", "G4"):
                self.assertEqual((self.output / sid / method / "episode.json").read_bytes(),
                                 (self.source / sid / method / "episode.json").read_bytes())
            original = self.read(self.source / sid / "F/episode.json")
            derived = self.read(self.output / sid / "no_verifier/episode.json")
            self.assertEqual(derived["report"], original["raw_report"] if original["schema_valid"] else None)
            self.assertEqual(derived["schema_valid"], original["schema_valid"])
            self.assertFalse(derived["verification_applied"])
            self.assertEqual(derived["derivation"]["additional_inference_calls"], 0)
            self.assertEqual(derived["derivation"]["additional_gpu_seconds"], 0)
            self.assertEqual(derived["model_gpu_seconds"], original["model_gpu_seconds"])
            self.assertEqual(derived["tool_gpu_seconds"], original["tool_gpu_seconds"])
            self.assertEqual((self.output / "sealed_truth" / (sid + ".json")).read_bytes(),
                             (self.source / "sealed_truth" / (sid + ".json")).read_bytes())
        truths = self.read(self.output / "exact_truth_provenance.json")
        self.assertEqual(len(truths), 12)
        for entry in truths:
            self.assertEqual(self.sha(self.root / entry["source"]), entry["sha256"])
            self.assertEqual(self.sha(self.root / entry["destination"]), entry["sha256"])
        self.assertEqual(self.read(self.output / "COMPLETED.json")["additional_inference_calls"], 0)
        self.assertEqual(self.read(self.output / "run.json")["methods"], ["F", "G4", "no_verifier"])
        self.assertEqual([call[1] for call in self.calls], ["scripts/compose_ablations.py", "scripts/evaluate_benchmark.py"])
        self.assertNotIn("torch", sys.modules)

    def test_rerun_is_byte_idempotent(self):
        self.invoke()
        before = self.files(self.root / "results")
        self.invoke()
        self.assertEqual(before, self.files(self.root / "results"))

    def test_tampered_reuse_proof_is_rejected(self):
        self.invoke()
        path = self.stub / "reuse_only_provenance.json"
        proof = self.read(path); proof["additional_inference_calls"] = 1; self.write(path, proof)
        before = self.files(self.root / "results"); calls = len(self.calls)
        with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite different artifact"):
            self.invoke()
        self.assertEqual(before, self.files(self.root / "results"))
        self.assertEqual(len(self.calls), calls)

    def test_tampered_stub_source_provenance_is_rejected(self):
        self.invoke()
        path = self.stub / "run.json"
        config = self.read(path); config["derivation"]["source_run_configuration_sha256"] = "0" * 64; self.write(path, config)
        with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite different artifact"):
            self.invoke()

    def test_tampered_truth_inventory_is_rejected_before_evaluation(self):
        self.invoke()
        path = self.output / "exact_truth_provenance.json"
        rows = self.read(path); rows[0]["sha256"] = "0" * 64; self.write(path, rows)
        calls = len(self.calls)
        with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite different artifact"):
            self.invoke()
        self.assertEqual([call[1] for call in self.calls[calls:]], ["scripts/compose_ablations.py"])

    def test_tampered_truth_bytes_are_not_overwritten(self):
        self.invoke()
        path = self.output / "sealed_truth" / (self.scenes[0] + ".json")
        path.write_text('{"tampered": true}\n'); before = path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "Existing exact truth differs"):
            self.invoke()
        self.assertEqual(path.read_bytes(), before)

    def test_tampered_composition_provenance_rejected_by_actual_composer(self):
        self.invoke()
        path = self.output / "composition_provenance.json"
        records = self.read(path); records[0]["sha256"] = "0" * 64; self.write(path, records)
        calls = len(self.calls)
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke()
        self.assertEqual([call[1] for call in self.calls[calls:]], ["scripts/compose_ablations.py"])

    def test_frozen_source_tamper_is_rejected_before_composition(self):
        (self.root / "scripts/compose_ablations.py").write_text("# tampered fixture\n")
        with self.assertRaisesRegex(RuntimeError, "Frozen input changed"):
            self.invoke()
        self.assertFalse(self.stub.exists())
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
