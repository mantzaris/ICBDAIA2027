"""CPU-only isolated administrative fixtures; no research computations."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cap_admin", ROOT / "scripts/prepare_capped_ablation.py")
cap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cap)


class CappedPreparation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sonar-cap-admin-", dir="/tmp")
        self.root = Path(self.temp.name)
        self.manifest = Path("data/manifests/original.jsonl")
        self.capped = Path("data/manifests/capped.jsonl")
        self.source = Path("results/original")
        self.output = Path("results/capped")
        self.freeze = Path("configs/freeze.json")
        for path in [self.manifest.parent, self.source, self.freeze.parent, Path("scripts")]:
            (self.root / path).mkdir(parents=True, exist_ok=True)
        self.records = [{"scene_id": "s%d" % i, "stratum": "real", "domain": "fixture"} for i in range(5)]
        (self.root / self.manifest).write_text("".join(json.dumps(r) + "\n" for r in self.records))
        (self.root / "scripts/run_benchmark.py").write_text("# isolated inert source fixture\n")
        frozen = {str(p): cap.digest(self.root / p) for p in [self.manifest, Path("scripts/run_benchmark.py")]}
        cap.write(self.root / self.freeze, {"sha256": frozen})
        self.config = {"manifest": str(self.manifest), "manifest_sha256": frozen[str(self.manifest)],
            "scene_count": 5, "methods": cap.METHODS, "source_sha256": {"scripts/run_benchmark.py": frozen["scripts/run_benchmark.py"]},
            "started_at_unix": 10, "budget": 6, "model": "models/fixture", "workers": 16,
            "max_batch_size": 8, "batch_wait_ms": 2000}
        cap.write(self.root / self.source / "run.json", self.config)

    def tearDown(self):
        self.temp.cleanup()

    def episode(self, scene, method, valid=True):
        path = self.root / self.source / scene / method
        path.mkdir(parents=True, exist_ok=True)
        cap.write(path / "episode.json", {"scene_id": scene, "method": method, "schema_valid": valid,
            "report": {"components": []}, "raw_report": {}, "generations": [], "actions": [], "final_state": {}})
        (path / "global_fixture.png").write_bytes(b"inert fixture asset")
        return path

    def prepare(self, **kwargs):
        return cap.prepare(self.root, self.source, self.output, self.manifest, self.capped,
                           self.freeze, cap=14, **kwargs)

    def test_manifest_selection_is_exact_prefix_without_run_outcomes(self):
        (self.root / self.source / "run.json").unlink()
        result = self.prepare(manifest_only=True)
        lines = (self.root / self.manifest).read_bytes().splitlines(keepends=True)
        self.assertEqual((self.root / self.capped).read_bytes(), b"".join(lines[:2]))
        self.assertEqual(result["selected_scene_ids"], ["s0", "s1"])
        self.assertEqual(result["expected_inference_episodes"], 12)
        self.assertFalse((self.root / self.output).exists())

    def test_stopped_guard_and_live_pid_guard(self):
        with self.assertRaisesRegex(ValueError, "writers-stopped"):
            self.prepare()
        with self.assertRaisesRegex(ValueError, "still alive"):
            self.prepare(writers_stopped=True, writer_pid=os.getpid())
        self.assertFalse((self.root / self.output).exists())

    def test_copies_complete_rejected_reports_and_preserves_partial_original(self):
        self.episode("s0", cap.METHODS[0], valid=False)
        self.episode("s1", cap.METHODS[1])
        partial = self.root / self.source / "s1" / cap.METHODS[2]
        partial.mkdir(parents=True); (partial / "episode.json").write_text('{"interrupted":')
        (partial / "action_0.png").write_bytes(b"partial retained")
        before = cap.inventory(self.root / self.source)
        result = self.prepare(writers_stopped=True)
        self.assertEqual(before, cap.inventory(self.root / self.source))
        self.assertEqual(len(result["reused_episodes"]), 2)
        self.assertFalse((self.root / self.output / "s1" / cap.METHODS[2]).exists())
        self.assertFalse(cap.read(self.root / self.output / "s0" / cap.METHODS[0] / "episode.json")["schema_valid"])
        config = cap.read(self.root / self.output / "run.json")
        self.assertEqual({k for k in config if config[k] != self.config[k]},
                         {"manifest", "manifest_sha256", "scene_count", "started_at_unix"})
        for item in result["reused_episodes"]:
            for entry in item["files"]:
                self.assertEqual(cap.digest(self.root / entry["source"]), entry["sha256"])
                self.assertEqual(cap.digest(self.root / entry["destination"]), entry["sha256"])
        # Execute only the frozen runner's pure administrative resume helper.
        tree = ast.parse((ROOT / "scripts/run_benchmark.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "freeze_or_resume")
        module = ast.Module(body=[function], type_ignores=[])
        namespace = {"Path": Path, "json": json}
        exec(compile(module, "frozen_resume_helper", "exec"), namespace)
        candidate = dict(config, started_at_unix=config["started_at_unix"] + 1)
        self.assertEqual(namespace["freeze_or_resume"](self.root / self.output, candidate), config)

    def test_existing_destination_is_never_overwritten(self):
        (self.root / self.output).mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.prepare(writers_stopped=True)

    def test_already_completed_outside_prefix_cannot_breach_cap(self):
        for method in cap.METHODS[:3]:
            self.episode("s4", method)
        with self.assertRaisesRegex(ValueError, "exceed the authorized cap"):
            self.prepare(writers_stopped=True)
        self.assertFalse((self.root / self.output).exists())

    def test_original_mutation_during_copy_is_detected(self):
        self.episode("s0", cap.METHODS[0])
        original = cap.shutil.copy2
        def mutation(src, dst, **kwargs):
            value = original(src, dst, **kwargs)
            (self.root / self.source / "late_writer.txt").write_text("unexpected writer")
            return value
        with patch.object(cap.shutil, "copy2", side_effect=mutation):
            with self.assertRaisesRegex(ValueError, "changed during preparation"):
                self.prepare(writers_stopped=True)
        self.assertFalse((self.root / self.output).exists())

    def test_frozen_hash_mismatch_fails_before_publication(self):
        (self.root / "scripts/run_benchmark.py").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "Frozen source changed"):
            self.prepare(writers_stopped=True)
        self.assertFalse((self.root / self.output).exists())


if __name__ == "__main__":
    unittest.main()
