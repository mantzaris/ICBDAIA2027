"""Administrative completion checks; no scientific imports or GPU work."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "scripts/verify_final_deliverables.py"
SPEC = importlib.util.spec_from_file_location("verify_final_deliverables", MODULE)
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checker = v.Verifier(self.root)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def raw_run(self):
        manifest = self.root / "data/manifests/toy.jsonl"
        manifest.parent.mkdir(parents=True)
        manifest.write_text('{"scene_id":"s1"}\n{"scene_id":"s2"}\n')
        config = dict(scene_count=2, methods=["F"], manifest_sha256=v.sha256(manifest))
        self.write("results/toy/run.json", config)
        # Resumed completion counter deliberately differs from complete cohort.
        self.write("results/toy/COMPLETED.json", dict(config, new_completed_episodes=1, execution_failures=1, finished_at_unix=1))
        self.write("results/toy/s1/F/episode.json", dict(scene_id="s1", method="F", schema_valid=False, report=None))
        self.write("results/toy/s2/F/failure.json", dict(scene_id="s2", method="F", error="retained model error"))
        return ("toy", "toy", 2, ("F",), False)

    def test_retained_failures_are_complete_and_resume_counter_is_not_total(self):
        evidence = self.checker.run(self.raw_run())
        self.assertEqual(evidence["execution_status_counts"], {"completed": 1, "failed": 1})

    def test_missing_episode_rejected_even_with_completion_marker(self):
        spec = self.raw_run()
        (self.root / "results/toy/s2/F/failure.json").unlink()
        with self.assertRaisesRegex(v.VerificationError, "missing=1"):
            self.checker.run(spec)

    def test_extra_episode_rejected(self):
        spec = self.raw_run()
        self.write("results/toy/s3/F/episode.json", dict(scene_id="s3"))
        with self.assertRaisesRegex(v.VerificationError, "extra=1"):
            self.checker.run(spec)

    def test_stale_marker_for_different_run_rejected(self):
        spec = self.raw_run()
        path = self.root / "results/toy/COMPLETED.json"
        marker = json.loads(path.read_text())
        marker["methods"] = ["G4"]
        path.write_text(json.dumps(marker))
        with self.assertRaisesRegex(v.VerificationError, "marker/config differ"):
            self.checker.run(spec)

    def test_nested_figure_array_reference_must_exist(self):
        path = self.write("figures/inputs/case_views.pt", {"fixture": True})
        record = {"case": {"view_arrays": {"path": "figures/inputs/case_views.pt", "sha256": v.sha256(path)}}}
        self.checker.embedded_hashes(record)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.checker.embedded_hashes(record)

    def test_duplicate_evaluation_rows_rejected(self):
        rows = [dict(scene_id="s1", method="F")] * 2
        with self.assertRaisesRegex(v.VerificationError, "duplicates=1"):
            self.checker.pairs(rows, {("s1", "F")}, "evaluation")

    def test_remote_paths_resolve_to_preserved_local_copy_and_hash_mismatch_fails(self):
        path = self.write("results/toy.json", {"complete": True})
        remote = "/workspace/reading-sonar-signals/results/toy.json"
        self.checker.hash(remote, v.sha256(path))
        with self.assertRaisesRegex(v.VerificationError, "SHA256 mismatch"):
            self.checker.hash(remote, "0" * 64)
        with self.assertRaisesRegex(v.VerificationError, "outside known project"):
            self.checker.path("/tmp/not-this-project.json")
        with self.assertRaisesRegex(v.VerificationError, "escapes project"):
            self.checker.path("../escape.json")

    def test_tex_comments_and_macro_definition_are_not_pending_calls(self):
        source = r"""\newcommand{\pending}[1]{\textbf{Pending: #1}}
\finalresultsfalse
\finalresultstrue % \finalresultsfalse
\begin{document}
% \pending{This is only a comment}
Results retain 20\% failures, including prose about pending historical work.
\end{document}"""
        self.assertIn(r"20\%", v.manuscript_text_checks(source))

    def test_actual_pending_and_disabled_final_flag_fail(self):
        with self.assertRaisesRegex(v.VerificationError, "actual"):
            v.manuscript_text_checks(r"\finalresultstrue\begin{document}\pending{add results}")
        with self.assertRaisesRegex(v.VerificationError, "enable"):
            v.manuscript_text_checks(r"\finalresultstrue\finalresultsfalse\begin{document}done")

    def test_failed_or_gpu_only_report_cannot_make_final_snapshot(self):
        for value in (dict(status="unfinished_or_inconsistent", scope="all_final_deliverables"),
                      dict(status="passed", scope="gpu_deliverables")):
            report = self.write("reports/check.json", value)
            with self.assertRaisesRegex(v.VerificationError, "successful full verification"):
                v.snapshot_sources(self.root, report)
            self.assertFalse((self.root / "provenance/final_source").exists())

    def test_snapshot_is_complete_hashed_and_cannot_overwrite(self):
        for directory in ("src", "scripts", "tests", "configs"):
            self.write(directory + "/fixture.json", {"source": directory})
        for filename in ("README.md", "underwater_acoustic_agent_research_plan.md", "manuscript/main.tex", "manuscript/main.pdf"):
            path = self.root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture content; no actual research output")
        report = self.write("reports/check.json", dict(status="passed", scope="all_final_deliverables", verified_file_sha256={}))
        result = v.snapshot_sources(self.root, report)
        self.assertEqual(result["files"], 8)
        final = self.root / "provenance/final_source"
        inventory = json.loads((final / "manifest.json").read_text())
        for row in inventory["files"]:
            self.assertEqual(v.sha256(final / row["path"]), row["sha256"])
        with self.assertRaisesRegex(v.VerificationError, "overwrite"):
            v.snapshot_sources(self.root, report)


if __name__ == "__main__":
    unittest.main()
