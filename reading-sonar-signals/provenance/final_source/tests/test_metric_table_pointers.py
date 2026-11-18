"""Administrative provenance fixtures; no scientific package or GPU imports."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_metric_table_pointers.py"
SPEC = importlib.util.spec_from_file_location("audit_metric_table_pointers", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class PointerAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "results/fixture/evaluation"
        self.folder.mkdir(parents=True)

    def fixture(self, value=0.5, metrics=None, scored=1, statuses=None):
        if metrics is None and value is not None:
            metrics = {metric: value for metric in audit.METRICS}
        group = dict(event_metrics=metrics, reference_scored_episodes=scored,
                     reference_statuses=statuses or ["complete"])
        summary = dict(status="complete_evaluation", results={"real": {"all": {"F": group}}})
        cell = dict(stratum="real", domain="all", method="F", metric="precision", value=value,
                    source="summary.json", json_pointer="/results/real/all/F/event_metrics/precision",
                    reference_scored_episodes=scored)
        (self.folder / "summary.json").write_text(json.dumps(summary))
        cells = [dict(cell, metric=metric, json_pointer="/results/real/all/F/event_metrics/" + metric) for metric in audit.METRICS]
        (self.folder / "table_cells.json").write_text(json.dumps(cells))
        (self.folder / "claim_to_evidence.json").write_text(json.dumps([dict(c, claim_id="real_F_" + c["metric"]) for c in cells]))
        return summary, cell

    def run_audit(self):
        return audit.audit(self.root)

    def test_numeric_value_resolves_without_modifying_exports(self):
        self.fixture()
        before = {p: p.read_bytes() for p in self.folder.iterdir()}
        result = self.run_audit()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["export_audits"]), 2)
        self.assertEqual(result["null_pointer_companions"], [])
        self.assertEqual(result["export_audits"][0]["checked_cells"][0]["source_value"], 0.5)
        self.assertEqual(before, {p: p.read_bytes() for p in self.folder.iterdir()})

    def test_explicit_null_parent_gets_additive_resolvable_companion(self):
        summary, _ = self.fixture(value=None, scored=0, statuses=["unavailable"])
        result = self.run_audit()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(result["null_pointer_companions"]), 2)
        for companion in result["null_pointer_companions"]:
            entry = companion["mappings"][0]
            self.assertIsNone(entry["unchanged_export_value"])
            self.assertIsNone(entry["parent_value"])
            self.assertIsNone(audit.resolve_pointer(summary, entry["resolvable_parent_pointer"])["value"])
            self.assertEqual(entry["source_support_value"], 0)
            self.assertEqual(entry["cell_index"], 0)
            self.assertIn("source_summary_sha256", companion)
            self.assertIn("export_sha256", companion)

    def test_nonnull_broken_reference_is_integrity_failure(self):
        self.fixture(value=0.5, metrics={})
        result = self.run_audit()
        self.assertEqual(result["status"], "integrity_failure")
        self.assertIn("missing_field", str(result["failures"]))
        self.assertEqual(result["null_pointer_companions"], [])

    def test_null_leaf_missing_from_object_is_not_an_unavailable_container(self):
        self.fixture(value=None, metrics={}, scored=0)
        result = self.run_audit()
        self.assertEqual(result["status"], "integrity_failure")
        self.assertEqual(result["null_pointer_companions"], [])

    def test_explicit_resolved_null_leaf_is_preserved(self):
        self.fixture(value=None, metrics={metric: None for metric in audit.METRICS}, scored=1)
        result = self.run_audit()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["null_pointer_companions"], [])
        self.assertIsNone(result["export_audits"][0]["checked_cells"][0]["source_value"])

    def test_null_parent_with_positive_support_is_not_silently_explained(self):
        self.fixture(value=None, scored=3)
        result = self.run_audit()
        self.assertEqual(result["status"], "integrity_failure")
        self.assertIn("zero reference-scored support", str(result["failures"]))

    def test_nonnull_mismatch_and_wrong_metric_identity_fail(self):
        summary, cell = self.fixture()
        cell["value"] = 0.6
        with self.assertRaisesRegex(ValueError, "differs from its resolved"):
            audit.cell_check(summary, cell, 0)
        cell["value"] = 0.5
        cell["method"] = "G4"
        with self.assertRaisesRegex(ValueError, "declared stratum"):
            audit.cell_check(summary, cell, 0)

    def test_missing_claim_export_is_reported(self):
        self.fixture()
        (self.folder / "claim_to_evidence.json").unlink()
        result = self.run_audit()
        self.assertEqual(result["status"], "integrity_failure")
        self.assertIn("claim_to_evidence.json", str(result["failures"]))

    def test_pointer_escapes_and_arrays_resolve_without_computation(self):
        document = {"a/b": {"c~d": [None, 7]}}
        self.assertEqual(audit.resolve_pointer(document, "/a~1b/c~0d/1"), dict(resolved=True, value=7))
        self.assertEqual(audit.resolve_pointer(document, "/a~1b/c~0d/01")["reason"], "invalid_array_index")
        with self.assertRaisesRegex(ValueError, "escape"):
            audit.resolve_pointer(document, "/a~2b")

    def test_summary_source_escape_is_rejected(self):
        _, cell = self.fixture()
        cell["source"] = "/tmp/outside-this-fixture.json"
        (self.folder / "table_cells.json").write_text(json.dumps([cell]))
        result = self.run_audit()
        self.assertEqual(result["status"], "integrity_failure")
        self.assertIn("outside project root", str(result["failures"]))

    def test_unknown_metric_cannot_be_certified_unavailable(self):
        summary, cell = self.fixture(value=None, scored=0)
        cell.update(metric="precison", json_pointer="/results/real/all/F/event_metrics/precison")
        with self.assertRaisesRegex(ValueError, "Unknown exported metric"):
            audit.cell_check(summary, cell, 0)

    def test_missing_table_is_discovered_from_completed_summary(self):
        self.fixture()
        (self.folder / "table_cells.json").unlink()
        result = self.run_audit()
        self.assertEqual(result["status"], "integrity_failure")
        self.assertIn("table_cells.json", str(result["failures"]))

    def test_empty_missing_or_duplicate_export_rows_fail_inventory(self):
        self.fixture()
        path = self.folder / "table_cells.json"
        original = json.loads(path.read_text())
        for rows in ([], original[:-1], original + [original[0]]):
            path.write_text(json.dumps(rows))
            result = self.run_audit()
            self.assertEqual(result["status"], "integrity_failure")
            self.assertIn("row inventory mismatch", str(result["failures"]))

    def test_missing_or_wrong_claim_id_is_rejected(self):
        self.fixture()
        path = self.folder / "claim_to_evidence.json"
        rows = json.loads(path.read_text())
        rows[0]["claim_id"] = "unrelated_claim"
        path.write_text(json.dumps(rows))
        result = self.run_audit()
        self.assertEqual(result["status"], "integrity_failure")
        self.assertIn("mismatched claim_id", str(result["failures"]))


if __name__ == "__main__":
    unittest.main()
