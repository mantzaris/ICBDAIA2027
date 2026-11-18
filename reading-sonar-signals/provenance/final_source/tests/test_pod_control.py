"""Local fault tests. Every subprocess creation is mocked; no network actions."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/pod_control.py"
SPEC = importlib.util.spec_from_file_location("pod_control", SCRIPT)
pod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pod)


def line(status, checked=1, pod_id=pod.POD_ID, **extra):
    return json.dumps(dict(id=pod_id, status=status, checked_at_unix=checked, **extra)) + "\n"


class PodControlTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.previous = Path.cwd()
        os.chdir(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(os.chdir, self.previous)
        artifact = Path("retained.json")
        artifact.write_text("preserved research fixture")
        self.receipt = Path("receipt.json")
        self.receipt.write_text(json.dumps(dict(pod_id=pod.POD_ID, verified=True, files=[
            dict(path=str(artifact), sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())])))
        self.popen = mock.patch.object(pod.subprocess, "Popen")
        self.launch = self.popen.start()
        self.addCleanup(self.popen.stop)
        self.process = self.launch.return_value
        self.process.returncode = 0
        self.output = io.StringIO()

    def invoke(self):
        with contextlib.redirect_stdout(self.output):
            pod.main(["stop", "--receipt", str(self.receipt)])

    def record(self):
        return json.loads(Path("reports/pod_shutdown.json").read_text())

    def test_normal_exited_retains_confirmation_and_attempt(self):
        self.process.communicate.return_value = (line("RUNNING") + line("EXITED", 2), "")
        self.invoke()
        record = self.record()
        self.assertTrue(record["confirmed_stopped"])
        self.assertFalse(record["transport"]["timed_out"])
        self.assertEqual(json.loads(Path(record["attempt_record"]).read_text()), record)
        self.assertEqual(record["local_backup_receipt_sha256"], hashlib.sha256(self.receipt.read_bytes()).hexdigest())
        self.process.kill.assert_not_called()

    def test_exited_then_timeout_keeps_bytes_evidence_and_kills_client(self):
        output = (line("RUNNING") + line("EXITED", 2)).encode()
        self.process.returncode = -9
        self.process.communicate.side_effect = [subprocess.TimeoutExpired("mock-ssh", 60, output=output),
                                               (output.decode(), "ignored transport diagnostic")]
        self.invoke()
        record = self.record()
        self.assertTrue(record["confirmed_stopped"])
        self.assertTrue(record["transport"]["timed_out"])
        self.assertTrue(record["transport"]["local_client_reaped"])
        self.assertEqual([r["status"] for r in record["history"]], ["RUNNING", "EXITED"])
        self.process.kill.assert_called_once_with()
        self.assertEqual(self.process.communicate.call_args_list[1], mock.call(timeout=5))

    def test_exited_with_ssh_disconnect_is_explicit_confirmation(self):
        self.process.returncode = 255
        self.process.communicate.return_value = (line("EXITED"), "Connection closed")
        self.invoke()
        self.assertTrue(self.record()["confirmed_stopped"])

    def test_disconnection_without_confirmation_is_not_success(self):
        self.process.returncode = 255
        self.process.communicate.return_value = (line("RUNNING"), "Connection closed")
        with self.assertRaisesRegex(RuntimeError, "not yet confirmed EXITED"):
            self.invoke()
        record = self.record()
        self.assertFalse(record["confirmed_stopped"])
        self.assertTrue(Path(record["attempt_record"]).is_file())

    def test_timeout_without_any_complete_correct_pod_record_is_saved(self):
        output = line("EXITED", pod_id="different-pod") + '{"id":"' + pod.POD_ID + '","status":"EXI'
        self.process.returncode = -9
        self.process.communicate.side_effect = [subprocess.TimeoutExpired("mock-ssh", 60, output=output.encode()), (output, "")]
        with self.assertRaisesRegex(RuntimeError, "No pod API status"):
            self.invoke()
        record = self.record()
        self.assertFalse(record["confirmed_stopped"])
        self.assertEqual(record["history"], [])
        self.assertTrue(Path(record["attempt_record"]).is_file())

    def test_cleanup_timeout_preserves_exited_and_closes_pipes(self):
        output = line("EXITED").encode()
        self.process.returncode = -9
        self.process.communicate.side_effect = [subprocess.TimeoutExpired("mock-ssh", 60, output=output),
                                               subprocess.TimeoutExpired("mock-ssh", 5, output=output)]
        self.invoke()
        self.assertTrue(self.record()["confirmed_stopped"])
        self.assertTrue(self.record()["transport"]["cleanup_drain_timed_out"])
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close.assert_called_once_with()
        self.process.wait.assert_called_once_with(timeout=5)

    def test_later_running_status_is_not_overridden_by_earlier_exited(self):
        self.process.communicate.return_value = (line("EXITED") + line("RUNNING", 2), "")
        with self.assertRaisesRegex(RuntimeError, "not yet confirmed EXITED"):
            self.invoke()
        self.assertFalse(self.record()["confirmed_stopped"])

    def test_unverified_or_wrong_pod_or_empty_receipt_never_launches(self):
        original = json.loads(self.receipt.read_text())
        for changes in ({"verified": False}, {"verified": "true"}, {"pod_id": "different-pod"}, {"files": []}):
            self.receipt.write_text(json.dumps(dict(original, **changes)))
            with self.assertRaisesRegex(RuntimeError, "verified local artifact receipt"):
                self.invoke()
        self.launch.assert_not_called()
        self.assertFalse(Path("reports/pod_shutdown.json").exists())

    def test_changed_preserved_file_never_launches(self):
        Path("retained.json").write_text("changed after receipt")
        with self.assertRaisesRegex(RuntimeError, "changed after verification"):
            self.invoke()
        self.launch.assert_not_called()

    def test_unexpected_fields_and_transport_text_are_not_logged(self):
        secret = "fixture-secret-never-log"
        output = "Authorization: Bearer " + secret + "\n" + line("EXITED", api_key=secret)
        self.process.communicate.return_value = (output, "stderr " + secret)
        self.invoke()
        persisted = "\n".join(p.read_text() for p in Path("reports").rglob("*.json"))
        self.assertNotIn(secret, persisted + self.output.getvalue())
        self.assertNotIn("api_key", persisted)


if __name__ == "__main__":
    unittest.main()
