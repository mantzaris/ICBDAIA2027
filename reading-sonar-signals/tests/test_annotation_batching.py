"""Administrative and token-bookkeeping tests; no model or CUDA execution."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
from threading import Barrier, get_ident

import pytest

from sonar.annotation_batching import BatchedAnnotationVLM, completion_info, fewshot_conversation
from sonar.reference_protocol import object_hash


def test_fewshot_turns_preserve_native_prompt_and_image_order():
    examples = [dict(image="demo1", prompt="example one\n", answer={"events": [[1, 2, 3, 3, "tone"]]}),
                dict(image="demo2", prompt="example two", answer={"events": []})]
    messages = fewshot_conversation("target\nfull raw grid", ["raw1", "raw2"], examples, lambda x: "IMAGE:" + x)
    assert [row["role"] for row in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert messages[0]["content"] == [{"type": "image", "image": "IMAGE:demo1"}, {"type": "text", "text": "example one\n"}]
    assert messages[1]["content"][0]["text"] == '{"events":[[1,2,3,3,"tone"]]}'
    assert messages[-1]["content"] == [{"type": "image", "image": "IMAGE:raw1"},
        {"type": "image", "image": "IMAGE:raw2"}, {"type": "text", "text": "target\nfull raw grid"}]


@pytest.mark.parametrize("tokens,eos,cap,count,reason,control", [
    ([10, 2, 0, 0], [2], 8, 2, "natural_eos", False),
    ([10, 3, 0, 0], [2, 3], 8, 2, "natural_eos", False),
    ([10, 11, 2, 0], [2], 2, 2, "max_new_tokens", True),
    ([10, 11], [2], 2, 2, "max_new_tokens", False),
    ([10, 0, 11, 2], [2], 8, 4, "natural_eos", False),
    ([10, 2], [2], 2, 2, "natural_eos", False),
    ([2, 2, 2], [2], 8, 1, "natural_eos", False),
])
def test_row_eos_padding_and_caps(tokens, eos, cap, count, reason, control):
    info = completion_info(tokens, eos, cap)
    assert info == dict(output_tokens=count, at_generation_limit=count == cap,
                        stop_reason=reason, control_eos_excluded=control)


class Backend:
    config = {"max_new_tokens": 768}
    device_audit = {"test_double": "administrative_queue_only"}


class QueueOnly(BatchedAnnotationVLM):
    def _run_model_batch(self, requests, batch_id):
        if any(r.prompt == "FAIL" for r in requests):
            raise ValueError("deliberate failure")
        return [dict(prompt=r.prompt, examples=r.examples, cap=r.max_new_tokens,
                     batch_id=batch_id, worker=get_ident()) for r in requests], dict(batch_id=batch_id, status="complete")


def test_batch_and_thread_api_preserve_examples_with_one_worker():
    examples = [dict(image="demo", prompt="example", answer={"events": []})]
    with QueueOnly(Backend(), max_batch_size=4, max_wait_ms=100) as adapter:
        rows = adapter.generate_batch([dict(prompt=str(i), images=[], examples=examples) for i in range(4)])
        assert [r["prompt"] for r in rows] == ["0", "1", "2", "3"]
        assert len({r["batch_id"] for r in rows}) == 1
        barrier = Barrier(4)
        def call(i):
            barrier.wait()
            return adapter.generate(str(i), [], 10 + i, examples)
        with ThreadPoolExecutor(max_workers=4) as executor:
            threaded = list(executor.map(call, range(4)))
        assert [r["cap"] for r in threaded] == [10, 11, 12, 13]
        assert all(r["examples"] == examples for r in rows + threaded)
        assert len({r["worker"] for r in rows + threaded}) == 1
    with pytest.raises(RuntimeError, match="closed"):
        adapter.generate("no", [])


def test_failures_propagate_and_examples_are_snapshotted():
    with QueueOnly(Backend(), max_batch_size=2, max_wait_ms=100) as adapter:
        first = adapter.submit("FAIL", [])
        second = adapter.submit("same batch", [])
        for future in [first, second]:
            with pytest.raises(ValueError, match="deliberate failure"):
                future.result(timeout=3)
        examples = [dict(image="demo", prompt="before", answer={"events": []})]
        future = adapter.submit("next", [], examples=examples)
        examples[0]["prompt"] = "after"
        assert future.result(timeout=3)["examples"][0]["prompt"] == "before"
    assert [r["status"] for r in adapter.batch_ledger] == ["failed", "complete"]


def test_invalid_caps_are_rejected_before_queueing():
    with QueueOnly(Backend(), max_batch_size=1, max_wait_ms=0) as adapter:
        for cap in [0, -1, True, 1.5]:
            with pytest.raises(ValueError, match="positive integer"):
                adapter.generate("test", [], cap)


def test_smoke_uses_only_completed_calibration_and_verified_images(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/smoke_annotation_batching.py"
    spec = importlib.util.spec_from_file_location("annotation_smoke_test", script)
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    policy = {"calibration_seeds": [101], "held_out_audit_seeds": [201]}
    calibration = dict(scene_id="calibration", annotation_control=True, control_role="annotation_calibration",
                       seed=101, split="pilot", stratum="synthetic")
    audit = dict(calibration, scene_id="audit", seed=201, control_role="reference_error_audit")
    manifest = tmp_path / "controls.jsonl"
    manifest.write_text(json.dumps(audit) + "\n" + json.dumps(calibration) + "\n")
    image = tmp_path / "image.bin"
    image.write_bytes(b"unchanged saved fixture bytes")
    image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
    example = dict(image=str(image), image_sha256=image_hash, prompt="saved demo", answer={"events": []},
                   provenance=dict(role="exact_separate_synthetic_training_demonstration", target_truth_used=False))
    passes = [dict(view=view, image=str(image), image_sha256=image_hash, prompt="saved prompt " + view,
                   generation=dict(training_examples=[example], max_new_tokens=768, raw_text='{"events":[]}'))
              for view in ["lofar_1s", "lofar_4s", "demon_500_2000", "demon_2000_5000"]]
    summary = dict(calibration_only=True, scenes=1, control_records_sha256=object_hash([calibration]),
                   control_seeds=[101], protocol_identity={"fixture": "saved identity"})
    summary_path = tmp_path / "internvl4b_summary.json"
    summary_path.write_text(json.dumps(summary))
    directory = tmp_path / "internvl4b"
    directory.mkdir()
    record = dict(scene_id="calibration", input_record_sha256=object_hash(calibration),
                  protocol_identity=summary["protocol_identity"],
                  generation=dict(passes=passes, model_id="saved model", model_revision="saved revision"))
    fixture_path = directory / "calibration.json"
    fixture_path.write_text(json.dumps(record))
    requests, _, provenance = smoke.load_fixture(tmp_path, "4b", manifest, policy)
    assert requests[0]["prompt"] == "saved prompt lofar_1s"
    assert provenance["scene_id"] == "calibration" and provenance["main_or_audit_data_used"] is False
    fixture_path.write_text(json.dumps(dict(record, scene_id="audit", input_record_sha256=object_hash(audit))))
    with pytest.raises(RuntimeError, match="first declared calibration"):
        smoke.load_fixture(tmp_path, "4b", manifest, policy)
    fixture_path.write_text(json.dumps(record))
    image.write_bytes(b"changed image")
    with pytest.raises(RuntimeError, match="target image hash changed"):
        smoke.load_fixture(tmp_path, "4b", manifest, policy)
    summary_path.write_text(json.dumps(dict(summary, calibration_only=False)))
    with pytest.raises(RuntimeError, match="calibration-only summary"):
        smoke.load_fixture(tmp_path, "4b", manifest, policy)
    with pytest.raises(ValueError, match="different numbers"):
        smoke.comparison([{}], [])
