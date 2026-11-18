"""One CUDA inference worker batching concurrent LocalVLM-compatible requests.

Timing is batch throughput accounting, not a claim that amortized GPU time equals
serial per-scene latency. Callers must not hold GPU_LOCK while waiting in generate;
the model worker acquires it. Acoustic tools may use the same lock independently.
"""
from __future__ import annotations
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
from threading import RLock, Thread
from typing import Any
import time
import uuid

import torch
from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList
from .vlm import parse_json
from .constrained_decoding import schema_sha256

GPU_LOCK = RLock()


class PerRowTokenLimit(LogitsProcessor):
    """Force a control EOS *after* each row has produced its full allowed prefix.

    A short row with cap160 emits up to160 original tokens, followed by a forced
    EOS at step161 if necessary. The control EOS is excluded from its returned
    prefix/token count. This preserves the cap-length raw output of serial
    max_new_tokens160, rather than replacing token160 with an artificial EOS.
    """
    def __init__(self, prompt_width: int, limits: list[int], eos_token_id: int,
                 device: torch.device | str = "cuda:0"):
        device = torch.device(device)
        if device.type != "cuda":
            raise RuntimeError("Generation token limits must execute on CUDA")
        self.prompt_width = int(prompt_width)
        self.limits = torch.tensor(limits, device=device, dtype=torch.long)
        self.eos_token_id = int(eos_token_id)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if input_ids.device.type != "cuda" or scores.device.type != "cuda":
            raise RuntimeError("Batched generation unexpectedly left CUDA")
        generated = input_ids.shape[1] - self.prompt_width
        due = generated >= self.limits
        scores[due] = -float("inf")
        scores[due, self.eos_token_id] = 0.
        return scores


@dataclass
class _Request:
    prompt: str
    images: list[str | Path]
    max_new_tokens: int
    future: Future
    enqueued_at: float
    request_id: str
    json_schema: dict | None = None


class BatchedLocalVLM:
    """Wrap one already-loaded LocalVLM, sharing its model, processor and audit.

    Multiple episode threads may call generate synchronously; exactly one worker
    touches the model/processor. Failures propagate to every request in that batch
    and are not retried or silently changed into CPU or serial experiments.
    """
    def __init__(self, backend, max_batch_size: int = 4, max_wait_ms: float = 80.,
                 gpu_lock: RLock | None = None):
        if max_batch_size < 1 or max_wait_ms < 0:
            raise ValueError("Invalid batch size/wait interval")
        self.backend = backend
        self.max_batch_size = int(max_batch_size)
        self.max_wait_seconds = float(max_wait_ms) / 1000
        self.gpu_lock = gpu_lock or GPU_LOCK
        self.device_audit = dict(backend.device_audit, execution="single_GPU_worker_batched_greedy",
                                 max_batch_size=self.max_batch_size,
                                 time_accounting="equal_share_batch_GPU_time; serial_latency_not_inferred")
        self._queue: Queue[_Request | None] = Queue()
        self._state_lock = RLock()
        self._closed = False
        self._session_id = uuid.uuid4().hex[:12]
        self._batch_number = 0
        self.batch_ledger: list[dict] = []
        self._worker = Thread(target=self._worker_loop, name="sonar-vlm-batch-worker", daemon=True)
        self._worker.start()

    def generate(self, prompt: str, images: list[str | Path], max_new_tokens: int,
                 json_schema: dict | None = None) -> dict:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        future = Future()
        request = _Request(prompt, list(images), int(max_new_tokens), future,
                           time.perf_counter(), uuid.uuid4().hex, deepcopy(json_schema))
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Batched model is closed")
            self._queue.put(request)
        return future.result()

    def close(self, wait: bool = True):
        with self._state_lock:
            if not self._closed:
                self._closed = True
                self._queue.put(None)
        if wait:
            self._worker.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def _worker_loop(self):
        stop_after_batch = False
        while True:
            first = self._queue.get()
            if first is None:
                return
            requests = [first]
            deadline = time.perf_counter() + self.max_wait_seconds
            while len(requests) < self.max_batch_size:
                wait = deadline - time.perf_counter()
                if wait <= 0:
                    break
                try:
                    item = self._queue.get(timeout=wait)
                except Empty:
                    break
                if item is None:
                    stop_after_batch = True
                    break
                requests.append(item)
            self._batch_number += 1
            batch_id = f"{self._session_id}:{self._batch_number:06d}"
            started = time.perf_counter()
            try:
                results, ledger = self._run_model_batch(requests, batch_id)
                if len(results) != len(requests):
                    raise RuntimeError("Batch result cardinality differs from requests")
                self.batch_ledger.append(ledger)
                for request, result in zip(requests, results):
                    if not request.future.cancelled():
                        request.future.set_result(result)
            except Exception as error:
                self.batch_ledger.append(dict(batch_id=batch_id, status="failed", batch_size=len(requests),
                                               error_type=type(error).__name__, error=str(error),
                                               wall_seconds=time.perf_counter() - started))
                for request in requests:
                    if not request.future.cancelled():
                        request.future.set_exception(error)
            if stop_after_batch:
                return

    @torch.inference_mode()
    def _run_model_batch(self, requests: list[_Request], batch_id: str):
        conversations, opened = [], []
        batch_start = time.perf_counter()
        try:
            for request in requests:
                content = []
                for path in request.images:
                    with Image.open(path) as source:
                        converted = source.convert("RGB")
                    opened.append(converted)
                    content.append(dict(type="image", image=converted))
                content.append(dict(type="text", text=request.prompt))
                conversations.append([dict(role="user", content=content)])
            with self.gpu_lock:
                processor, model = self.backend.processor, self.backend.model
                # A decoder-only batch must align final prompt positions. This
                # processor is accessed solely by this worker after wrapping.
                processor.tokenizer.padding_side = "left"
                inputs = processor.apply_chat_template(
                    conversations, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt", padding=True,
                ).to("cuda:0")
                prompt_width = inputs["input_ids"].shape[1]
                if inputs["input_ids"].shape[0] != len(requests):
                    raise RuntimeError("Processor did not preserve conversation batch size")
                lengths = inputs["attention_mask"].sum(dim=1)
                eos = model.generation_config.eos_token_id
                if eos is None:
                    eos = processor.tokenizer.eos_token_id
                eos_ids = [int(eos)] if isinstance(eos, int) else [int(v) for v in eos]
                if not eos_ids:
                    raise RuntimeError("No explicit EOS token is configured")
                caps = [r.max_new_tokens for r in requests]
                capper = PerRowTokenLimit(prompt_width, caps, eos_ids[0])
                schemas = [r.json_schema for r in requests]
                generation_options = {}
                if any(schema is not None for schema in schemas):
                    generation_options['prefix_allowed_tokens_fn'] = self.backend.grammar_factory().for_batch(
                        schemas,prompt_width,caps,eos_ids)
                torch.cuda.synchronize()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.reset_peak_memory_stats()
                begin.record()
                output = model.generate(**inputs, max_new_tokens=max(caps), do_sample=False,
                                        use_cache=True, logits_processor=LogitsProcessorList([capper]),
                                        **generation_options)
                end.record(); torch.cuda.synchronize()
                gpu_seconds = begin.elapsed_time(end) / 1000
                peak_bytes = torch.cuda.max_memory_allocated()
                tails = output[:, prompt_width:]
                eos_tensor = torch.tensor(eos_ids, device=tails.device)
                stop_mask = (tails[:, :, None] == eos_tensor[None, None, :]).any(dim=-1)
                positions = torch.arange(tails.shape[1], device=tails.device)[None].expand_as(tails)
                first_eos = torch.where(stop_mask, positions, tails.shape[1]).min(dim=1).values
                caps_tensor = torch.tensor(caps, device=tails.device)
                counts = torch.minimum(torch.minimum(first_eos + 1, caps_tensor),
                                       torch.full_like(caps_tensor, tails.shape[1]))
                input_lengths = lengths.tolist()
                count_list, eos_positions = counts.tolist(), first_eos.tolist()
                prefixes = [tails[i, :n] for i, n in enumerate(count_list)]
                texts = processor.batch_decode(prefixes, skip_special_tokens=True)
                grids = inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else []
            batch_wall = time.perf_counter() - batch_start
            shared = dict(batch_id=batch_id, batch_size=len(requests), batch_gpu_seconds=gpu_seconds,
                          batch_wall_seconds=batch_wall, batch_input_tokens=sum(input_lengths),
                          batch_output_tokens=sum(count_list), padded_prompt_width=prompt_width,
                          batch_padded_input_tokens=prompt_width * len(requests),
                          gpu_time_accounting="equal_share_of_measured_batch_event_span",
                          latency_interpretation="batched_request_sojourn; not serial_latency")
            results, grid_start = [], 0
            for index, (request, text) in enumerate(zip(requests, texts)):
                image_count = len(request.images)
                result = dict(raw_text=text, input_tokens=input_lengths[index], output_tokens=count_list[index],
                              gpu_seconds=gpu_seconds / len(requests),
                              wall_seconds=time.perf_counter() - request.enqueued_at,
                              queue_seconds=batch_start - request.enqueued_at,
                              peak_allocated_bytes=peak_bytes,
                              image_grid_thw=grids[grid_start:grid_start + image_count],
                              decode="batched_greedy", max_new_tokens=request.max_new_tokens,
                              at_generation_limit=count_list[index] == request.max_new_tokens,
                              stop_reason="natural_eos" if eos_positions[index] < request.max_new_tokens else "max_new_tokens",
                              control_eos_excluded=(eos_positions[index] == request.max_new_tokens
                                                    and eos_positions[index] < tails.shape[1]),
                              request_id=request.request_id, **shared)
                result['json_schema_sha256'] = schema_sha256(request.json_schema) if request.json_schema is not None else None
                result['structured_decoding'] = self.backend.grammar_factory().provenance['protocol'] if request.json_schema is not None else None
                grid_start += image_count
                try:
                    result["parsed"] = parse_json(text)
                    result["parse_error"] = None
                except ValueError as error:
                    result["parsed"] = None
                    result["parse_error"] = str(error)
                results.append(result)
            ledger = dict(status="complete", **shared, max_new_tokens=caps,
                          json_schema_sha256=[r['json_schema_sha256'] for r in results],
                          request_ids=[r.request_id for r in requests],
                          peak_allocated_bytes=peak_bytes,
                          token_count_convention="natural_EOS_included; padding_and_forced_control_EOS_excluded")
            return results, ledger
        finally:
            for image in opened:
                image.close()


def model_smoke(model_path: str, output_dir: str):
    """Four-request compatibility fixture, not acoustic performance evidence."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from PIL import ImageDraw
    import json
    from .vlm import LocalVLM
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (640, 320), "white")
    ImageDraw.Draw(canvas).text((30, 130), "BATCH COMPATIBILITY FIXTURE. NO ACOUSTIC EVIDENCE.", fill="black")
    image_path = output / "fixture.png"; canvas.save(image_path)
    model = LocalVLM(model_path)
    fixtures = [
        ("Return exactly this JSON object and no other text: {\"action_index\":0,\"reason\":\"inspect tone\"}", [image_path], 160),
        ("Return exactly this JSON object and no other text: {\"components\":[],\"claims\":[],\"abstained\":true}", [image_path], 2048),
        ("Return one JSON object with key numbers and a list of every integer from 1 through 200. No prose.", [image_path], 8),
        ("Return exactly this JSON object and no other text: {\"image_count\":2}", [image_path, image_path], 80),
    ]
    serial = [model.generate(*fixture) for fixture in fixtures[:2]]
    barrier = Barrier(4)
    with BatchedLocalVLM(model, max_batch_size=4, max_wait_ms=100) as batched:
        def request(fixture):
            barrier.wait()
            return batched.generate(*fixture)
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(request, fixtures))
        ledger = list(batched.batch_ledger)
    checks = dict(single_batch=len({r["batch_id"] for r in results}) == 1,
                  caps_respected=all(r["output_tokens"] <= fixture[2] for r, fixture in zip(results, fixtures)),
                  short_prefix_exact_cap=results[2]["output_tokens"] == 8,
                  truncated_outer_json_not_salvaged=results[2]["parsed"] is None,
                  serial_batched_json_agreement=[serial[i]["parsed"] == results[i]["parsed"] for i in range(2)],
                  expected_structures_returned=(results[0]["parsed"] == {"action_index": 0, "reason": "inspect tone"}
                                                and results[1]["parsed"] == {"components": [], "claims": [], "abstained": True}
                                                and results[3]["parsed"] == {"image_count": 2}),
                  different_image_counts_preserved=[len(r["image_grid_thw"]) for r in results] == [1, 1, 1, 2])
    passed = (checks["single_batch"] and checks["caps_respected"] and checks["short_prefix_exact_cap"]
              and checks["truncated_outer_json_not_salvaged"] and all(checks["serial_batched_json_agreement"])
              and checks["expected_structures_returned"]
              and checks["different_image_counts_preserved"])
    record = dict(passed=passed, checks=checks, serial=serial, batched=results, batch_ledger=ledger,
                  interpretation="Compatibility fixture only; greedy equality is checked on these prompts, not guaranteed for all floating-point batches")
    (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(passed=passed, checks=checks)), flush=True)
    if not passed:
        raise RuntimeError("Batch compatibility fixture failed; do not enable main batched evaluation")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/qwen3-vl-8b")
    parser.add_argument("--output", default="reports/batch_smoke")
    options = parser.parse_args()
    model_smoke(options.model, options.output)
