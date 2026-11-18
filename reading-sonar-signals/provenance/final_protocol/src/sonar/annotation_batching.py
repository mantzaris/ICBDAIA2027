"""Optional execution adapter for AnnotationVLM's unchanged few-shot requests.

This is separate from the reference protocol. Deploying it requires a recorded
execution amendment and calibration-only serial/batch equivalence evidence.
GPU seconds are equal shares of one measured batch, never serial latency claims.
"""
from __future__ import annotations

from concurrent.futures import Future
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from queue import Empty, Queue
from threading import RLock, Thread
import time
import uuid

import torch
from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList

from .json_protocol import parse_json


ANNOTATION_GPU_LOCK = RLock()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PerRowTokenLimit(LogitsProcessor):
    """Force EOS only after a row has kept its complete serial cap prefix.

    The extra control EOS occurs at cap+1 and is excluded from that row's result.
    All score changes execute on CUDA; there is no model inference CPU fallback.
    """
    def __init__(self, prompt_width, limits, eos_token_id, device="cuda:0"):
        device = torch.device(device)
        if device.type != "cuda":
            raise RuntimeError("Annotation token limits must execute on CUDA")
        self.prompt_width = int(prompt_width)
        self.limits = torch.tensor(limits, device=device, dtype=torch.long)
        self.eos_token_id = int(eos_token_id)

    def __call__(self, input_ids, scores):
        if input_ids.device.type != "cuda" or scores.device.type != "cuda":
            raise RuntimeError("Annotation token-cap tensors left CUDA")
        generated = input_ids.shape[1] - self.prompt_width
        due = generated >= self.limits
        scores[due] = -float("inf")
        scores[due, self.eos_token_id] = 0.
        return scores


@dataclass
class AnnotationRequest:
    prompt: str
    images: list[str | Path]
    max_new_tokens: int
    examples: list[dict]
    future: Future
    enqueued_at: float
    request_id: str


def fewshot_conversation(prompt, images, examples, load_image):
    """Use exactly AnnotationVLM.generate's message and image order."""
    content = [{"type": "image", "image": load_image(path)} for path in images]
    content.append({"type": "text", "text": prompt})
    messages = []
    for example in examples or []:
        messages.extend([
            {"role": "user", "content": [
                {"type": "image", "image": load_image(example["image"])},
                {"type": "text", "text": example["prompt"]}]},
            {"role": "assistant", "content": [
                {"type": "text", "text": json.dumps(example["answer"], separators=(",", ":"))}]}])
    messages.append({"role": "user", "content": content})
    return messages


def completion_info(tokens, eos_token_ids, limit):
    """Bookkeeping oracle: natural EOS counts, post-EOS padding does not.

    A forced EOS at index ``limit`` is outside the original serial prefix. A
    distinct pad token before EOS is not a stop signal in native generate either.
    """
    if type(limit) is not int or limit < 1 or not eos_token_ids:
        raise ValueError("Require a positive token cap and explicit EOS IDs")
    first = next((i for i, token in enumerate(tokens) if token in eos_token_ids), len(tokens))
    count = min(first + 1, limit, len(tokens))
    natural = first < min(limit, len(tokens))
    return {
        "output_tokens": count,
        "at_generation_limit": count == limit,
        "stop_reason": "natural_eos" if natural else ("max_new_tokens" if count == limit else "generation_stopped"),
        "control_eos_excluded": first == limit and first < len(tokens),
    }


class AnnotationBatchAdapter:
    """One model worker; ``generate`` accepts AnnotationVLM's existing arguments.

    ``submit`` permits multiple pending requests. ``generate_batch`` atomically
    queues request dictionaries and returns results in the submitted order.
    The backend must be exclusively owned by this adapter until it is closed.
    """

    def __init__(self, backend, max_batch_size=4, max_wait_ms=80., gpu_lock=None):
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if not math.isfinite(max_wait_ms) or max_wait_ms < 0:
            raise ValueError("max_wait_ms must be finite and nonnegative")
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = float(max_wait_ms) / 1000
        self.gpu_lock = gpu_lock or ANNOTATION_GPU_LOCK
        self.device_audit = dict(backend.device_audit, execution="single_annotation_GPU_worker_batched_greedy",
                                 max_batch_size=max_batch_size,
                                 time_accounting="equal_share_batch_GPU_time; serial_latency_not_inferred")
        self._queue = Queue()
        self._state_lock = RLock()
        self._closed = False
        self._session_id = uuid.uuid4().hex[:12]
        self._batch_number = 0
        self.batch_ledger = []
        self._worker = Thread(target=self._worker_loop, name="sonar-annotation-batch-worker", daemon=True)
        self._worker.start()

    def close(self, wait=True):
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
                    raise RuntimeError("Annotation batch result count differs from requests")
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

    def _request(self, prompt, images, max_new_tokens=None, examples=None):
        maximum = self.backend.config.get("max_new_tokens", 1536) if max_new_tokens is None else max_new_tokens
        if type(maximum) is not int or maximum < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if not isinstance(prompt, str):
            raise ValueError("Annotation prompt must be a string")
        return AnnotationRequest(prompt, list(images), maximum, deepcopy(examples or []),
                                 Future(), time.perf_counter(), uuid.uuid4().hex)

    def submit(self, prompt, images, max_new_tokens=None, examples=None):
        request = self._request(prompt, images, max_new_tokens, examples)
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Batched model is closed")
            self._queue.put(request)
        return request.future

    def generate(self, prompt, images, max_new_tokens=None, examples=None):
        return self.submit(prompt, images, max_new_tokens, examples).result()

    def generate_batch(self, requests: list[dict]) -> list[dict]:
        prepared = [self._request(**request) for request in requests]
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Batched model is closed")
            for request in prepared:
                self._queue.put(request)
        return [request.future.result() for request in prepared]

    @torch.inference_mode()
    def _run_model_batch(self, requests, batch_id):
        opened = []
        batch_start = time.perf_counter()

        def load_image(path):
            with Image.open(path) as source:
                converted = source.convert("RGB")
            opened.append(converted)
            return converted

        try:
            conversations = [fewshot_conversation(r.prompt, r.images, r.examples, load_image) for r in requests]
            with self.gpu_lock:
                processor, model = self.backend.processor, self.backend.model
                for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
                    if tensor.device.type != "cuda":
                        raise RuntimeError(f"Annotation model tensor left CUDA: {name}")
                previous_padding = processor.tokenizer.padding_side
                try:
                    processor.tokenizer.padding_side = "left"
                    # Batch1 follows the exact native processor call, including
                    # its unbatched conversation shape and absent padding flag.
                    arguments = dict(tokenize=True, add_generation_prompt=True,
                                     return_dict=True, return_tensors="pt")
                    if len(requests) > 1:
                        arguments["padding"] = True
                    inputs = processor.apply_chat_template(
                        conversations[0] if len(requests) == 1 else conversations,
                        **arguments).to("cuda:0")
                finally:
                    processor.tokenizer.padding_side = previous_padding
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        if value.device.type != "cuda":
                            raise RuntimeError(f"Annotation input tensor left CUDA: {key}")
                        if value.is_floating_point() and key.startswith("pixel_values"):
                            inputs[key] = value.to(torch.bfloat16)
                if inputs["input_ids"].shape[0] != len(requests):
                    raise RuntimeError("Processor did not preserve annotation conversation count")
                prompt_width = inputs["input_ids"].shape[1]
                lengths = inputs["attention_mask"].sum(dim=1)
                eos = model.generation_config.eos_token_id
                if eos is None:
                    eos = processor.tokenizer.eos_token_id
                eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
                if not eos_ids:
                    raise RuntimeError("Annotation generation requires explicit EOS IDs")
                caps = [r.max_new_tokens for r in requests]
                options = dict(max_new_tokens=max(caps), do_sample=False, use_cache=True)
                if len(set(caps)) > 1:
                    options["logits_processor"] = LogitsProcessorList([
                        PerRowTokenLimit(prompt_width, caps, eos_ids[0])])
                torch.cuda.synchronize()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.reset_peak_memory_stats()
                begin.record()
                output = model.generate(**inputs, **options)
                end.record()
                torch.cuda.synchronize()
                gpu_seconds = begin.elapsed_time(end) / 1000
                peak_bytes = torch.cuda.max_memory_allocated()
                if output.device.type != "cuda":
                    raise RuntimeError("Annotation generation output left CUDA")
                if output.shape[0] != len(requests):
                    raise RuntimeError("Model did not preserve annotation request count")
                tails = output[:, prompt_width:]
                token_rows = tails.tolist()
                boundaries = [completion_info(row, eos_ids, cap) for row, cap in zip(token_rows, caps)]
                prefixes = [tails[i, :info["output_tokens"]] for i, info in enumerate(boundaries)]
                texts = processor.batch_decode(prefixes, skip_special_tokens=True)
                if len(texts) != len(requests):
                    raise RuntimeError("Decoder did not preserve annotation request count")
                input_lengths = lengths.tolist()
                tensor_shapes = {key: list(value.shape) for key, value in inputs.items() if isinstance(value, torch.Tensor)}
            batch_wall = time.perf_counter() - batch_start
            shared = dict(batch_id=batch_id, batch_size=len(requests), batch_gpu_seconds=gpu_seconds,
                          batch_wall_seconds=batch_wall, batch_input_tokens=sum(input_lengths),
                          batch_output_tokens=sum(x["output_tokens"] for x in boundaries),
                          padded_prompt_width=prompt_width, batch_padded_input_tokens=prompt_width * len(requests),
                          gpu_time_accounting="equal_share_of_measured_batch_event_span",
                          latency_interpretation="batched_request_sojourn; not serial_latency",
                          token_count_convention="natural_EOS_included; post_EOS_padding_and_control_EOS_excluded")
            results = []
            for index, (request, raw, boundary) in enumerate(zip(requests, texts, boundaries)):
                result = dict(raw_text=raw, input_tokens=input_lengths[index],
                              gpu_seconds=gpu_seconds / len(requests),
                              wall_seconds=time.perf_counter() - request.enqueued_at,
                              queue_seconds=batch_start - request.enqueued_at,
                              peak_allocated_bytes=peak_bytes, max_new_tokens=request.max_new_tokens,
                              decode="greedy", model_id=self.backend.spec["model_id"],
                              model_revision=self.backend.spec["revision"],
                              device_audit=dict(self.device_audit, model_and_inputs_cuda_verified=True,
                                                pixel_values_dtype="bfloat16"),
                              input_tensor_shapes=tensor_shapes, request_id=request.request_id,
                              **boundary, **shared)
                result["training_examples"] = [dict(image=x["image"], image_sha256=file_hash(x["image"]),
                    answer=x["answer"], prompt=x["prompt"], provenance=x.get("provenance")) for x in request.examples]
                try:
                    result.update(parsed=parse_json(raw), parse_error=None)
                except ValueError as error:
                    result.update(parsed=None, parse_error=str(error))
                results.append(result)
            ledger = dict(status="complete", **shared, max_new_tokens=caps,
                          request_ids=[r.request_id for r in requests], peak_allocated_bytes=peak_bytes)
            return results, ledger
        finally:
            for image in opened:
                image.close()


BatchedAnnotationVLM = AnnotationBatchAdapter
