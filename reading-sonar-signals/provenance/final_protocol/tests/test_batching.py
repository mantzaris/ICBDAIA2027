"""Administrative queue tests and CUDA token-cap invariants, no CPU model runs."""
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, get_ident
import unittest
import torch
from sonar.batching import BatchedLocalVLM, PerRowTokenLimit


class FakeBackend:
    device_audit = {"test_double": "queue_scheduling_only_no_model"}


class QueueOnlyBatcher(BatchedLocalVLM):
    def _run_model_batch(self, requests, batch_id):
        if any(request.prompt == "RAISE" for request in requests):
            raise ValueError("Deliberate batch failure")
        results = [dict(payload=request.prompt, cap=request.max_new_tokens, worker=get_ident(),
                        batch_id=batch_id) for request in requests]
        return results, dict(batch_id=batch_id, size=len(requests), status="complete")


class BatchQueueTests(unittest.TestCase):
    def test_concurrent_requests_preserve_identity_and_one_model_worker(self):
        barrier = Barrier(4)
        with QueueOnlyBatcher(FakeBackend(), max_batch_size=4, max_wait_ms=100) as model:
            def request(index):
                barrier.wait()
                return model.generate(str(index), [], 160 if index % 2 else 2048)
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(request, range(4)))
            self.assertEqual([r["payload"] for r in results], ["0", "1", "2", "3"])
            self.assertEqual([r["cap"] for r in results], [2048, 160, 2048, 160])
            self.assertEqual(len({r["worker"] for r in results}), 1)
            self.assertEqual(len({r["batch_id"] for r in results}), 1)
        with self.assertRaises(RuntimeError):
            model.generate("closed", [], 1)

    def test_batch_failures_reach_every_waiter(self):
        barrier = Barrier(2)
        with QueueOnlyBatcher(FakeBackend(), max_batch_size=2, max_wait_ms=100) as model:
            def request(prompt):
                barrier.wait()
                return model.generate(prompt, [], 10)
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(request, prompt) for prompt in ("RAISE", "also fails")]
                for future in futures:
                    with self.assertRaisesRegex(ValueError, "Deliberate"):
                        future.result(timeout=3)
            self.assertEqual(model.batch_ledger[0]["status"], "failed")

    def test_single_request_batches_recover_after_failure(self):
        with QueueOnlyBatcher(FakeBackend(), max_batch_size=1, max_wait_ms=0) as model:
            with self.assertRaises(ValueError):
                model.generate("RAISE", [], 10)
            recovered = model.generate("next request", [], 10)
            self.assertEqual(recovered["payload"], "next request")
            self.assertEqual([row["status"] for row in model.batch_ledger], ["failed", "complete"])


class TokenCapCUDATests(unittest.TestCase):
    def test_caps_keep_last_original_token_before_forcing_eos(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA token-cap test is executed on the pod; no CPU fallback")
        processor = PerRowTokenLimit(5, [2, 4], eos_token_id=3)
        scores = torch.arange(10., device="cuda").repeat(2, 1)
        # At generated length1, row0 still gets its original second token.
        before = processor(torch.zeros((2, 6), device="cuda", dtype=torch.long), scores.clone())
        self.assertTrue(torch.equal(before, scores))
        # Once two tokens exist, only row0 receives a control EOS.
        after = processor(torch.zeros((2, 7), device="cuda", dtype=torch.long), scores.clone())
        self.assertEqual(after[0].argmax().item(), 3)
        self.assertTrue(torch.isneginf(after[0, torch.arange(10, device="cuda") != 3]).all())
        self.assertTrue(torch.equal(after[1], scores[1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
