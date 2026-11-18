"""Run with PYTHONPATH=src python -m unittest discover -s tests -p test_signal_gpu.py.

These analytic checks are CUDA experiments. Absence of CUDA is a failure, not a
skip or a silent CPU execution of a nominal GPU experiment.
"""
import math
import unittest
import torch
from sonar.signal import (require_cuda, stft_psd, resample, demon, spectral_peaks,
                           band_power, harmonic_fit, propose_components)
from sonar.synthetic import generate_scene, remove_component, real_mixture
from sonar.evidence import EvidenceGraph, AcousticTools, verify_claims
from sonar.metrics import event_metrics, aggregate_event_metrics, grouped_bootstrap, holm


class SignalGPUValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = require_cuda()

    def tone(self, freq=200., sr=8000, seconds=8., amplitude=.3):
        t = torch.arange(round(sr * seconds), device=self.device) / sr
        return amplitude * torch.cos(2 * math.pi * freq * t)

    def test_psd_parseval_and_coordinates(self):
        x = self.tone()
        view = stft_psd(x, 8000, window_s=1., hop_s=.25, offset_s=47.)
        power = view.psd.sum(dim=0) * view.metadata["bin_spacing_hz"]
        self.assertLess((power - .3 ** 2 / 2).abs().max().item(), 1e-5)
        self.assertEqual(spectral_peaks(view)[0]["frequency_hz"], 200.)
        self.assertAlmostEqual(view.times_s[0].item(), 47. + 7999 / 16000, places=5)
        self.assertEqual(view.psd.device.type, "cuda")

    def test_dc_nyquist_scaling(self):
        n = 32000
        for x in (torch.ones(n, device=self.device) * .2,
                  (torch.arange(n, device=self.device) % 2 * 2 - 1).float() * .2):
            view = stft_psd(x, 8000)
            self.assertLess((view.psd.sum(dim=0) - .04).abs().max().item(), 1e-6)

    def test_antialias(self):
        out = resample(self.tone(freq=2800., amplitude=1), 8000., 4000.)
        self.assertLess(out[1000:-1000].square().mean().sqrt().item(), .002)
        low = resample(self.tone(freq=200., amplitude=1), 8000., 4000.)
        self.assertLess(abs(low[1000:-1000].square().mean().item() - .5), .002)

    def test_am_recovery_and_carrier_provenance(self):
        sr = 16000
        t = torch.arange(12 * sr, device=self.device) / sr
        x = (1 + .7 * torch.cos(2 * math.pi * 12 * t)) * torch.cos(2 * math.pi * 1100 * t)
        view = demon(x, sr, (500, 2000), offset_s=80.)
        peak = spectral_peaks(view)[0]
        self.assertAlmostEqual(peak["frequency_hz"], 12., places=1)
        self.assertEqual(peak["frequency_domain"], "modulation")
        self.assertEqual(peak["carrier_band_hz"], [500., 2000.])
        self.assertGreater(view.times_s[0].item(), 82.)

    def test_beating_has_no_propulsion_label(self):
        sr = 16000
        x = self.tone(1000., sr, 12., 1.) + self.tone(1012., sr, 12., 1.)
        view = demon(x, sr, (500, 2000))
        self.assertAlmostEqual(spectral_peaks(view)[0]["frequency_hz"], 12., places=1)
        self.assertEqual(view.metadata["mechanism"], "periodic_envelope_not_source_identity")

    def test_insufficient_band_and_time_fail_explicitly(self):
        x = self.tone(1000., 16000, 5.)
        with self.assertRaises(ValueError):
            demon(x, 16000, (5000, 8000))
        with self.assertRaises(ValueError):
            stft_psd(x[:16000], 16000, window_s=4.)
        with self.assertRaises(RuntimeError):
            stft_psd(torch.ones(8000), 8000)

    def test_silence_produces_no_peaks_or_tracks(self):
        view = stft_psd(torch.zeros(64000, device=self.device), 8000.)
        self.assertEqual(spectral_peaks(view), [])
        self.assertEqual(propose_components(view), [])

    def test_harmonic_compatibility_not_source_identity(self):
        result = harmonic_fit([100., 200., 300.])
        self.assertTrue(result["supported"])
        self.assertIn("not_unique_source_identity", result["interpretation"])
        self.assertFalse(harmonic_fit([100., 200.])["supported"])

    def test_graph_retains_overlapping_morphologies(self):
        graph = EvidenceGraph("opaque")
        base = dict(frequency_domain="acoustic", time_s=[1., 5.], frequency_hz=[100., 105.], bin_spacing_hz=1.)
        graph.add_measurement(dict(tool="test"), [dict(base, type="tone"), dict(base, type="chirp")])
        self.assertEqual(len(graph.components), 2)
        self.assertTrue(any(r["type"] == "overlaps_time" for r in graph.relations))
        graph.add_measurement(dict(tool="test_again"), [dict(base, type="tone")])
        self.assertEqual(len(graph.components), 2)
        self.assertEqual(len(graph.components[0]["evidence_ids"]), 2)
        raw = graph.measurements[1]["observed_components"]
        self.assertEqual(len(raw), 1)
        self.assertEqual(raw[0]["id"], "E0001:O000")
        self.assertEqual(raw[0]["scene_id"], "opaque")
        self.assertEqual(raw[0]["evidence_ids"], ["E0001"])
        self.assertEqual(graph.measurements[0]["observed_components"][0]["evidence_ids"], ["E0000"])

    def test_graph_retains_crossing_ridges_with_same_bounding_box(self):
        graph = EvidenceGraph("opaque")
        base = dict(type="chirp", frequency_domain="acoustic", time_s=[1., 5.], frequency_hz=[100., 140.], bin_spacing_hz=1.)
        first = dict(base, frequency_track=[[1., 100.], [5., 140.]])
        second = dict(base, frequency_track=[[1., 140.], [5., 100.]])
        graph.add_measurement(dict(tool="test"), [first, second])
        self.assertEqual(len(graph.components), 2)

    def test_flat_observation_ids_have_equivalent_tool_access(self):
        x = self.tone(100.) + self.tone(200.) + self.tone(300.)
        tools = AcousticTools(x, 8000, "opaque")
        tools.overview(include_demon=False)
        observations = tools.graph.measurements[0]["observed_components"]
        ids = [o["id"] for o in observations if o["type"] == "tone"][:3]
        self.assertEqual(len(ids), 3)
        tracked = tools.execute(dict(name="track_component", component_id=ids[0]))
        self.assertEqual(tracked["status"], "ok")
        harmonic = tools.execute(dict(name="test_harmonic_relation", component_ids=ids))
        self.assertEqual(harmonic["status"], "ok")
        self.assertTrue(harmonic["measured"]["supported"])
        overlap = tools.execute(dict(name="inspect_overlap", component_ids=ids[:2]))
        self.assertEqual(overlap["status"], "ok")
        self.assertFalse(overlap["graph_memory_used"])
        self.assertTrue(overlap["relations"])

    def test_numeric_claim_requires_correct_domain(self):
        state = dict(measurements=[dict(id="e0", peaks=[dict(frequency_hz=12., bin_spacing_hz=.25,
                                                            frequency_domain="modulation")])])
        result = verify_claims([dict(type="frequency", evidence_ids=["e0"], value_hz=12.,
                                     frequency_domain="acoustic")], state)
        self.assertEqual(len(result["accepted"]), 0)

    def test_exact_duplicate_is_false_positive(self):
        e = dict(type="tone", frequency_domain="acoustic", time_s=[1., 5.], frequency_hz=[100., 100.])
        score = event_metrics([e, e], [e])
        self.assertEqual((score["tp"], score["fp"], score["fn"]), (1, 1, 0))
        self.assertAlmostEqual(score["f1"], 2 / 3, places=6)
        partial = event_metrics([e], [e], exhaustive=False)
        self.assertIsNone(partial["f1"])

    def test_known_optional_beat_harmonic_is_not_false_positive(self):
        fundamental = dict(type="modulation", frequency_domain="modulation", time_s=[1., 5.],
                           frequency_hz=[12., 12.], carrier_band_hz=[500., 2000.])
        harmonic = dict(fundamental, frequency_hz=[24., 24.], evaluation_required=False)
        prediction = dict(harmonic)
        prediction.pop("evaluation_required")
        score = event_metrics([fundamental, prediction], [fundamental, harmonic])
        self.assertEqual((score["tp"], score["fp"], score["fn"]), (1, 0, 0))
        self.assertEqual(score["ignored_predictions"], [1])

    def test_chirp_track_coordinates(self):
        sr = 8000
        t = torch.arange(sr * 10, device=self.device) / sr
        x = torch.cos(2 * math.pi * (150 * t + 2 * t.square()))
        view = stft_psd(x, sr, offset_s=32.)
        proposed = propose_components(view)
        gold = [dict(type="chirp", frequency_domain="acoustic", time_s=[32., 42.],
                     frequency_hz=[150., 190.], frequency_track=[[32., 150.], [42., 190.]])]
        score = event_metrics(proposed, gold)
        self.assertEqual(score["tp"], 1)
        self.assertLess(score["matches"][0]["frequency_error_hz"], 1.)

    def test_bootstrap_uses_declared_parent_groups(self):
        a = [dict(tp=1, fp=0, fn=0)] * 4
        b = [dict(tp=0, fp=1, fn=1)] * 4
        result = grouped_bootstrap(a, b, ["a", "a", "b", "b"], repetitions=64)
        self.assertEqual(result["independent_groups"], 2)
        self.assertEqual(result["confidence_interval_95"], [1., 1.])
        adjusted = holm([.01, .04])
        self.assertAlmostEqual(adjusted[0], .02, places=6)
        self.assertAlmostEqual(adjusted[1], .04, places=6)

    def test_generator_disjoint_and_exact_intervention(self):
        a, truth, components = generate_scene(1, "train", "tones", duration_s=6., return_components=True)
        again, _ = generate_scene(1, "train", "tones", duration_s=6.)
        other, _ = generate_scene(1, "test", "tones", duration_s=6.)
        self.assertTrue(torch.equal(a, again))
        self.assertFalse(torch.equal(a, other))
        self.assertNotIn("tones", truth["scene_id"])
        removed = remove_component(a, components, "s0")
        expected = torch.stack([x for name, x in components.items() if name != "s0"]).sum(dim=0)
        self.assertLess((removed - expected).abs().max().item(), 2e-7)
        with self.assertRaises(ValueError):
            real_mixture(a, a, 16000, {"split": "train"}, {"split": "test"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
