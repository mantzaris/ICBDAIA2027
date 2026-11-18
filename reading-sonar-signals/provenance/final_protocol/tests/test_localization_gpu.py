import unittest
import torch
from sonar.signal import require_cuda
from sonar.synthetic import generate_scene
from sonar.localization import (DenseMorphologyCNN, localizer_features, dense_targets,
                                segmentation_loss, decode_components, auxiliary_broadband_scene, MAX_COMPONENTS)
from sonar.metrics import event_metrics


class LocalizerGPUValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_cuda()
        cls.waveform, cls.truth = generate_scene(990, "pilot", "tones", duration_s=12.)
        cls.features, cls.views = localizer_features(cls.waveform, 16000)
        cls.target, cls.valid = dense_targets(cls.views, cls.truth["events"])

    def test_auxiliary_broadband_has_valid_positive_masks_and_disjoint_seeds(self):
        waveform,truth=auxiliary_broadband_scene(10012,"train",10.)
        other,validation=auxiliary_broadband_scene(10012,"validation",10.)
        _,views=localizer_features(waveform,16000)
        target,valid=dense_targets(views,truth["events"],truth["modulation_exhaustive"])
        self.assertTrue((target[3]*valid[3]).sum()>0)
        self.assertEqual(valid[4:].sum().item(),0.)
        self.assertNotEqual(truth["generator_seed"],validation["generator_seed"])
        self.assertFalse(torch.equal(waveform,other))
        self.assertEqual(MAX_COMPONENTS,6)
        with self.assertRaises(ValueError): auxiliary_broadband_scene(1,"test",10.)

    def test_domains_and_parameters_stay_on_cuda(self):
        for dual in (False, True):
            model = DenseMorphologyCNN(dual_view=dual)
            result = model(self.features[None])
            self.assertEqual(result.shape, (1, 6 if dual else 4, 128, 128))
            self.assertTrue(all(p.device.type == "cuda" for p in model.parameters()))
            self.assertEqual(self.views[0].metadata["frequency_domain"], "acoustic")
            self.assertEqual(self.views[1].metadata["frequency_domain"], "modulation")

    def test_training_gradient_is_finite(self):
        model = DenseMorphologyCNN(dual_view=True)
        features = (self.features - self.features.mean()) / self.features.std().clamp_min(1e-6)
        loss = segmentation_loss(model(features[None]), self.target[None], self.valid[None], torch.ones(6, device="cuda"))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_oracle_mask_decoder_preserves_two_tones(self):
        # Decoder correctness only, not a trained-model performance claim.
        predicted = decode_components(self.target, self.views, .5)
        scores = event_metrics(predicted, self.truth["events"])
        self.assertEqual(scores["tp"], 2)

    def test_optional_harmonics_are_masked_from_training_negatives(self):
        waveform, truth = generate_scene(991, "pilot", "beating", duration_s=12.)
        _, views = localizer_features(waveform, 16000)
        target, valid = dense_targets(views, truth["events"])
        self.assertGreater((1 - valid[4]).sum().item(), 0)
        self.assertEqual((1 - valid[:4]).sum().item(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
