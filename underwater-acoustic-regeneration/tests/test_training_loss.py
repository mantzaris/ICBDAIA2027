import pytest
import torch

from acoustic_regeneration.models.losses import deterministic_loss


def test_deterministic_loss_has_no_gradient_on_observed_samples():
    reference = torch.ones(1, 1024)
    prediction = torch.zeros(1, 1024, requires_grad=True)
    mask = torch.ones_like(reference, dtype=torch.bool)
    mask[:, 300:700] = False
    loss = deterministic_loss(prediction, reference, mask, torch.tensor([1e-6]))
    assert loss.item() == pytest.approx(1.5)
    loss.backward()
    assert torch.all(prediction.grad[mask] == 0)
    assert torch.isfinite(prediction.grad).all()


def test_deterministic_loss_does_not_cross_between_separated_gaps():
    reference = torch.randn(1, 4096)
    prediction = reference.clone().requires_grad_()
    mask = torch.ones_like(reference, dtype=torch.bool)
    mask[:, 500:900] = False
    mask[:, 2500:2900] = False
    assert deterministic_loss(prediction, reference, mask, torch.tensor([1e-6])).item() == 0
