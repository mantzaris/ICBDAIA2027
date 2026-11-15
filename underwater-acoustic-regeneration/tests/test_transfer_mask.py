import torch

from acoustic_regeneration.restoration.transfer import smooth_observation_mask


def test_padding_and_short_observed_regions_never_become_guidance():
    mask = torch.zeros(2, 180)
    mask[0, 30:80] = 1
    mask[0, 100:150] = 1
    mask[1, :4] = 1
    mask[1, -4:] = 1
    smooth = smooth_observation_mask(mask, 10)
    assert torch.equal(smooth[mask == 0], torch.zeros_like(smooth[mask == 0]))
    assert torch.all((smooth >= 0) & (smooth <= 1))
    assert torch.equal(smooth[0, 40:70], torch.ones(30))


def test_interior_gap_preserves_published_hann_ramps():
    mask = torch.ones(1, 100)
    mask[:, 40:60] = 0
    result = smooth_observation_mask(mask, 10)
    hann = torch.hann_window(20)
    assert torch.equal(result[0, 30:40], hann[10:])
    assert torch.equal(result[0, 60:70], hann[:10])
    assert torch.equal(result[0, :30], mask[0, :30])
    assert torch.equal(result[0, 70:], mask[0, 70:])
