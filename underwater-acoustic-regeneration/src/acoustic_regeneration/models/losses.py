from __future__ import annotations

import torch


def deterministic_loss(predicted: torch.Tensor, reference: torch.Tensor,
                       mask: torch.Tensor, normalized_rms_floors: torch.Tensor) -> torch.Tensor:
    """Masked MAE plus 0.5 times unpadded gap spectral convergence."""
    hidden = ~mask.bool()
    if not hidden.any():
        raise ValueError("Restoration loss needs missing coordinates")
    loss = (predicted.float()[hidden] - reference.float()[hidden]).abs().mean()
    spectral = []
    for batch in range(len(predicted)):
        padded = torch.cat([torch.zeros(1, device=mask.device, dtype=torch.int8), hidden[batch].to(torch.int8),
                            torch.zeros(1, device=mask.device, dtype=torch.int8)])
        transitions = padded[1:] - padded[:-1]
        starts = torch.where(transitions == 1)[0].tolist()
        ends = torch.where(transitions == -1)[0].tolist()
        for size in (64, 128, 256):
            window = torch.hann_window(size, device=predicted.device)
            targets, estimates = [], []
            for start, end in zip(starts, ends):
                if end - start < size:
                    raise ValueError("Gap shorter than spectral window")
                targets.append(torch.stft(reference[batch, start:end].float(), size, hop_length=size // 4,
                                          window=window, center=False, normalized=True, return_complex=True).abs().flatten())
                estimates.append(torch.stft(predicted[batch, start:end].float(), size, hop_length=size // 4,
                                            window=window, center=False, normalized=True, return_complex=True).abs().flatten())
            target, estimate = torch.cat(targets), torch.cat(estimates)
            floor = normalized_rms_floors[batch] * (window.square().sum() / size * target.numel()).sqrt()
            spectral.append(torch.linalg.vector_norm(estimate - target) /
                            torch.maximum(torch.linalg.vector_norm(target), floor))
    return loss + 0.5 * torch.stack(spectral).mean()
