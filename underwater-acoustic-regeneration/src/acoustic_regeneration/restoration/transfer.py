"""Input-mask compatibility adapter for the pinned general-audio sampler."""
import torch


def smooth_observation_mask(mask: torch.Tensor, size: int = 50) -> torch.Tensor:
    """Use the published Hann ramps, bounded by signal edges and known samples.

    Unlike the upstream helper, a mask beginning at zero has no fictitious 1→0
    transition at index zero. Unknown padding is always excluded from projection.
    """
    if mask.ndim != 2 or size < 1 or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Expected a binary batch-by-sample observation mask")
    window = torch.hann_window(2 * size, dtype=mask.dtype, device=mask.device)
    output = mask.clone()
    for row in range(len(mask)):
        transitions = torch.where(mask[row, 1:] != mask[row, :-1])[0].cpu().tolist()
        for previous in transitions:
            index = previous + 1
            if mask[row, index] == 0:
                start = max(0, index - size)
                output[row, start:index] = torch.minimum(output[row, start:index], window[2 * size - (index - start):])
            else:
                end = min(mask.shape[1], index + size)
                output[row, index:end] = torch.minimum(output[row, index:end], window[:end - index])
    return output * mask
