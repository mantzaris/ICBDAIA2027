"""Host-only display rendering of measurements already computed on CUDA."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from threading import RLock
from .batching import GPU_LOCK

# Matplotlib's global state is not thread-safe; the model GPU worker can still
# generate while host rendering proceeds on a different episode.
PLOT_LOCK=RLock()


def render_view(view, path, title='', components=(), max_frequency_hz=None, cmap='magma'):
    with PLOT_LOCK:
        return _render_view(view,path,title,components,max_frequency_hz,cmap)


def _render_view(view, path, title='', components=(), max_frequency_hz=None,cmap='magma'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with GPU_LOCK:
        power = view.log_power()
        if max_frequency_hz is not None:
            selected = view.frequencies_hz <= max_frequency_hz
            power = power[selected]
            frequencies = view.frequencies_hz[selected]
        else:
            frequencies = view.frequencies_hz
        # Preserve sparse ridges under raster downsampling. This is a display
        # aggregation only; every quantitative tool uses the original PSD.
        maximum=power.max(); minimum=power.min()
        limits=[maximum.item()-60.,maximum.item()]
        if bool(maximum==minimum): limits=[maximum.item(),maximum.item()+60.]
        if power.shape[0]>256:
            original_bins=power.shape[0]
            low=torch.floor(torch.arange(256,device=power.device)*original_bins/256).long()
            high=torch.ceil((torch.arange(256,device=power.device)+1)*original_bins/256).long()-1
            frequencies=(frequencies[low]+frequencies[high])/2
            power=torch.nn.functional.adaptive_max_pool2d(power[None,None],(256,power.shape[1]))[0,0]
        times = view.times_s.detach().cpu().numpy()
        freq = frequencies.detach().cpu().numpy()
        display_power=power.detach().cpu().numpy()
    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    mesh = axis.pcolormesh(times, freq, display_power, shading='nearest',
                           cmap=cmap, vmin=limits[0], vmax=limits[1])
    domain = view.metadata['frequency_domain']
    axis.set(xlabel='Original-recording time (s)',
             ylabel=f'{domain.capitalize()} frequency (Hz)', title=title)
    if domain == 'modulation':
        band = view.metadata.get('carrier_band_hz')
        axis.set_title(f'{title} | carrier {band[0]:g}–{band[1]:g} Hz')
    for component in components:
        if component.get('frequency_domain', 'acoustic') != domain:
            continue
        track = component.get('frequency_track')
        if track:
            axis.plot([p[0] for p in track], [p[1] for p in track], linewidth=1,
                      label=component.get('id', 'component'))
    figure.colorbar(mesh, ax=axis, label='Display max-pooled PSD (dB re digital amplitude²/Hz)')
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return str(path)
