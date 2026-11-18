"""Separately labelled sensitivity probes, with all signal operations on CUDA.

Real spectral attenuation is an output-sensitivity probe, never source removal
ground truth. Display mismatch deliberately replaces only the initial DEMON
image; numerical measurements and later reinspection still use the target wave.
"""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import torch
from .batching import GPU_LOCK
from .data import load_waveform
from .signal import demon, resample
from .rendering import render_view


def localized_attenuation(x,sr,band,time_s,gain):
    """Invertible STFT-domain attenuation with smooth time/frequency shoulders."""
    nfft=4096; hop=1024
    window=torch.hann_window(nfft,device=x.device)
    spectrum=torch.stft(x,n_fft=nfft,hop_length=hop,window=window,return_complex=True)
    frequency=torch.fft.rfftfreq(nfft,d=1/sr,device=x.device)
    times=torch.arange(spectrum.shape[1],device=x.device)*hop/sr
    def envelope(coordinate,lo,hi,shoulder):
        weight=torch.minimum((coordinate-lo)/shoulder,(hi-coordinate)/shoulder).clamp(0,1)
        return .5-.5*torch.cos(torch.pi*weight)
    weights=envelope(frequency,*band,20.)[:,None]*envelope(times,*time_s,.5)[None,:]
    modified=spectrum*(1-(1-gain)*weights)
    return torch.istft(modified,n_fft=nfft,hop_length=hop,window=window,length=x.numel())


def load_experimental_waveform(record):
    if 'sensitivity_base' not in record:
        return load_waveform(record)
    base=record['sensitivity_base']; probe=record['sensitivity']
    x,sr,origin,truth=load_waveform(base)
    name=probe['name']
    if name=='duration':
        x=x[:round(probe['duration_seconds']*sr)]
        if truth is not None:
            from .metrics import _interpolate_track
            truth=copy.deepcopy(truth); retained=[]; end=origin+x.numel()/sr
            for event in truth.get('events',[]):
                bounds=torch.tensor(event['time_s'],device=x.device)
                bounds[0]=bounds[0].clamp_min(origin);bounds[1]=bounds[1].clamp_max(end)
                if bool(bounds[1]<=bounds[0]): continue
                track=event.get('frequency_track')
                if track:
                    samples=torch.linspace(bounds[0],bounds[1],3,device=x.device)
                    frequencies=_interpolate_track(event,samples)
                    event['frequency_track']=torch.stack([samples,frequencies],dim=1).tolist()
                    if event['type'] in ['tone','chirp','modulation']:
                        event['frequency_hz']=[frequencies.min().item(),frequencies.max().item()]
                event['time_s']=bounds.tolist(); retained.append(event)
            truth['events']=retained;truth['duration_seconds']=x.numel()/sr
    elif name in ['spectral_attenuation','sham_attenuation']:
        x=localized_attenuation(x,sr,probe['band_hz'],probe['scene_local_time_s'],probe['linear_gain'])
        truth=None
    elif name=='remove_real_parent':
        if base['stratum']!='semi_synthetic': raise ValueError('Parent removal requires a declared real-excerpt mixture')
        remaining=1-int(probe['removed_parent'])
        parent,parent_sr,_,_=load_waveform(base['parents'][remaining])
        parent=resample(parent,parent_sr,sr)
        offset=round(base.get('offset_seconds',0)*sr) if remaining else 0
        output=torch.zeros_like(x)
        # Reuse the exact original gains and common scale; do not renormalize.
        gain=truth['gain_second'] if remaining else 1.
        output[offset:offset+parent.numel()]=parent*gain/truth['global_scale']
        x=output
        truth=dict(truth,removed_parent=int(probe['removed_parent']),
            sensitivity_truth='known excerpt removal only; internal source events remain unknown')
    elif name=='display_mismatch':
        pass
    else:
        raise ValueError('Unknown separately labelled sensitivity probe')
    return x,sr,origin,truth


class DisplayOverride:
    def __init__(self,model,replacement,provenance):
        self.model=model;self.replacement=replacement;self.provenance=provenance

    def generate(self,prompt,images,max_new_tokens,**kwargs):
        actual=[self.replacement if Path(path).name=='global_demon.png' else path for path in images]
        generated=self.model.generate(prompt,actual,max_new_tokens,**kwargs)
        return dict(generated,display_intervention=dict(self.provenance,
            actual_input_paths=[str(path) for path in actual],
            replaced_initial_DEMON=any(Path(path).name=='global_demon.png' for path in images)))


def wrap_display_inputs(record,model,output):
    probe=record.get('sensitivity',{})
    if probe.get('name')!='display_mismatch': return model
    with GPU_LOCK:
        x,sr,_,_=load_waveform(probe['donor_record'])
        view=demon(x,sr,(500.,2000.),window_s=min(4.,x.numel()/sr-1),offset_s=0.)
    path=render_view(view,Path(output)/'mismatched_initial_demon.png','DEMON overview')
    provenance=dict(kind='initial_DEMON_image_from_other_scene',donor_scene_id=probe['donor_record']['scene_id'],
        image_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        numerical_measurements='target waveform',later_tool_views='target waveform',
        scientific_scope='deliberate image/measurement inconsistency; not a physically coherent transformed recording')
    return DisplayOverride(model,path,provenance)
