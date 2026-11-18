"""Label-free waveform input boundary; CPU use is restricted to file decoding."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def read_manifest(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_waveform(record):
    import soundfile as sf
    import torch
    from .signal import require_cuda
    from .synthetic import generate_scene, real_mixture
    device = require_cuda()
    if 'base_record' in record:
        from .signal import resample
        base=record['base_record']
        waveform,sample_rate,origin,truth=load_waveform(base)
        intervention=record['intervention']
        name=intervention['name']
        if name=='gain':
            waveform=waveform*float(intervention['linear_gain'])
        elif name=='added_noise':
            generator=torch.Generator(device=device).manual_seed(intervention['seed'])
            noise=torch.randn(waveform.shape,device=device,generator=generator)
            noise=noise/noise.square().mean().sqrt().clamp_min(1e-12)
            scale=waveform.square().mean().sqrt()*10**(-intervention['nominal_ratio_db']/20)
            waveform=waveform+scale*noise
        elif name=='bandwidth_loss':
            target=int(intervention['sample_rate'])
            waveform=resample(waveform,sample_rate,target); sample_rate=target
            if truth:
                # The original carrier reference no longer denotes the same
                # waveform bandwidth. Score acoustic structure only in this
                # condition; separately record unavailable carrier evidence.
                truth['unavailable_modulation_events']=[e for e in truth['events'] if e.get('frequency_domain')=='modulation']
                truth['events']=[e for e in truth['events'] if e.get('frequency_domain')!='modulation']
                truth['modulation_exhaustive']=False
                truth['evaluation_scope']='acoustic events only; original carrier-defined modulation references unavailable after bandwidth loss'
        elif name=='remove_component':
            from .synthetic import remove_component
            if base['stratum']!='synthetic': raise ValueError('Exact component removal requires synthetic truth')
            waveform,truth,components=generate_scene(base['seed'],base['split'],base['kind'],
                sample_rate=base.get('sample_rate',16000),duration_s=base.get('duration_seconds',30),
                snr_db=base.get('snr_db',10),return_components=True)
            cid=intervention.get('component_id','s0')
            waveform=remove_component(waveform,components,cid)
            truth['events']=[e for e in truth['events'] if e.get('component_waveform_id')!=cid
                             and not (base['kind']=='beating' and e.get('frequency_domain')=='modulation')]
        elif name=='blank':
            waveform=torch.zeros_like(waveform)
            if truth: truth=dict(truth,events=[],modulation_exhaustive=True)
        elif name=='plot_style':
            pass  # Only final rendering changes; waveform and truth stay intact.
        else: raise ValueError('Unknown intervention')
        return waveform,sample_rate,origin,truth
    if record['stratum'] == 'synthetic':
        waveform, truth = generate_scene(record['seed'], record['split'], record['kind'],
                                         sample_rate=record.get('sample_rate',16000),
                                         duration_s=record.get('duration_seconds',30),
                                         snr_db=record.get('snr_db',10))
        return waveform, int(truth['sample_rate_hz']), 0., truth
    if record['stratum'] == 'semi_synthetic':
        from .signal import resample
        a, sr_a, _, _ = load_waveform(record['parents'][0])
        b, sr_b, _, _ = load_waveform(record['parents'][1])
        sr = min(sr_a,sr_b,16000)
        a, b = resample(a,sr_a,sr), resample(b,sr_b,sr)
        waveform, truth = real_mixture(a,b,sr,record['parents'][0],record['parents'][1],
                                       offset_s=record.get('offset_seconds',0),ratio_db=record.get('ratio_db',0))
        return waveform, sr, 0., truth
    audio, sample_rate = sf.read(record['path'],dtype='float32',always_2d=True)
    waveform = torch.from_numpy(audio).to(device=device,dtype=torch.float32).mean(dim=1)
    return waveform, sample_rate, float(record.get('source_start_seconds',0)), None


def opaque_id(namespace, value):
    return namespace+'_'+hashlib.sha256(str(value).encode()).hexdigest()[:16]


def input_metadata(record):
    """This whitelist is the only manifest metadata permitted in model inputs."""
    return {k:record[k] for k in ['scene_id','duration_seconds','sample_rate'] if k in record}
