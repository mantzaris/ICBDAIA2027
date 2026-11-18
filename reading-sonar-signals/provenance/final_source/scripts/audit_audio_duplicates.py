#!/usr/bin/env python3
"""CUDA near-duplicate screening; screening is not proof of vessel independence."""
import json
from pathlib import Path
import torch
from sonar.data import read_manifest,load_waveform
from sonar.signal import resample,stft_psd,require_cuda


def main():
    device=require_cuda();rows=read_manifest("data/manifests/real_scenes.jsonl")
    waveform_features=[];spectral_features=[]
    for row in rows:
        x,sr,_,_=load_waveform(row);x=resample(x,sr,4000.)[2000:-2000]
        x=x-x.mean();x=x/x.square().sum().sqrt().clamp_min(1e-15)
        waveform_features.append(x)
        view=stft_psd(x,4000.,window_s=1.,hop_s=.5).crop(frequency_hz=(10.,1800.))
        feature=view.psd.mean(dim=1).clamp_min(1e-20).log()
        feature=feature-feature.mean();feature=feature/feature.square().sum().sqrt().clamp_min(1e-15)
        spectral_features.append(feature)
    wave=torch.stack(waveform_features);spectral=torch.stack(spectral_features)
    wc=wave@wave.T;sc=spectral@spectral.T
    upper=torch.triu(torch.ones((len(rows),len(rows)),device=device,dtype=torch.bool),diagonal=1)
    candidates=((wc.abs()>=.995)|(sc>=.995))&upper
    indices=torch.nonzero(candidates)
    records=[]
    fft_size=1<<(2*wave.shape[1]-1).bit_length()
    for i,j in indices.tolist():
        correlation=torch.fft.irfft(torch.fft.rfft(wave[i],fft_size)*torch.fft.rfft(wave[j],fft_size).conj(),fft_size)
        # Full linear cross-correlation of unit-energy waveforms; FFT padding is
        # numerical correlation support, never a claim of improved resolution.
        peak=correlation.abs().max();lag=int(correlation.abs().argmax().item())
        if lag>fft_size//2:lag-=fft_size
        records.append(dict(scene_a=rows[i]["scene_id"],scene_b=rows[j]["scene_id"],
                            split_a=rows[i]["split"],split_b=rows[j]["split"],
                            group_a=rows[i]["group_id"],group_b=rows[j]["group_id"],
                            spectral_cosine=float(sc[i,j].item()),zero_lag_waveform_cosine=float(wc[i,j].item()),
                            maximum_absolute_waveform_correlation=float(peak.item()),lag_seconds=lag/4000,
                            suspected_duplicate=bool(peak>=.995)))
    diagonal=wc.diag()
    if not bool(torch.allclose(diagonal,torch.ones_like(diagonal),atol=1e-5)):
        raise RuntimeError("Waveform normalization failed, including possible zero recording")
    cross=[x for x in records if x["suspected_duplicate"] and x["split_a"]!=x["split_b"]]
    report=dict(scenes=len(rows),pairs=int(upper.sum().item()),candidate_pairs=len(records),
                suspected_duplicate_pairs=sum(x["suspected_duplicate"] for x in records),
                cross_split_suspected_duplicates=len(cross),candidates=records,device=str(device),
                method="CUDA normalized waveform cosine and centered log-PSD cosine screening; FFT correlation on candidates",
                thresholds=dict(waveform_absolute_correlation=.995,spectral_cosine=.995),
                caveats=["Screening can miss highly transformed or partial duplicate recordings",
                         "Common spectral shape alone is not a duplicate",
                         "No screening result establishes vessel/session identity or independence",
                         "No sibling UniqueShip source audio was included in this screen"])
    Path("reports").mkdir(exist_ok=True)
    Path("reports/audio_duplicate_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="candidates"},indent=2),flush=True)
    if cross:raise RuntimeError("Cross-split suspected duplicate requires investigation before final evaluation")


if __name__=="__main__":main()
