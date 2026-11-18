"""Two positive, blinded calibration controls; minimal acoustic-only geometry."""
import argparse
import math
import gc
import json
from pathlib import Path
import torch
from sonar.annotators import AnnotationVLM,render_reference_view,render_modulation_reference,parse_complete_event_rows
from sonar.data import read_manifest,load_waveform
from sonar.signal import lofar,demon,bandpass
from sonar.metrics import event_metrics

parser=argparse.ArgumentParser();parser.add_argument('--annotator',required=True,choices=['internvl','qwen25'])
parser.add_argument('--few-shot',action='store_true')
parser.add_argument('--config',default='configs/ai_annotations.json')
parser.add_argument('--output')
parser.add_argument('--psd-panel',action='store_true')
parser.add_argument('--raw-psd-grid',action='store_true')
args=parser.parse_args();config=json.loads(Path(args.config).read_text())
config['max_new_tokens']=512
out=Path(args.output or ('results/ai_annotation_fewshot_pilot' if args.few_shot else 'results/ai_annotation_minimal_pilot'))/args.annotator;out.mkdir(parents=True,exist_ok=True)
model=AnnotationVLM(config['models'][args.annotator],config)
available=read_manifest('data/manifests/annotation_controls.jsonl')
records=[available[0],available[7]]
for record in records:
    wave,sr,offset,truth=load_waveform(record)
    is_modulation=record is records[1]
    view=demon(wave,sr,(2000.,5000.),window_s=4.,offset_s=offset) if is_modulation else lofar(wave,sr,window_s=1.,offset_s=offset)
    image=out/(record['scene_id']+'.png')
    scale=render_modulation_reference(view,image) if is_modulation and args.psd_panel else render_reference_view(view,image,'Envelope modulation spectrogram' if is_modulation else 'Acoustic spectrogram',max_pool_frequency_bins=360)
    prompt=('Inspect this envelope-modulation spectrogram. Time is horizontal in seconds; modulation frequency is vertical in Hz.\n' if is_modulation else 'Inspect this acoustic spectrogram. Time is horizontal in seconds; acoustic frequency is vertical in Hz.\n')+'''
Find bright thin horizontal or diagonal lines, distinct from the speckled background.
Ignore the speckled background and do not label the whole image as an event.
Read the visible axes to estimate each line's starting/ending time and frequency.
Each event extends over its whole visible continuous line. Do not split one continuous line into adjacent short segments.
Return ONLY a JSON object with one key named events. Its value is a list of at most six arrays.
Each array contains four measured numbers followed by one type string, in this order:
starting time, ending time, starting frequency, ending frequency, type.
Return an empty list when no bright persistent line is visible. Do not discuss sources.
'''+('The type string must be modulation. Carrier-band numbers in the title are NOT modulation frequencies.\n' if is_modulation else 'The type string is tone for a horizontal line, or chirp for a sloping line.\n')
    if is_modulation and args.psd_panel:prompt+='The left chart shows when a modulation line is present; the right chart shows its frequency as a spectral peak. Use both charts.\n'
    base_prompt=prompt
    def grid(v):
        # Every fixed one-Hz cell is supplied, with no peak selection or labels.
        power=v.psd.mean(dim=1)
        values=[]
        for lower in range(200):
            selected=(v.frequencies_hz>=lower)&(v.frequencies_hz<lower+1)
            if selected.any():values.append([lower+.5,round(float((10*power[selected].mean().clamp_min(1e-20).log10()).item()),2)])
        return json.dumps(values,separators=(',',':'))
    if is_modulation and args.raw_psd_grid:prompt+='Complete fixed frequency grid [modulation_Hz, mean_PSD_dB] from this target, not detected peaks: '+grid(view)+'\n'
    examples=[]
    if args.few_shot:
        # Fixed separate TRAINING demonstrations, not either target's parameters.
        # Frequencies deliberately differ from the target controls, so copying
        # example coordinates will fail the target localization audit.
        demo_rate=16000;t=torch.arange(30*demo_rate,device='cuda')/demo_rate
        generator=torch.Generator(device='cuda').manual_seed(920007 if is_modulation else 920000)
        if is_modulation:
            noise=bandpass(torch.randn(t.numel(),device='cuda',generator=generator),demo_rate,(2500.,4500.))
            gate=((t-5)/.1).clamp(0,1)*((25-t)/.1).clamp(0,1)
            demo_wave=noise*(1+.8*torch.cos(2*math.pi*37*t))*gate
            demo_view=demon(demo_wave,demo_rate,(2000.,5000.),window_s=4.)
            answer={'events':[[5.,25.,37.,37.,'modulation']]}
        else:
            gate_a=((t-2)/.1).clamp(0,1)*((28-t)/.1).clamp(0,1)
            gate_b=((t-10)/.1).clamp(0,1)*((20-t)/.1).clamp(0,1)
            demo_wave=.6*torch.cos(2*math.pi*120*t)*gate_a+.4*torch.cos(2*math.pi*440*t)*gate_b+.03*torch.randn(t.numel(),device='cuda',generator=generator)
            demo_view=lofar(demo_wave,demo_rate,window_s=1.)
            answer={'events':[[2.,28.,120.,120.,'tone'],[10.,20.,440.,440.,'tone']]}
        demo_image=out/('training_demo_modulation.png' if is_modulation else 'training_demo_acoustic.png')
        if is_modulation and args.psd_panel:render_modulation_reference(demo_view,demo_image)
        else:render_reference_view(demo_view,demo_image,'Envelope modulation spectrogram' if is_modulation else 'Acoustic spectrogram',max_pool_frequency_bins=360)
        demo_prompt=base_prompt
        if is_modulation and args.raw_psd_grid:demo_prompt+='Complete fixed frequency grid [modulation_Hz, mean_PSD_dB] from this training example, not detected peaks: '+grid(demo_view)+'\n'
        examples=[dict(image=str(demo_image),prompt=demo_prompt,answer=answer,
                       provenance={'role':'separate_exact_training_demonstration','seed':920007 if is_modulation else 920000,'target_truth_used':False})]
    result=model.generate(prompt,[str(image)],max_new_tokens=512,examples=examples)
    result['parsed'],result['syntax_validation']=parse_complete_event_rows(result['raw_text'])
    events=[];errors=[]
    if result['parsed'] is not None:
        for i,row in enumerate(result['parsed'].get('events',[])):
            try:
                t0,t1,f0,f1=[float(v) for v in row[:4]];typ=row[4]
                if typ not in (['modulation'] if is_modulation else ['tone','chirp']) or not 0<=t0<t1<=30 or not 0<=min(f0,f1)<=max(f0,f1)<=(200 if is_modulation else 1800):raise ValueError('Invalid geometry')
                events.append(dict(id=str(i),type=typ,frequency_domain='modulation' if is_modulation else 'acoustic',time_s=[t0,t1],
                                   frequency_hz=[min(f0,f1),max(f0,f1)],frequency_track=[[t0,f0],[t1,f1]],
                                   bin_spacing_hz=scale['display_frequency_bin_width_hz']))
                if is_modulation:events[-1]['carrier_band_hz']=[2000.,5000.]
            except (KeyError,IndexError,TypeError,ValueError) as exc:errors.append(str(exc))
    metrics=event_metrics(events,truth['events'],frequency_tolerance_hz=4.,exhaustive=True)
    output=dict(scene_id=record['scene_id'],prompt=prompt,generation=result,events=events,errors=errors,
                metrics=metrics,display_scale=scale,scope='training calibration only; no real/test labels',
                localization_tolerance='max(4 Hz, two displayed frequency bins); exact errors retained in matches')
    (out/(record['scene_id']+'.json')).write_text(json.dumps(output,indent=2))
    print(args.annotator,record['scene_id'],events,metrics,flush=True)
del model;gc.collect();torch.cuda.empty_cache()
