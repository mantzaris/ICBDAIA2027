#!/usr/bin/env python3
"""Blinded per-domain AI reference labels and prespecified calibration eligibility.

This runner keeps all raw outputs, exact training demonstrations, fixed raw PSD
grids, rejected event rows, and bounded syntax repairs. It never supplies the
target's generator truth, source labels, or controller predictions to an AI.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path
import torch
from sonar.annotators import AnnotationVLM,render_reference_view,render_modulation_reference,parse_complete_event_rows,file_hash
from sonar.data import load_waveform,read_manifest
from sonar.signal import lofar,demon,bandpass
from sonar.metrics import event_metrics,aggregate_event_metrics
from sonar.reference_protocol import object_hash,protocol_identity,verify_frozen,verify_record,select_controls

TYPES=['tone','chirp','transient','broadband','modulation']
VIEWS=[('lofar_1s','acoustic',1.,None),('lofar_4s','acoustic',4.,None),
       ('demon_500_2000','modulation',4.,(500.,2000.)),('demon_2000_5000','modulation',4.,(2000.,5000.))]


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def raw_grid(view):
    power=view.psd.mean(dim=1);values=[]
    for lower in range(200):
        selected=(view.frequencies_hz>=lower)&((view.frequencies_hz<=lower+1) if lower==199 else (view.frequencies_hz<lower+1))
        if selected.any():values.append([lower+.5,round(float((10*power[selected].mean().clamp_min(1e-20).log10()).item()),2)])
    return dict(columns=['modulation_frequency_hz','mean_PSD_dB_re_digital_amplitude_squared_per_Hz'],
                cells=values,selection='complete_fixed_one_Hz_cells_no_peak_selection',last_cell_includes_200Hz_endpoint=True,numeric_device=str(power.device))


def prompt(domain,duration,grid=None):
    text=f'''Inspect the supplied {'acoustic' if domain=='acoustic' else 'envelope-modulation'} evidence.
Time is horizontal in seconds in the spectrogram, from 0 to {duration:g}; frequency is vertical in Hz.
Find localized structure distinct from the speckled background. A whole plotting rectangle is not an event.
One continuous line is one event over its full visible duration; do not divide it into adjacent pieces.
Return ONLY a JSON object with key events. Its value is a list of at most eight arrays.
Each array contains four measured numbers followed by a type string: starting time, ending time,
starting frequency, ending frequency, type. Return an empty list if no localized structure is supported.
Do not identify vessels, machinery, blades, source power, range, or absolute acoustic SPL.
'''
    if domain=='acoustic':
        text+='A horizontal line is tone; a sloping line is chirp. A short wide patch is transient; a sustained wide patch is broadband. For wide patches the frequency numbers are the lower and upper bounds. Ignore diffuse background without a localized event.\n'
    else:
        text+='Type is modulation. Carrier-band numbers in the title are NOT modulation frequencies. Use the time plot for duration and the mean spectrum for frequency. Periodic ridges are events, but slow onset/offset leakage alone is not a periodic component.\n'
        text+='Complete fixed one-Hz raw mean-PSD grid (all cells, no selected peaks): '+json.dumps(grid,separators=(',',':'))+'\n'
    return text


def make_view(wave,sr,domain,window,band):
    return lofar(wave,sr,window_s=window,offset_s=0.) if domain=='acoustic' else demon(wave,sr,band,window_s=window,offset_s=0.)


def render(view,path,domain):
    return render_reference_view(view,path,'Acoustic spectrogram',max_pool_frequency_bins=360,time_label='Scene-local time (s)') if domain=='acoustic' else render_modulation_reference(view,path,time_label='Scene-local time (s)')


def demonstration(domain,window,band,directory):
    directory.mkdir(parents=True,exist_ok=True);sr=16000
    t=torch.arange(30*sr,device='cuda')/sr
    def gate(start,end):return ((t-start)/.1).clamp(0,1)*((end-t)/.1).clamp(0,1)
    if domain=='acoustic':
        g=torch.Generator(device='cuda').manual_seed(920000)
        a=.6*torch.cos(2*math.pi*120*t)*gate(2,28)+.4*torch.cos(2*math.pi*440*t)*gate(10,20)+.03*torch.randn(t.numel(),device='cuda',generator=g)
        cases=[('lines',a,[[2.,28.,120.,120.,'tone'],[10.,20.,440.,440.,'tone']],920000)]
        g=torch.Generator(device='cuda').manual_seed(920001)
        phase=2*math.pi*(250*(t-4)+.5*(300/22)*(t-4).square())
        b=.5*torch.cos(phase)*gate(4,26)
        noise=torch.randn(t.numel(),device='cuda',generator=g)
        b+=1.2*bandpass(noise,sr,(900.,1500.))*gate(15,15.6)
        b+=.7*bandpass(noise,sr,(600.,850.))*gate(8,20)
        b+=.02*torch.randn(t.numel(),device='cuda',generator=g)
        cases.append(('morphology',b,[[4.,26.,250.,550.,'chirp'],[15.,15.6,900.,1500.,'transient'],[8.,20.,600.,850.,'broadband']],920001))
    else:
        seed=920007+int(band[0]);g=torch.Generator(device='cuda').manual_seed(seed)
        lo,hi=band[0]+.2*(band[1]-band[0]),band[1]-.2*(band[1]-band[0])
        noise=bandpass(torch.randn(t.numel(),device='cuda',generator=g),sr,(lo,hi))
        a=noise*(1+.8*torch.cos(2*math.pi*37*t))*gate(5,25)
        cases=[('modulation',a,[[5.,25.,37.,37.,'modulation']],seed)]
    examples=[]
    for name,wave,events,seed in cases:
        v=make_view(wave,sr,domain,window,band);path=directory/(name+'.png');scale=render(v,path,domain)
        grid=raw_grid(v) if domain=='modulation' else None
        example=dict(image=str(path),prompt=prompt(domain,30,grid),answer={'events':events},
                     provenance=dict(role='exact_separate_synthetic_training_demonstration',seed=seed,
                                     target_truth_used=False,display_scale=scale,raw_grid=grid))
        examples.append(example)
    save(directory/'examples.json',examples)
    return examples


def rows_to_events(parsed,domain,band,duration,offset,window,display_width,tag):
    events=[];errors=[]
    if parsed is None or not isinstance(parsed.get('events'),list):return events,['Missing valid events list']
    for i,row in enumerate(parsed['events']):
        try:
            if not isinstance(row,list) or len(row)!=5:raise ValueError('Expected four numbers and type')
            if i>=8:raise ValueError('Eight-event per-view budget exceeded')
            if not all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in row[:4]):raise ValueError('Geometry must contain numbers')
            a,b,f,g=[float(v) for v in row[:4]];typ=row[4]
            if not all(math.isfinite(v) for v in [a,b,f,g]):raise ValueError('Nonfinite geometry')
            if not 0<=a<b<=duration or not 0<=min(f,g)<=max(f,g)<=(1800 if domain=='acoustic' else 200):raise ValueError('Outside displayed scene')
            if typ not in (TYPES[:-1] if domain=='acoustic' else ['modulation']):raise ValueError('Ineligible view-domain type')
            times=(torch.tensor([a,b],device='cuda')+offset).tolist()
            event=dict(id=f'{tag}:{i}',type=typ,frequency_domain=domain,time_s=times,frequency_hz=[min(f,g),max(f,g)],
                       bin_spacing_hz=display_width,frequency_uncertainty_hz=display_width,
                       display_frequency_bin_width_hz=display_width,
                       boundary_uncertainty_s=window/2,annotation_origin='measurement_assisted_AI_reference',
                       independent_ground_truth=False,source_view=tag)
            if typ in ['tone','chirp','modulation']:event['frequency_track']=[[times[0],f],[times[1],g]]
            if domain=='modulation':event['carrier_band_hz']=list(band)
            events.append(event)
        except (TypeError,ValueError) as exc:errors.append(dict(row_index=i,raw_row=row,error=str(exc)))
    return events,errors


def merge_resolution_duplicates(coarse,fine):
    matching=event_metrics(fine,coarse,frequency_tolerance_hz=4.,exhaustive=True)
    matched_fine={x['prediction_index'] for x in matching['matches']}
    # Preserve the coarse view's better temporal localization for a matched
    # duplicate, while keeping its fine-view ID as corroborating provenance.
    result=[dict(x) for x in coarse]
    for pair in matching['matches']:
        result[pair['gold_index']]['corroborating_view_event_id']=fine[pair['prediction_index']]['id']
    result.extend(x for i,x in enumerate(fine) if i not in matched_fine)
    return result,matching


def score_controls(stored,truth,metrics,coverage):
    scored=[dict(e,bin_spacing_hz=0.) for e in stored['annotation']['events']]
    for typ in TYPES:
        domain='modulation' if typ=='modulation' else 'acoustic'
        if domain=='modulation' and not truth.get('modulation_exhaustive',False):continue
        passes=[p for p in stored['generation']['passes'] if p['view'].startswith('demon' if domain=='modulation' else 'lofar')]
        valid=sum(not p['errors'] for p in passes)
        coverage[typ].append((valid,len(passes)))
        rejected=sum(isinstance(error,dict) and isinstance(error.get('raw_row'),list) and bool(error['raw_row']) and error['raw_row'][-1]==typ
                     for p in passes for error in p['errors'])
        row=event_metrics([e for e in scored if e['type']==typ],[e for e in truth['events'] if e['type']==typ],frequency_tolerance_hz=4.,exhaustive=True)
        row['fp']+=rejected;row['rejected_typed_rows']=rejected
        metrics[typ].append(row)


def summarize_controls(metrics,coverage,policy,qualify):
    result={}
    for typ,rows in metrics.items():
        metric=aggregate_event_metrics(rows)
        counts=torch.tensor(coverage[typ],device='cuda',dtype=torch.float64)
        total=counts.sum(dim=0) if counts.numel() else torch.zeros(2,device='cuda')
        fraction=float((total[0]/total[1]).item()) if total[1]>0 else 0.
        metric.update(valid_relevant_views=int(total[0].item()),relevant_views=int(total[1].item()),valid_relevant_view_fraction=fraction,
                      rejected_typed_rows=sum(r.get('rejected_typed_rows',0) for r in rows))
        if qualify:
            metric['eligible']=bool(metric['tp']+metric['fn']>=policy['minimum_ground_truth_events'] and metric['tp']>=policy['minimum_matched_calibration_events']
                and (metric['precision'] or 0)>=policy['minimum_calibration_precision'] and (metric['recall'] or 0)>=policy['minimum_calibration_recall']
                and fraction>=policy['minimum_valid_relevant_view_fraction'])
        result[typ]=metric
    return result


def main():
    run_started=time.perf_counter()
    ap=argparse.ArgumentParser();ap.add_argument('--model-size',choices=['4b','8b'],required=True)
    ap.add_argument('--output');ap.add_argument('--calibration-only',action='store_true');ap.add_argument('--manifest',action='append')
    ap.add_argument('--frozen-protocol');ap.add_argument('--batch-size',type=int)
    args=ap.parse_args()
    execution=json.loads(Path('configs/ai_reference_execution.json').read_text())
    if execution['selection_status']!='fixed_before_full_calibration':raise RuntimeError('Fix execution mode after training-fixture smoke before calibration')
    if args.batch_size is not None and args.batch_size!=execution['batch_size']:raise RuntimeError('Execution mode differs from fixed protocol')
    args.batch_size=execution['batch_size']
    if args.batch_size<1 or args.batch_size>4:ap.error('Reference batches must contain1–4 fixed view requests')
    if not args.calibration_only and not args.frozen_protocol:ap.error('Final labels require the recorded frozen protocol file')
    config_path='configs/ai_annotations.json' if args.model_size=='4b' else 'configs/ai_annotations_internvl8b_pilot.json'
    identity=protocol_identity(config_path)
    policy=json.loads(Path('configs/ai_reference_eligibility.json').read_text())
    freeze=None
    if args.frozen_protocol:
        freeze=json.loads(Path(args.frozen_protocol).read_text());verify_frozen(freeze,identity)
        for path,expected_hash in freeze['expected_input_manifest_sha256'].items():
            if file_hash(path)!=expected_hash:raise RuntimeError('Frozen input manifest changed: '+path)
    config=json.loads(Path(config_path).read_text());config['max_new_tokens']=768
    spec=config['models']['internvl'];name='internvl'+args.model_size
    controls=read_manifest('data/manifests/annotation_controls.jsonl')
    calibration=select_controls(controls,policy,'calibration');audit=select_controls(controls,policy,'audit')
    if args.calibration_only:records=calibration
    else:
        records=[]
        for path in args.manifest or ['data/manifests/real_scenes.jsonl','data/manifests/mixture_scenes.jsonl']:
            if path not in freeze['expected_input_manifest_sha256']:raise RuntimeError('Unfrozen input manifest: '+path)
            records.extend(read_manifest(path))
        records=audit+records
    if len({r['scene_id'] for r in records})!=len(records):raise RuntimeError('Duplicate scene IDs')
    output=Path(args.output or ('results/ai_reference_calibration' if args.calibration_only else 'results/ai_annotations'))
    directory=output/name;directory.mkdir(parents=True,exist_ok=True)
    model=AnnotationVLM(spec,config)
    batch_adapter=None
    if args.batch_size>1:
        from sonar.annotation_batching import AnnotationBatchAdapter
        batch_adapter=AnnotationBatchAdapter(model,max_batch_size=args.batch_size)
    examples={v[0]:demonstration(v[1],v[2],v[3],output/'training_examples'/v[0]) for v in VIEWS}
    example_identity={tag:[dict(image_sha256=file_hash(x['image']),prompt=x['prompt'],answer=x['answer'],provenance=x['provenance']) for x in cases] for tag,cases in examples.items()}
    example_hash=object_hash(example_identity)
    totals=[];class_rows={t:[] for t in TYPES};coverage={t:[] for t in TYPES}
    for record in records:
        sid=record['scene_id'];destination=directory/(sid+'.json');truth=None
        if destination.exists():
            stored=json.loads(destination.read_text());verify_record(stored,identity,record)
            if stored['generation']['model_revision']!=spec['revision'] or stored['generation']['model_id']!=spec['model_id']:raise RuntimeError('Model identity changed')
            if stored.get('training_demonstrations_sha256')!=example_hash:raise RuntimeError('Training demonstration identity changed')
            if not args.calibration_only and stored.get('frozen_protocol_sha256')!=file_hash(args.frozen_protocol):raise RuntimeError('Frozen protocol identity changed')
            if stored.get('batch_size')!=args.batch_size:raise RuntimeError('Cached execution mode changed')
            if record.get('annotation_control'):
                wave,sr,origin,truth=load_waveform(record);del wave
        else:
            scene_started=time.perf_counter()
            wave,sr,origin,truth=load_waveform(record);duration=wave.numel()/sr
            passes=[];by_view={};prepared=[]
            for tag,domain,window,band in VIEWS:
                view=make_view(wave,sr,domain,window,band)
                image=output/'views'/sid/(tag+'.png');image.parent.mkdir(parents=True,exist_ok=True)
                scale=render(view,image,domain);grid=raw_grid(view) if domain=='modulation' else None
                text=prompt(domain,duration,grid)
                prepared.append(dict(tag=tag,domain=domain,window=window,band=band,image=image,scale=scale,grid=grid,text=text))
            generated_outputs=[]
            for batch_start in range(0,len(prepared),args.batch_size):
                requests=[dict(prompt=p['text'],images=[str(p['image'])],max_new_tokens=768,examples=examples[p['tag']]) for p in prepared[batch_start:batch_start+args.batch_size]]
                generated_outputs.extend(batch_adapter.generate_batch(requests) if batch_adapter else [model.generate(**r) for r in requests])
            for prepared_pass,generated in zip(prepared,generated_outputs):
                tag,domain,window,band,image,scale,grid,text=[prepared_pass[k] for k in ['tag','domain','window','band','image','scale','grid','text']]
                parsed,syntax=parse_complete_event_rows(generated['raw_text'])
                events,errors=rows_to_events(parsed,domain,band,duration,origin,window,scale['display_frequency_bin_width_hz'],tag)
                by_view[tag]=events
                passes.append(dict(view=tag,generation=generated,prompt=text,image=str(image),image_sha256=file_hash(image),
                                   raw_grid=grid,display_scale=scale,syntax_validation=syntax,events=events,errors=errors,
                                   original_coordinate_offset_s=origin,display_coordinate_system='scene-local'))
            acoustic,merge=merge_resolution_duplicates(by_view['lofar_1s'],by_view['lofar_4s'])
            all_events=acoustic+by_view['demon_500_2000']+by_view['demon_2000_5000']
            generation=dict(model_id=spec['model_id'],model_revision=spec['revision'],passes=passes,
                            gpu_seconds=sum(p['generation']['gpu_seconds'] for p in passes),
                            output_tokens=sum(p['generation']['output_tokens'] for p in passes),
                            wall_seconds=time.perf_counter()-scene_started,
                            wall_seconds_scope='scene_decode_fixed_views_rendering_and_inference',
                            inference_request_sojourn_seconds_sum=sum(p['generation']['wall_seconds'] for p in passes),
                            gpu_seconds_scope='model_generation_only; equal_shares_sum_once_for_batched_requests',
                            syntax_repairs=sum(p['syntax_validation'].get('syntax_repair_applied',False) for p in passes))
            stored=dict(scene_id=sid,annotation=dict(events=all_events,annotation_status='complete' if all(not p['errors'] for p in passes) else 'partial',
                                                    invalid_events=[e for p in passes for e in p['errors']],independently_exhaustive=False),
                        generation=generation,scene_interval_s=[origin,origin+duration],
                        annotation_origin='measurement_assisted_AI_reference',source_labels_supplied=False,
                        controller_candidates_supplied=False,target_simulator_truth_supplied=False,
                        resolution_duplicate_matching=merge,independent_ground_truth=False,
                        eligibility_protocol_sha256=identity['eligibility_protocol_sha256'],
                        protocol_identity=identity,input_record_sha256=object_hash(record),
                        training_demonstrations_sha256=example_hash,
                        frozen_protocol_sha256=file_hash(args.frozen_protocol) if args.frozen_protocol else None,
                        batch_size=args.batch_size,execution_config_sha256=file_hash('configs/ai_reference_execution.json'),
                        runner_sha256=file_hash(__file__))
            save(destination,stored);del wave;gc.collect()
        if record.get('annotation_control'):
            truth=dict(truth,generator_scene_id=truth['scene_id'],scene_id=sid)
            save(output/'control_truth'/(sid+'.json'),truth)
            score_controls(stored,truth,class_rows,coverage)
        totals.append(stored)
        print('REFERENCE',name,sid,len(stored['annotation']['events']),stored['annotation']['annotation_status'],flush=True)
    summary=summarize_controls(class_rows,coverage,policy,args.calibration_only)
    summary_name=name+'_summary.json' if args.calibration_only else name+'_completion_and_heldout_audit.json'
    save(output/summary_name,dict(model_id=spec['model_id'],revision=spec['revision'],scenes=len(totals),
                    complete_schema_scenes=sum(x['annotation']['annotation_status']=='complete' for x in totals),
                    per_type_exact_control_metrics=summary,gpu_seconds=sum(x['generation']['gpu_seconds'] for x in totals),
                    gpu_seconds_scope='cumulative_recorded_model_generation_seconds_including_cached_records',run_wall_seconds=time.perf_counter()-run_started,
                    syntax_repairs=sum(x['generation']['syntax_repairs'] for x in totals),
                    annotation_origin='AI_reference_not_physical_ground_truth',calibration_only=args.calibration_only,
                    control_role='annotation_calibration' if args.calibration_only else 'reference_error_audit',
                    control_seeds=[r['seed'] for r in records if r.get('annotation_control')],
                    control_scene_ids=[r['scene_id'] for r in records if r.get('annotation_control')],
                    control_records_sha256=object_hash([r for r in records if r.get('annotation_control')]),
                    protocol_identity=identity,training_demonstrations_sha256=example_hash,
                    frozen_protocol_sha256=file_hash(args.frozen_protocol) if args.frozen_protocol else None))
    if batch_adapter:batch_adapter.close()
    del model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
