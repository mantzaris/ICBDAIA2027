#!/usr/bin/env python3
"""Freeze class eligibility from calibration, then assemble scoped AI references."""
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
from sonar.reference_protocol import object_hash,protocol_identity,verify_frozen,verify_record,select_controls
from sonar.data import read_manifest


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','assemble'])
    ap.add_argument('--calibration',default='results/ai_reference_calibration')
    ap.add_argument('--annotations',default='results/ai_annotations')
    ap.add_argument('--freeze',default='configs/ai_reference_freeze.json')
    args=ap.parse_args();policy=json.loads(Path('configs/ai_reference_eligibility.json').read_text())
    configs={'internvl4b':'configs/ai_annotations.json','internvl8b':'configs/ai_annotations_internvl8b_pilot.json'}
    identities={name:protocol_identity(path) for name,path in configs.items()}
    controls=read_manifest('data/manifests/annotation_controls.jsonl')
    calibration=select_controls(controls,policy,'calibration');audit=select_controls(controls,policy,'audit')
    if args.stage=='freeze':
        summaries={name:json.loads((Path(args.calibration)/(name+'_summary.json')).read_text()) for name in ['internvl4b','internvl8b']}
        for name,summary in summaries.items():
            if not summary['calibration_only'] or summary['scenes']!=12:raise RuntimeError('Require exactly12 calibration scenes for each checkpoint')
            if summary.get('protocol_identity')!=identities[name]:raise RuntimeError('Calibration protocol is stale')
            if summary.get('control_seeds')!=policy['calibration_seeds'] or summary.get('control_records_sha256')!=object_hash(calibration):raise RuntimeError('Calibration controls mismatch')
            spec=json.loads(Path(configs[name]).read_text())['models']['internvl']
            if summary['model_id']!=spec['model_id'] or summary['revision']!=spec['revision']:raise RuntimeError('Calibration checkpoint mismatch')
            for record in calibration:
                stored=json.loads((Path(args.calibration)/name/(record['scene_id']+'.json')).read_text())
                verify_record(stored,identities[name],record)
        selected={}
        for typ in policy['event_types']:
            candidates=[name for name,s in summaries.items() if s['per_type_exact_control_metrics'][typ]['eligible']]
            if candidates:
                name=max(candidates,key=lambda n:(summaries[n]['per_type_exact_control_metrics'][typ]['f1'],n=='internvl4b'))
                s=summaries[name]
                selected[typ]=dict(checkpoint_key=name,model_id=s['model_id'],revision=s['revision'],calibration_metrics=s['per_type_exact_control_metrics'][typ])
        value=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),reference_kind='ai_selected_checkpoint_agreement_proxy',
                   eligible_types=list(selected),eligible_frequency_domains=sorted({'modulation' if t=='modulation' else 'acoustic' for t in selected}),
                   selected_models_by_type=selected,all_calibration_summaries=summaries,
                   eligible_carrier_bands_hz=[[500,2000],[2000,5000]] if 'modulation' in selected else [],
                   observed_view_scope=dict(acoustic_frequency_hz=[0,1800],acoustic_windows_s=[1,4],modulation_frequency_hz=[0,200],modulation_window_s=4,carrier_bands_hz=[[500,2000],[2000,5000]],time_scope='full supplied scene'),
                   eligibility_protocol_sha256=digest('configs/ai_reference_eligibility.json'),
                   source_hashes=identities['internvl4b']['source_hashes'],
                   expected_input_manifest_sha256={p:digest(p) for p in ['data/manifests/real_scenes.jsonl','data/manifests/mixture_scenes.jsonl','data/manifests/annotation_controls.jsonl']},
                   model_dependence='InternVL4B and8B share multimodal-model ancestry and Qwen3 text-decoder family',
                   independently_verified_ground_truth=False,exhaustive_negative_regions_verified=False,
                   held_out_audit_status='not used for selection; run once after this freeze',
                   no_eligible_scope=not bool(selected),
                   interpretation='Agreement with admitted AI reference classes, not proof of real physical correctness')
        if Path(args.freeze).exists():raise RuntimeError('Refusing to overwrite a frozen reference protocol')
        save(args.freeze,value);print(json.dumps(value,indent=2));return
    freeze=json.loads(Path(args.freeze).read_text());root=Path(args.annotations)
    for identity in identities.values():verify_frozen(freeze,identity)
    expected=[]
    for path in ['data/manifests/real_scenes.jsonl','data/manifests/mixture_scenes.jsonl']:
        if digest(path)!=freeze['expected_input_manifest_sha256'][path]:raise RuntimeError('Frozen input manifest changed')
        expected.extend(read_manifest(path))
    expected.extend(audit)
    if len({r['scene_id'] for r in expected})!=len(expected):raise RuntimeError('Duplicate expected scene IDs')
    rows=[]
    for source in expected:
        sid=source['scene_id']
        available={name:json.loads((root/name/(sid+'.json')).read_text()) for name in ['internvl4b','internvl8b'] if (root/name/(sid+'.json')).exists()}
        for name,record in available.items():
            verify_record(record,identities[name],source)
            if record.get('frozen_protocol_sha256')!=digest(args.freeze):raise RuntimeError('Annotation freeze identity mismatch')
        events=[];missing=[];failed_views=[];provenance={}
        for typ,selection in freeze['selected_models_by_type'].items():
            name=selection['checkpoint_key'];record=available.get(name)
            if record is None:missing.append(name);continue
            if record['generation']['model_revision']!=selection['revision'] or record['generation']['model_id']!=selection['model_id']:raise RuntimeError('Selected model identity mismatch')
            annotation_path=root/name/(sid+'.json');file_digest=digest(annotation_path)
            provenance[name]=dict(path=str(annotation_path),sha256=file_digest,model_id=selection['model_id'],revision=selection['revision'])
            for event in record['annotation']['events']:
                if event['type']==typ:
                    events.append(dict(event,reference_model_id=selection['model_id'],reference_model_revision=selection['revision'],
                                       source_annotation_sha256=file_digest,calibration_frequency_tolerance_hz=4.))
            relevant='demon' if typ=='modulation' else 'lofar'
            failed_views.extend(p['view'] for p in record['generation']['passes'] if p['view'].startswith(relevant) and p['errors'])
        start=float(source.get('source_start_seconds',0))
        interval=next(iter(available.values()))['scene_interval_s'] if available else [start,start+float(source.get('duration_seconds',30))]
        status='unavailable' if not freeze['eligible_types'] or missing else ('partial' if failed_views else 'complete')
        rows.append(dict(scene_id=sid,events=events,eligible_types=freeze['eligible_types'],
                         eligible_frequency_domains=freeze['eligible_frequency_domains'],reference_status=status,
                         eligible_carrier_bands_hz=freeze['eligible_carrier_bands_hz'],observed_view_scope=freeze['observed_view_scope'],
                         reference_kind=freeze['reference_kind'],selected_models_by_type=freeze['selected_models_by_type'],
                         eligibility_protocol_sha256=freeze['eligibility_protocol_sha256'],source_annotations=provenance,
                         scene_interval_s=interval,unavailable_checkpoints=sorted(set(missing)),
                         failed_eligible_views=sorted(set(failed_views)),independently_exhaustive=False,
                         independent_ground_truth=False,annotation_origin='measurement_assisted_AI_reference',
                         input_record_sha256=object_hash(source),frozen_protocol_sha256=digest(args.freeze),
                         annotation_control=bool(source.get('annotation_control')),control_role=source.get('control_role')))
    root.mkdir(parents=True,exist_ok=True)
    (root/'primary_reference.jsonl').write_text(''.join(json.dumps(x,sort_keys=True,allow_nan=False)+'\n' for x in rows))
    save(root/'primary_reference_status.json',dict(scenes=len(rows),eligible_types=freeze['eligible_types'],
             complete=sum(x['reference_status']=='complete' for x in rows),partial=sum(x['reference_status']=='partial' for x in rows),
             unavailable=sum(x['reference_status']=='unavailable' for x in rows),
             independent_ground_truth=False,manifest_sha256=digest(root/'primary_reference.jsonl')))
    print('ASSEMBLED',len(rows),'scoped AI references; eligible types:',freeze['eligible_types'])


if __name__=='__main__':main()
