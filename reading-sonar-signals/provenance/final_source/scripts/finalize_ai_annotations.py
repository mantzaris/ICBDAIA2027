#!/usr/bin/env python3
"""Write completion marker only after both frozen reference passes are present."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
from sonar.data import read_manifest
from sonar.reference_protocol import object_hash,protocol_identity,select_controls,verify_frozen,verify_record


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    root=Path('results/ai_annotations');marker=root/'COMPLETED.json'
    if marker.exists():raise RuntimeError('Completion marker already exists')
    freeze_path=Path('configs/ai_reference_freeze.json');freeze=json.loads(freeze_path.read_text());fhash=digest(freeze_path)
    policy=json.loads(Path('configs/ai_reference_eligibility.json').read_text())
    execution=json.loads(Path('configs/ai_reference_execution.json').read_text())
    audit=select_controls(read_manifest('data/manifests/annotation_controls.jsonl'),policy,'audit')
    real=read_manifest('data/manifests/real_scenes.jsonl');mixed=read_manifest('data/manifests/mixture_scenes.jsonl');records=audit+real+mixed
    ids={r['scene_id'] for r in records}
    if len(ids)!=257 or len(records)!=257:raise RuntimeError('Expected frozen257 scenes')
    for path,sha in freeze['expected_input_manifest_sha256'].items():
        if digest(path)!=sha:raise RuntimeError('Frozen manifest changed: '+path)
    inventory=[];images={};model_status={}
    for name,config_path in [('internvl4b','configs/ai_annotations.json'),('internvl8b','configs/ai_annotations_internvl8b_pilot.json')]:
        identity=protocol_identity(config_path);verify_frozen(freeze,identity)
        spec=json.loads(Path(config_path).read_text())['models']['internvl']
        calibration=freeze['all_calibration_summaries'][name]
        statuses={};event_count=0
        for source in records:
            p=root/name/(source['scene_id']+'.json');row=json.loads(p.read_text());verify_record(row,identity,source)
            if row['frozen_protocol_sha256']!=fhash or row['batch_size']!=execution['batch_size']:raise RuntimeError('Frozen execution mismatch')
            if row['generation']['model_id']!=spec['model_id'] or row['generation']['model_revision']!=spec['revision']:raise RuntimeError('Checkpoint identity mismatch')
            if row['training_demonstrations_sha256']!=calibration['training_demonstrations_sha256']:raise RuntimeError('Training demonstrations changed')
            if [x['view'] for x in row['generation']['passes']]!=['lofar_1s','lofar_4s','demon_500_2000','demon_2000_5000']:raise RuntimeError('Incomplete fixed-view generation')
            for view in row['generation']['passes']:
                for image in [dict(image=view['image'],image_sha256=view['image_sha256'])]+view['generation']['training_examples']:
                    path=image['image'];expected=image['image_sha256']
                    if path in images and images[path]!=expected:raise RuntimeError('Overwritten image differs between retained generations')
                    images[path]=expected
            status=row['annotation']['annotation_status'];statuses[status]=statuses.get(status,0)+1;event_count+=len(row['annotation']['events'])
            inventory.append(dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size))
        p=root/(name+'_completion_and_heldout_audit.json');summary=json.loads(p.read_text())
        if summary['calibration_only'] or summary['control_seeds']!=policy['held_out_audit_seeds'] or summary['control_records_sha256']!=object_hash(audit) or summary['scenes']!=257:raise RuntimeError('Missing complete heldout audit')
        if summary['frozen_protocol_sha256']!=fhash or summary['protocol_identity']!=identity:raise RuntimeError('Audit protocol mismatch')
        if any('eligible' in m for m in summary['per_type_exact_control_metrics'].values()):raise RuntimeError('Audit must not reselect eligibility')
        model_status[name]=dict(model_id=spec['model_id'],revision=spec['revision'],raw_records=257,schema_status_counts=statuses,event_count=event_count,heldout_summary_sha256=digest(p))
    for path,sha in images.items():
        if digest(path)!=sha:raise RuntimeError('Image evidence checksum mismatch: '+path)
    primary=read_manifest(root/'primary_reference.jsonl');consensus=read_manifest(root/'secondary_consensus.jsonl')
    if len(primary)!=257 or {r['scene_id'] for r in primary}!=ids or len(consensus)!=257 or {r['scene_id'] for r in consensus}!=ids:raise RuntimeError('Final assembled references incomplete')
    if any(r.get('eligible_types')!=freeze['eligible_types'] or r.get('frozen_protocol_sha256')!=fhash for r in primary):raise RuntimeError('Primary scope differs from frozen selection')
    agreement=json.loads((root/'checkpoint_agreement.json').read_text())
    if agreement['available_pairs']!=257 or agreement['real_and_mixed_scene_count']!=245:raise RuntimeError('Checkpoint disagreement report incomplete')
    diagnostics={}
    for name in model_status:
        path=root/('diagnostic_'+name+'.jsonl');rows=read_manifest(path)
        if len(rows)!=257 or {r['scene_id'] for r in rows}!=ids or any(r['reference_status']!='partial' or r['reference_kind']!='ai_unqualified_checkpoint_diagnostic' for r in rows):raise RuntimeError('Unqualified diagnostic export incomplete')
        diagnostics[name]=digest(path)
    (root/'raw_record_inventory.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in inventory))
    save(root/'image_inventory.json',images)
    value=dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),raw_annotation_records=514,final_scene_count=257,real_scenes=210,real_mixture_scenes=35,heldout_audit_controls=12,
        separate_calibration_controls=12,models=model_status,unique_verified_image_files=len(images),
        frozen_protocol_sha256=fhash,eligibility_protocol_sha256=freeze['eligibility_protocol_sha256'],execution_config_sha256=digest('configs/ai_reference_execution.json'),
        primary_reference_sha256=digest(root/'primary_reference.jsonl'),secondary_consensus_sha256=digest(root/'secondary_consensus.jsonl'),
        checkpoint_agreement_sha256=digest(root/'checkpoint_agreement.json'),raw_inventory_sha256=digest(root/'raw_record_inventory.jsonl'),image_inventory_sha256=digest(root/'image_inventory.json'),
        unqualified_diagnostic_reference_sha256=diagnostics,
        eligible_types=freeze['eligible_types'],independent_ground_truth=False,interpretation='Completed frozen AI annotation attempts and audits; not a claim that machine labels are physically correct')
    marker_text=json.dumps(value,indent=2,allow_nan=False)+'\n'
    marker_sha256=hashlib.sha256(marker_text.encode()).hexdigest()
    status_path=Path('annotations/status.json');status=json.loads(status_path.read_text());status.update(final_raw_annotation_status='completed',completed_ai_annotations=514,heldout_reference_audit_status='completed under frozen protocol',reconciliation_status='checkpoint agreement and disagreements retained',ai_reference_agreement_evaluation_ready=bool(freeze['eligible_types']),completion_marker_sha256=marker_sha256);save(status_path,status)
    marker.write_text(marker_text)
    print(json.dumps(value,indent=2),flush=True)


if __name__=='__main__':main()
