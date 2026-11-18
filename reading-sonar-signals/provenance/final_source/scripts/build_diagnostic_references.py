#!/usr/bin/env python3
"""Export failed-checkpoint annotations for exploratory agreement only."""
import hashlib
import json
from pathlib import Path
from sonar.data import read_manifest
from sonar.reference_protocol import protocol_identity,verify_frozen,verify_record,select_controls


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path('results/ai_annotations');freeze_path=Path('configs/ai_reference_freeze.json');freeze=json.loads(freeze_path.read_text())
    policy=json.loads(Path('configs/ai_reference_eligibility.json').read_text())
    records=read_manifest('data/manifests/real_scenes.jsonl')+read_manifest('data/manifests/mixture_scenes.jsonl')+select_controls(read_manifest('data/manifests/annotation_controls.jsonl'),policy,'audit')
    results={};status={}
    for name,config_path in [('internvl4b','configs/ai_annotations.json'),('internvl8b','configs/ai_annotations_internvl8b_pilot.json')]:
        identity=protocol_identity(config_path);verify_frozen(freeze,identity);rows=[]
        for source in records:
            path=root/name/(source['scene_id']+'.json');raw=json.loads(path.read_text());verify_record(raw,identity,source)
            if raw['frozen_protocol_sha256']!=digest(freeze_path):raise RuntimeError('Diagnostic source uses another freeze')
            model=raw['generation'];file_sha=digest(path)
            rows.append(dict(scene_id=source['scene_id'],events=raw['annotation']['events'],
                reference_status='partial',reference_kind='ai_unqualified_checkpoint_diagnostic',
                qualification_status='failed_frozen_calibration_gate',primary_eligible_types=[],not_for_primary=True,
                observed_view_scope=freeze['observed_view_scope'],eligible_carrier_bands_hz=freeze['observed_view_scope']['carrier_bands_hz'],
                carrier_scope_interpretation='observed carrier bands only; this field does not certify annotation quality',
                scene_interval_s=raw['scene_interval_s'],raw_annotation_schema_status=raw['annotation']['annotation_status'],
                invalid_reference_rows=raw['annotation']['invalid_events'],
                source_annotation=dict(path=str(path),sha256=file_sha,model_id=model['model_id'],revision=model['model_revision']),
                frozen_protocol_sha256=digest(freeze_path),eligibility_protocol_sha256=freeze['eligibility_protocol_sha256'],
                independently_exhaustive=False,independent_ground_truth=False,annotation_control=bool(source.get('annotation_control')),
                interpretation='Exploratory event-set agreement with a failed AI reference checkpoint; unmatched events are disagreement, not physical errors. Empty outputs do not establish negative regions.'))
        results[name]=rows
    for name,rows in results.items():
        destination=root/('diagnostic_'+name+'.jsonl');destination.write_text(''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows))
        status[name]=dict(rows=len(rows),path=str(destination),sha256=digest(destination),all_reference_statuses_partial=all(r['reference_status']=='partial' for r in rows))
    (root/'diagnostic_reference_status.json').write_text(json.dumps(dict(models=status,primary_promotion_allowed=False,independent_ground_truth=False),indent=2)+'\n')
    print(json.dumps(status,indent=2),flush=True)


if __name__=='__main__':main()
