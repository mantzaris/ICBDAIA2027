#!/usr/bin/env python3
"""Report checkpoint agreement without claiming physical correctness or negatives."""
import argparse
import hashlib
import json
from pathlib import Path
from sonar.data import read_manifest
from sonar.metrics import event_metrics,aggregate_event_metrics
from sonar.reference_protocol import protocol_identity,verify_frozen,verify_record,select_controls


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='results/ai_annotations');ap.add_argument('--freeze',default='configs/ai_reference_freeze.json');args=ap.parse_args()
    root=Path(args.root);freeze=json.loads(Path(args.freeze).read_text());policy=json.loads(Path('configs/ai_reference_eligibility.json').read_text())
    identities={n:protocol_identity(c) for n,c in [('internvl4b','configs/ai_annotations.json'),('internvl8b','configs/ai_annotations_internvl8b_pilot.json')]}
    for identity in identities.values():verify_frozen(freeze,identity)
    records=read_manifest('data/manifests/real_scenes.jsonl')+read_manifest('data/manifests/mixture_scenes.jsonl')+select_controls(read_manifest('data/manifests/annotation_controls.jsonl'),policy,'audit')
    output=[];metrics=[];by_type={t:[] for t in policy['event_types']}
    schema_counts={n:{} for n in identities};invalid_counts={n:0 for n in identities}
    for source in records:
        sid=source['scene_id'];paths={n:root/n/(sid+'.json') for n in identities}
        if not all(p.exists() for p in paths.values()):
            output.append(dict(scene_id=sid,events=[],reference_status='unavailable',reference_kind='AI_checkpoint_geometric_agreement',independent_ground_truth=False,independently_exhaustive=False));continue
        labels={n:json.loads(p.read_text()) for n,p in paths.items()}
        for n,record in labels.items():
            verify_record(record,identities[n],source)
            if record['frozen_protocol_sha256']!=digest(args.freeze):raise RuntimeError('Reference freeze mismatch')
            status=record['annotation']['annotation_status'];schema_counts[n][status]=schema_counts[n].get(status,0)+1
            invalid_counts[n]+=len(record['annotation']['invalid_events'])
        a=labels['internvl4b']['annotation']['events'];b=labels['internvl8b']['annotation']['events']
        # Suppress self-reported bin relaxation; only bounded reference display
        # uncertainty may widen the4Hz floor, and this is explicitly AI agreement.
        pa=[dict(e,bin_spacing_hz=0.) for e in a]
        gb=[dict(e,frequency_uncertainty_hz=min(8.,e.get('frequency_uncertainty_hz',0))) for e in b]
        metric=event_metrics(pa,gb,frequency_tolerance_hz=4.,exhaustive=True)
        matched_a={m['prediction_index'] for m in metric['matches']};matched_b={m['gold_index'] for m in metric['matches']}
        events=[dict(a[m['prediction_index']],reference_geometry='4B_coordinates_corroborated_by_8B',corroborating_8b_event_id=b[m['gold_index']]['id'],reference_models=[labels[n]['generation']['model_id'] for n in labels],agreement_match=m) for m in metric['matches']]
        row=dict(scene_id=sid,events=events,reference_status='complete' if all(r['annotation']['annotation_status']=='complete' for r in labels.values()) else 'partial',
            reference_kind='AI_checkpoint_geometric_agreement',independent_ground_truth=False,independently_exhaustive=False,
            unmatched_4b_events=[e for i,e in enumerate(a) if i not in matched_a],unmatched_8b_events=[e for i,e in enumerate(b) if i not in matched_b],
            checkpoint_agreement_metric=metric,source_annotations={n:dict(path=str(p),sha256=digest(p)) for n,p in paths.items()},
            interpretation='Unmatched events are checkpoint disagreement, not physical false positives or negatives',annotation_control=bool(source.get('annotation_control')))
        output.append(row)
        if not source.get('annotation_control'):
            metrics.append(metric)
            for typ in by_type:by_type[typ].append(event_metrics([e for e in pa if e['type']==typ],[e for e in gb if e['type']==typ],frequency_tolerance_hz=4.,exhaustive=True))
    (root/'secondary_consensus.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in output))
    write(root/'checkpoint_agreement.json',dict(real_and_mixed_scene_count=len(metrics),overall=aggregate_event_metrics(metrics),per_type={t:aggregate_event_metrics(v) for t,v in by_type.items()},
        expected_scenes=len(records),available_pairs=sum(r['reference_status']!='unavailable' for r in output),
        metric_interpretation='Event-set agreement between related AI checkpoints; neither serves as physical truth',
        frequency_rule='4Hz floor;8B reference display uncertainty capped8Hz;4B prediction bins cannot relax matching',
        heldout_controls_excluded_from_aggregate=True,independent_ground_truth=False,
        agreement_scope='schema-valid retained event subsets only; malformed rows are reported separately, not silently interpreted as physical negatives',
        schema_status_counts_by_model=schema_counts,invalid_reference_row_counts_by_model=invalid_counts,
        consensus_sha256=digest(root/'secondary_consensus.jsonl')))
    print('AI reference comparison complete:',len(output),'scene records',flush=True)


if __name__=='__main__':main()
