"""Predeclare robustness/budget subsets before examining main model outcomes."""
import hashlib
import json
from pathlib import Path
from sonar.data import read_manifest, opaque_id

root=Path('data/manifests')
main=read_manifest(root/'main_test.jsonl')
real=sorted((r for r in main if r['stratum']=='real'),key=lambda r:r['scene_id'])
synthetic=sorted((r for r in main if r['stratum']=='synthetic'),key=lambda r:r['scene_id'])
# All30 internal+50 external scenes, chosen by opaque ID, never observed outcomes.
internal=[r for r in real if r['domain']=='deepship']
external=[r for r in real if r['domain']=='mbari']
ablation=internal+external[:50]
budgets=internal[:3]+external[:3]+synthetic
robust=[]
selected=[r for r in synthetic if r['kind'] in ['tones','chirp','am_noise','beating','mixed','transient']]
for row in selected:
    conditions=[{'name':'gain','linear_gain':.5}]
    conditions += [{'name':'added_noise','nominal_ratio_db':v,'seed':99100+i} for i,v in enumerate([-5,0,5,10])]
    conditions += [{'name':'bandwidth_loss','sample_rate':4000},
                   {'name':'remove_component','component_id':'s0'},{'name':'blank'},
                   {'name':'plot_style','colormap':'viridis'}]
    for condition in conditions:
        sid=opaque_id('int',row['scene_id']+json.dumps(condition,sort_keys=True))
        robust.append({'scene_id':sid,'base_record':row,'stratum':'synthetic_intervention',
                       'split':'test','group_id':row['group_id'],'intervention':condition,
                       'base_scene_id':row['scene_id'],'sample_rate':row['sample_rate'],
                       'duration_seconds':row['duration_seconds']})
inventory={}
sensitivity=[]
def add_probe(base,probe,stratum):
    sid=opaque_id('probe',base['scene_id']+json.dumps(probe,sort_keys=True))
    sensitivity.append(dict(scene_id=sid,sensitivity_base=base,sensitivity=probe,stratum=stratum,
        split='test',group_id=base['group_id'],base_scene_id=base['scene_id'],sample_rate=base['sample_rate'],
        duration_seconds=probe.get('duration_seconds',base['duration_seconds'])))
for row in selected:
    add_probe(row,dict(name='duration',duration_seconds=10),'synthetic_intervention')
    donor=next(other for other in selected if other['scene_id']!=row['scene_id'] and other['kind'] in ['beating','am_noise'])
    add_probe(row,dict(name='display_mismatch',donor_record=donor),'synthetic_intervention')
for row in internal[:3]+external[:3]:
    for name,band in [('spectral_attenuation',[200.,400.]),('sham_attenuation',[1000.,1200.])]:
        add_probe(row,dict(name=name,band_hz=band,scene_local_time_s=[10.,20.],linear_gain=.1),'real_intervention')
for row in sorted((r for r in main if r['stratum']=='semi_synthetic'),key=lambda r:r['scene_id'])[:6]:
    add_probe(row,dict(name='remove_real_parent',removed_parent=1),'mixture_intervention')
for name,rows in [('ablation_test.jsonl',ablation),('budget_test.jsonl',budgets),('robustness_test.jsonl',robust),('sensitivity_test.jsonl',sensitivity)]:
    path=root/name; path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    inventory[name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scenes':len(rows)}
Path('reports/mechanism_manifest_freeze.json').write_text(json.dumps(inventory,indent=2)+'\n')
print(json.dumps(inventory,indent=2))
