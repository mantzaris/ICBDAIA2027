"""Administrative sampling and parent-disjoint splits; no test-model outcomes."""
import hashlib
import json
from pathlib import Path
from sonar.data import read_manifest, opaque_id


def save(name,rows):
    path=Path('data/manifests')/name
    path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scenes':len(rows)}


real=read_manifest('data/manifests/real_scenes.jsonl')
train=sorted((r for r in real if r['split']=='train'),key=lambda r:r['scene_id'])
# Two development real scenes from different groups; no test scene used for pilot.
pilot=[]; groups=set()
for row in train:
    if row['group_id'] not in groups:
        pilot.append(row); groups.add(row['group_id'])
    if len(pilot)==2: break
for seed,kind in [(9001,'am_noise'),(9002,'mixed')]:
    pilot.append({'scene_id':opaque_id('syn',f'pilot:{seed}'),'stratum':'synthetic','seed':seed,
                  'split':'pilot','kind':kind,'sample_rate':16000,'duration_seconds':30,'snr_db':10,
                  'group_id':f'pilot:{seed}'})
inventory={'pilot':save('pilot.jsonl',pilot)}
# Balanced mechanism conditions with a 75% real majority in the dense benchmark:
# 210 untouched real +35 declared real mixtures +35 synthetic =280 unique scenes.
kinds=['tones','chirp','harmonics','missing_fundamental','crossing','occlusion',
       'transient','am_noise','beating','mixed','noise','silence']
synthetic=[]
for split,count,offset in [('train',17,1000),('validation',6,2000),('test',12,3000)]:
    for index in range(count):
        seed=offset+index
        synthetic.append({'scene_id':opaque_id('syn',f'{split}:{seed}'),'stratum':'synthetic',
                          'seed':seed,'split':split,'kind':kinds[index%len(kinds)],
                          'sample_rate':16000,'duration_seconds':30,'snr_db':10,
                          'group_id':f'synthetic:{split}:{seed}'})
inventory['synthetic']=save('synthetic_scenes.jsonl',synthetic)
mixtures=[]
factorial=[(fraction,ratio) for fraction in [0.,.25,.5,.75,1.] for ratio in [-10,0,10]]
# Twelve test mixtures cover every overlap fraction and each level four times.
# This incomplete factorial is a sensitivity grid, not replicated factor inference.
test_cells=[cell for i,cell in enumerate(factorial) if i not in [0,7,14]]
for split,count in [('train',17),('validation',6),('test',12)]:
    parents=sorted((r for r in real if r['split']==split and r['domain']=='deepship'),key=lambda r:r['scene_id'])
    for index in range(count):
        first=parents[index%len(parents)]
        second=next(r for r in parents[(index+1)%len(parents):]+parents[:(index+1)%len(parents)]
                    if r['group_id']!=first['group_id'])
        fraction,ratio=(test_cells[index] if split=='test' else factorial[(index*7)%15])
        offset=30*(1-fraction)
        mixtures.append({'scene_id':opaque_id('mix',f'v2:{split}:{index}:{fraction}:{ratio}'),'stratum':'semi_synthetic',
                         'split':split,'parents':[first,second],'sample_rate':16000,
                         'duration_seconds':30+offset,'offset_seconds':offset,'ratio_db':ratio,
                         'nominal_overlap_fraction':fraction,
                         'overlap_definition':'intersection duration divided by either 30-second parent duration',
                         'ratio_definition':'first-to-second full-excerpt RMS power ratio before shared peak scaling',
                         'group_id':'mixture:'+':'.join(sorted([first['group_id'],second['group_id']]))})
inventory['mixtures']=save('mixture_scenes.jsonl',mixtures)
inventory['development']=save('development.jsonl',[r for r in real+synthetic+mixtures if r['split']=='validation'])
inventory['main_test']=save('main_test.jsonl',[r for r in real+synthetic+mixtures if r['split']=='test'])
Path('reports/manifest_freeze.json').write_text(json.dumps(inventory,indent=2)+'\n')
print(json.dumps(inventory,indent=2))
