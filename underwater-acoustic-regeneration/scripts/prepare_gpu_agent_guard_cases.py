"""Freeze group/date-disjoint inputs for the configuration-gate follow-up."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from acoustic_regeneration.agents.benchmark_tools import BenchmarkTools, solve_rules

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/gpu_agent_guard/inputs'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    follow=json.loads((ROOT/'preregistration/gpu_agent_guard_followup.json').read_text())
    primary=json.loads((ROOT/'preregistration/gpu_agent_benchmark.json').read_text())
    used=json.loads((ROOT/'results/gpu_agent_benchmark/inputs/manifest.json').read_text())['cases']
    groups={r['group_id'] for r in used};dates={r['date'] for r in used}
    exclude=(ROOT/'figures/acoustic_example_selection.json').read_text()
    with gzip.open(ROOT/'manifests/pilot_sources.jsonl.gz','rt') as f:sources=list(map(json.loads,f))
    sources.sort(key=lambda r:hashlib.sha256((follow['selection_seed']+'|'+r['original_file_id']).encode()).hexdigest())
    selected=[]
    for row in sources:
        if row['partition']!='V' or row['group_id'] in groups or row['date'] in dates or row['original_file_id'] in exclude:continue
        selected.append(row);groups.add(row['group_id']);dates.add(row['date'])
        if len(selected)==follow['cases']:break
    assert len(selected)==16 and not (OUT/'manifest.json').exists()
    OUT.mkdir(parents=True,exist_ok=True);cases=[]
    for index,row in enumerate(selected):
        cid=f'guard_{index+1:02d}';family=primary['families'][index%4]
        source=ROOT/row['processed_path'];assert sha(source)==row['processed_sha256']
        raw=np.load(source,allow_pickle=False)[:32768].astype(np.float64)
        observed=np.ones(32768,dtype=bool);observed[14000:18000]=False
        if family=='fragmented_observation':
            period=(64 if (index//4)%2==0 else 160)*8
            for start in range(0,len(observed),period):observed[start:start+128]=False
        mean=float(raw[observed].mean());scale=max(float(np.sqrt(np.mean((raw[observed]-mean)**2))),1e-8)
        audio=np.where(observed,(raw-mean)/scale,0.)
        limits={'frequency_detail':(8,160),'timing_detail':(40,64),'incompatible_constraints':(6,128),'fragmented_observation':(16,128)}[family]
        task={'case_id':cid,'constraints':{'max_bin_spacing_hz':limits[0],'max_time_support_ms':limits[1]},
              'request':f'Inspect this recording and label the observed low-frequency evidence before and after its missing interval. Choose the shortest supported display window with bin spacing <= {limits[0]} Hz and time support <= {limits[1]} ms, with at least three wholly observed frames in EACH region. Apply and verify that setting; if none qualifies, decline without changing the display. Report the observed-cue label and measurements using the stated policy.'}
        np.savez_compressed(OUT/(cid+'.npz'),audio=audio,observed=observed)
        (OUT/(cid+'.json')).write_text(json.dumps(task,indent=2)+'\n')
        tools=BenchmarkTools(audio,observed,task);expected=solve_rules(tools)
        cases.append({'case_id':cid,'family':family,'source_partition':'V','source_id':row['original_file_id'],
                      'group_id':row['group_id'],'date':row['date'],'source_array':row['processed_path'],
                      'source_array_sha256':row['processed_sha256'],'normalization_mean':mean,'normalization_scale':scale,
                      'input_sha256':sha(OUT/(cid+'.npz')),'task_sha256':sha(OUT/(cid+'.json')),
                      'expected_report':expected,'reference_tool_trace':tools.events})
    (OUT/'manifest.json').write_text(json.dumps({'protocol_sha256':sha(ROOT/'preregistration/gpu_agent_guard_followup.json'),
        'primary_groups_and_dates_excluded':True,'cases':cases},indent=2)+'\n')
    print('Frozen 16 new groups/dates for the guarded-versus-unguarded reasoning comparison.')


if __name__=='__main__':main()
