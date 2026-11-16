"""Freeze observed inputs and explicit operator tasks without running a language model."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from acoustic_regeneration.agents.benchmark_tools import BenchmarkTools, solve_rules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/gpu_agent_benchmark/inputs'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol = json.loads((ROOT/'preregistration/gpu_agent_benchmark.json').read_text())
    if (OUT/'manifest.json').exists():
        raise RuntimeError('Inputs already frozen; do not overwrite')
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(ROOT/'manifests/pilot_sources.jsonl.gz','rt') as f:
        sources = list(map(json.loads,f))
    examples = json.loads((ROOT/'figures/acoustic_example_selection.json').read_text())
    # Find source identifiers without relying on display names or acoustic results.
    excluded_text = json.dumps(examples)
    sources.sort(key=lambda r: hashlib.sha256((protocol['selection_seed']+'|'+r['original_file_id']).encode()).hexdigest())
    selected, groups, dates = [], set(), set()
    for row in sources:
        if row['partition']!='V' or row['original_file_id'] in excluded_text or row['group_id'] in groups or row['date'] in dates:
            continue
        selected.append(row); groups.add(row['group_id']); dates.add(row['date'])
        if len(selected)==protocol['evaluation_cases']:
            break
    assert len(selected)==32
    smoke = [r for r in sources if r['partition']=='G'][:2]
    manifest=[]
    for i, row in enumerate(smoke+selected):
        is_smoke=i<2
        index=i if is_smoke else i-2
        case_id=('smoke_' if is_smoke else 'case_')+f'{index+1:02d}'
        family=protocol['families'][index%4]
        source=ROOT/row['processed_path']
        assert sha(source)==row['processed_sha256']
        raw=np.load(source,allow_pickle=False)[:32768].astype(np.float64)
        observed=np.ones(32768,dtype=bool)
        observed[14000:18000]=False
        if family=='fragmented_observation':
            period=(64 if (index//4)%2==0 else 160)*8
            for start in range(0,len(observed),period): observed[start:start+128]=False
        mean=float(raw[observed].mean())
        scale=max(float(np.sqrt(np.mean((raw[observed]-mean)**2))),1e-8)
        audio=np.where(observed,(raw-mean)/scale,0.)
        constraints={'frequency_detail':(8,160),'timing_detail':(40,64),
                     'incompatible_constraints':(6,128),'fragmented_observation':(16,128)}[family]
        task={'case_id':case_id,'constraints':{'max_bin_spacing_hz':constraints[0],'max_time_support_ms':constraints[1]},
              'request':f'Inspect this recording and label the observed low-frequency evidence before and after its missing interval. Choose the shortest supported display window with bin spacing <= {constraints[0]} Hz and time support <= {constraints[1]} ms, with at least three wholly observed frames in EACH region. Apply and verify that setting; if none qualifies, decline without changing the display. Report the observed-cue label and measurements using the stated policy.'}
        np.savez_compressed(OUT/(case_id+'.npz'),audio=audio,observed=observed)
        (OUT/(case_id+'.json')).write_text(json.dumps(task,indent=2)+'\n')
        tools=BenchmarkTools(audio,observed,task)
        expected=solve_rules(tools)
        item={'case_id':case_id,'family':family,'split':'smoke' if is_smoke else 'evaluation',
              'source_partition':row['partition'],'source_id':row['original_file_id'],
              'group_id':row['group_id'],'date':row['date'],'source_array':row['processed_path'],
              'source_array_sha256':row['processed_sha256'],'normalization_mean':mean,'normalization_scale':scale,
              'input_sha256':sha(OUT/(case_id+'.npz')),'task_sha256':sha(OUT/(case_id+'.json')),
              'expected_report':expected,'reference_tool_trace':tools.events}
        manifest.append(item)
    document={'selection_protocol_sha256':sha(ROOT/'preregistration/gpu_agent_benchmark.json'),
              'model_inputs':'Only per-case task JSON, observed NPZ measurements, shared schemas and policy; manifest and expected reports are evaluator-only',
              'cases':manifest}
    (OUT/'manifest.json').write_text(json.dumps(document,indent=2)+'\n')
    from collections import Counter
    print('Frozen',len(selected),'evaluation inputs and',len(smoke),'transport inputs')
    print('Reference decisions',Counter(r['expected_report']['decision'] for r in manifest if r['split']=='evaluation'))
    print('Reference cue labels',Counter(r['expected_report']['cue_label'] for r in manifest if r['split']=='evaluation'))


if __name__=='__main__': main()
