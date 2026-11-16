"""Run the frozen rule/direct/reasoning/review comparison with CUDA-only LLM inference."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from acoustic_regeneration.agents.benchmark_tools import BenchmarkTools, schemas, solve_rules, score_report

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'results/gpu_agent_benchmark/inputs'
SYSTEM='''You are an acoustic analysis agent operating a virtual display. Use actual tool returns as evidence. The tools do not choose an answer for you. You may request multiple tools, but a request is not evidence until its return arrives. Dataset labels and erased audio are unavailable. No physical hardware is connected.
Policy: choose the SHORTEST available window satisfying BOTH user constraints and at least 3 fully observed frames in EACH before/after region. Inspect first. Measure any setting before applying it. If no window qualifies, decline and do not configure. If one qualifies, apply it and call verify_display AFTER applying it. Only entirely observed frames enter measurements.
Observed-cue label: insufficient if no setting qualifies, or either region at the selected setting has peak prominence <10 dB. Otherwise stable if the two peak bins differ by <= one selected-window frequency bin AND the absolute before/after band-level difference is <=6 dB; changed otherwise. These labels concern observed context only, not continuity inside missing audio or object classification. Identity must remain unknown.
Final output: <final_report>JSON</final_report> with exactly these keys: decision (configure or decline), window_ms (selected integer or null for decline), cue_label (stable, changed, insufficient), peak_before_hz and peak_after_hz (measured peak bins at selected window, or null for decline), identity (unknown), gap_unobserved (true), verified (true only after actual post-configuration verification; false for decline), evidence_ids (list of actual tool-return IDs supporting the reported values/decision). Do not invent values or evidence IDs. You may continue calling tools until ready to report. Keep external explanations short.'''
REVIEW='''Review the proposed answer against the task and ACTUAL executed tool returns. Check setting feasibility and shortest-window choice, observed-cue label, missing data, identity, numeric values, citations, and whether verification actually followed configuration. Correct any errors and perform missing actions using the available tools. This message supplies no measurements or expected answer. Return the same final_report JSON schema. If no answer was completed, finish the task. Review within the remaining shared budget.'''


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def parse_action(text):
    calls=[]
    for item in re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>',text,re.S):
        call=json.loads(item)
        if not isinstance(call,dict) or not isinstance(call.get('name'),str) or not isinstance(call.get('arguments'),dict):
            raise ValueError('Invalid tool-call object')
        calls.append({'type':'function','function':call})
    if calls: return calls,None
    match=re.search(r'<final_report>\s*(.*?)\s*</final_report>',text,re.S)
    candidate=match.group(1) if match else text.strip()
    if candidate.startswith('```'):
        candidate=re.sub(r'^```(?:json)?\s*|\s*```$','',candidate)
    report=json.loads(candidate)
    if not isinstance(report,dict): raise ValueError('Final report must be a JSON object')
    return [],report


class GPUModel:
    def __init__(self,protocol):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch=torch
        self.protocol=protocol
        model_path=ROOT/'.local/models/Qwen3-8B'
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,
                   torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa')
        self.model.eval()
        assert {str(p.device) for p in self.model.parameters()}=={'cuda:0'},'CPU offload forbidden'
        self.think_end=self.tokenizer.convert_tokens_to_ids('</think>')
        self.end=self.tokenizer.eos_token_id
        print('CUDA model loaded',torch.cuda.get_device_name(0),flush=True)

    def generate(self,messages,thinking,budget,seed):
        torch=self.torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        text=self.tokenizer.apply_chat_template(messages,tools=schemas(),tokenize=False,
              add_generation_prompt=True,enable_thinking=thinking)
        encoded=self.tokenizer(text,return_tensors='pt').to('cuda:0')
        if encoded.input_ids.shape[1]>self.protocol['budgets']['max_context_tokens']:
            raise RuntimeError('Context cap reached')
        sampling=self.protocol['sampling']
        kwargs=dict(do_sample=True,temperature=sampling['temperature'],top_p=sampling['top_p'],
                    top_k=sampling['top_k'],pad_token_id=self.tokenizer.pad_token_id or self.end,
                    use_cache=True)
        torch.cuda.synchronize(); start=time.perf_counter()
        generated=[]; forced_close=False; thought_text=''
        with torch.inference_mode():
            if thinking:
                limit=min(self.protocol['budgets']['max_thinking_tokens_per_turn'],max(1,budget-64))
                first=self.model.generate(**encoded,max_new_tokens=limit,eos_token_id=[self.think_end,self.end],**kwargs)
                thought=first[0,encoded.input_ids.shape[1]:].tolist()
                generated.extend(thought)
                thought_text=self.tokenizer.decode(thought,skip_special_tokens=False)
                if self.think_end not in thought:
                    forced_close=True
                    extra=self.tokenizer('</think>\n\n',add_special_tokens=False,return_tensors='pt').input_ids.to('cuda:0')
                    first=torch.cat([first,extra],dim=1)
                encoded={'input_ids':first,'attention_mask':torch.ones_like(first)}
            action_limit=min(self.protocol['budgets']['max_action_tokens_per_turn'],budget-len(generated))
            if action_limit<=0: raise RuntimeError('No action tokens remain')
            prefix=encoded['input_ids'].shape[1]
            outputs=self.model.generate(**encoded,max_new_tokens=action_limit,eos_token_id=self.end,**kwargs)
            action=outputs[0,prefix:].tolist()
            generated.extend(action)
        torch.cuda.synchronize()
        return {'text':self.tokenizer.decode(action,skip_special_tokens=True),
                'reasoning_text':thought_text,'reasoning_tokens':len(generated)-len(action),
                'generated_tokens':len(generated),'action_tokens':len(action),
                'forced_reasoning_close':forced_close,'action_hit_token_cap':len(action)==action_limit and action[-1]!=self.end,
                'prompt_tokens':len(self.tokenizer.encode(text)), 'elapsed_seconds':time.perf_counter()-start,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'model_device':'cuda:0','request_messages':messages}


def run_episode(model,case,arm,protocol,out):
    case_id=case['case_id']; task=json.loads((INPUT/(case_id+'.json')).read_text())
    with np.load(INPUT/(case_id+'.npz'),allow_pickle=False) as d: tools=BenchmarkTools(d['audio'],d['observed'],task)
    out.mkdir(parents=True,exist_ok=True)
    event_path=out/(case_id+'__'+arm+'.jsonl')
    summary_path=out/(case_id+'__'+arm+'.json')
    if summary_path.exists(): return json.loads(summary_path.read_text())
    if event_path.exists(): raise RuntimeError('Partial episode exists; retain and investigate instead of rerunning')
    start=time.perf_counter(); events=event_path.open('x')
    def event(kind,**kw):
        events.write(json.dumps({'utc':now(),'elapsed_seconds':time.perf_counter()-start,'kind':kind,**kw},allow_nan=False)+'\n'); events.flush()
    event('start',case_id=case_id,arm=arm,input_sha256=sha(INPUT/(case_id+'.npz')),task_sha256=sha(INPUT/(case_id+'.json')))
    messages=[{'role':'system','content':SYSTEM},{'role':'user','content':task['request']}]
    report=None; initial_report=None; reviewed=False; error=None; status='budget_exhausted'
    total_tokens=0; turns=0; total_inference=0.; forced_closures=0; drafts=[]
    review_arm=arm in ('self_review','two_agents')
    budget=protocol['budgets']
    try:
        if arm=='rules':
            report=solve_rules(tools); status='completed'
            for item in tools.events: event('tool_result',**item)
        else:
            for turn in range(budget['max_model_turns']):
                if time.perf_counter()-start>=budget['max_episode_seconds']: status='time_limit'; break
                remaining=budget['max_generated_tokens']-total_tokens
                if remaining<65: break
                if review_arm and not reviewed:
                    remaining=min(remaining,budget['max_generated_tokens']//2-total_tokens)
                    if remaining<65 or turns>=4:
                        messages=review_messages(arm,messages,task,tools,report)
                        reviewed=True; event('review_start',role='reviewer' if arm=='two_agents' else 'self_review')
                        remaining=budget['max_generated_tokens']-total_tokens
                result=model.generate(copy.deepcopy(messages),arm!='direct',remaining,
                                      protocol['sampling']['seed']+1000*int(case_id.split('_')[-1])+turn)
                turns+=1; total_tokens+=result['generated_tokens']; total_inference+=result['elapsed_seconds']
                forced_closures+=int(result['forced_reasoning_close'])
                event('model_response',turn=turn+1,agent='reviewer' if reviewed and arm=='two_agents' else 'analyst',**result)
                try:
                    calls, candidate=parse_action(result['text'])
                except (ValueError,TypeError,KeyError) as exc:
                    messages.extend([{'role':'assistant','content':result['text']},
                                     {'role':'user','content':'Output format error: '+str(exc)+'. Use native tool_call objects or final_report JSON. No task verdict has been supplied.'}])
                    event('format_error',message=str(exc)); continue
                messages.append({'role':'assistant','content':result['text'] if not calls else '',**({'tool_calls':calls} if calls else {})})
                if calls:
                    for call in calls:
                        if len(tools.events)>=budget['max_tool_calls']: raise RuntimeError('Tool-call cap reached')
                        item=tools.call(call['function']['name'],call['function']['arguments'])
                        event('tool_result',**item)
                        messages.append({'role':'tool','name':item['name'],'content':json.dumps(item,allow_nan=False)})
                else:
                    report=candidate; drafts.append(copy.deepcopy(report))
                    if review_arm and not reviewed:
                        initial_report=copy.deepcopy(report)
                        messages=review_messages(arm,messages,task,tools,report)
                        reviewed=True; event('review_start',role='reviewer' if arm=='two_agents' else 'self_review')
                        report=None
                    else:
                        status='completed'; break
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'; status='failed'; event('failure',error=error)
    # Scoring happens only AFTER all model interaction; expected answers never enter prompts.
    checks=score_report(report,tools,case['expected_report'])
    summary={'case_id':case_id,'arm':arm,'family':case['family'],'group_id':case['group_id'],'date':case['date'],
             'status':status,'error':error,'final_report':report,'initial_report':initial_report,'drafts':drafts,
             'checks':checks,'model_turns':turns,'tool_calls':len(tools.events),
             'tool_errors':sum('error' in e['result'] for e in tools.events),
             'generated_tokens':total_tokens,'inference_seconds':total_inference,'elapsed_seconds':time.perf_counter()-start,
             'forced_reasoning_closures':forced_closures,'review_started':reviewed,
             'input_unchanged':tools.input_hash()==tools.initial_hash,'input_sha256':sha(INPUT/(case_id+'.npz')),
             'tool_trace':tools.events}
    event('end',status=status,checks=checks)
    events.close(); summary['events_sha256']=sha(event_path); write(summary_path,summary)
    print(case_id,arm,status,'success',checks['strict_success'],'tokens',total_tokens,'seconds',round(summary['elapsed_seconds'],1),flush=True)
    return summary


def review_messages(arm,messages,task,tools,report):
    if arm=='self_review':
        return messages+[{'role':'user','content':REVIEW}]
    return [{'role':'system','content':SYSTEM+'\nYou are a separate reviewing agent. Treat the planner draft as an unverified proposal. '+REVIEW},
            {'role':'user','content':task['request']+'\nExecuted tool trace:\n'+json.dumps(tools.events,allow_nan=False)+'\nPlanner draft:\n'+json.dumps(report,allow_nan=False)}]


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',choices=['smoke','evaluation'],required=True)
    parser.add_argument('--limit',type=int); args=parser.parse_args()
    protocol=json.loads((ROOT/'preregistration/gpu_agent_benchmark.json').read_text())
    authorization=json.loads((ROOT/'configs/gpu_agent_authorization.json').read_text())
    manifest=json.loads((INPUT/'manifest.json').read_text())
    assert sha(ROOT/'preregistration/gpu_agent_benchmark.json')==manifest['selection_protocol_sha256']
    cases=[r for r in manifest['cases'] if r['split']==args.phase]
    if args.limit: cases=cases[:args.limit]
    for case in cases:
        assert sha(INPUT/(case['case_id']+'.npz'))==case['input_sha256']
        assert sha(INPUT/(case['case_id']+'.json'))==case['task_sha256']
    model=GPUModel(protocol)
    model.torch.cuda.reset_peak_memory_stats()
    out=ROOT/'results/gpu_agent_benchmark'/args.phase
    out.mkdir(parents=True,exist_ok=True)
    code_paths=['scripts/run_gpu_agent_benchmark.py','src/acoustic_regeneration/agents/benchmark_tools.py',
                'preregistration/gpu_agent_benchmark.json','results/gpu_agent_benchmark/inputs/manifest.json']
    code_hashes={p:sha(ROOT/p) for p in code_paths}
    run_record={'started_utc':now(),'phase':args.phase,'artifact_sha256':code_hashes,
                'model_environment_sha256':sha(ROOT/'provenance/gpu_agent_environment.json'),
                'authorization_sha256':sha(ROOT/'configs/gpu_agent_authorization.json')}
    run_path=out/'run_definition.json'
    if run_path.exists():
        assert json.loads(run_path.read_text())['artifact_sha256']==code_hashes,'Runner changed after start'
    else: write(run_path,run_record)
    deadline=datetime.fromisoformat(authorization['stop_jobs_by_utc'])
    summaries=[]
    for i,case in enumerate(cases):
        arms=protocol['arms'][1:]
        arms=arms[i%len(arms):]+arms[:i%len(arms)]
        for arm in ['rules',*arms]:
            if datetime.now(timezone.utc)>=deadline: raise RuntimeError('Authorized job deadline reached')
            summaries.append(run_episode(model,case,arm,protocol,out))
            write(out/'progress.json',{'updated_utc':now(),'completed_episodes':len(summaries),
                                      'planned_episodes':len(cases)*len(protocol['arms']),
                                      'last_case':case['case_id'],'last_arm':arm})
    write(out/'completion.json',{'completed_utc':now(),'cases':len(cases),'episodes':len(summaries),
          'neural_inference_device':'cuda:0','peak_allocated_bytes':model.torch.cuda.max_memory_allocated(),
          'protocol_sha256':sha(ROOT/'preregistration/gpu_agent_benchmark.json'),
          'model_environment_sha256':sha(ROOT/'provenance/gpu_agent_environment.json'),
          'runner_sha256':sha(Path(__file__)),'tools_sha256':sha(ROOT/'src/acoustic_regeneration/agents/benchmark_tools.py')})
    assert all(sha(ROOT/p)==expected for p,expected in code_hashes.items()),'Code mutated during execution'


if __name__=='__main__': main()
