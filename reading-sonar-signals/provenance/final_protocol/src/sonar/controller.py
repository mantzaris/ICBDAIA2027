"""Matched bounded inspection controllers with raw, auditable model outputs."""
from __future__ import annotations
import json
import math
import time
from pathlib import Path
import torch
from .rendering import render_view
from .batching import GPU_LOCK
from .constrained_decoding import action_json_schema, report_json_schema
from .prompting import compact_measurements, prompt_numbers, PROMPT_PROTOCOL


OBSERVATION_RULES = '''You analyze passive underwater acoustic displays. Acoustic Hz and envelope modulation Hz are different quantities. A DEMON peak is not evidence of blade count, RPM, a ship class, or a unique physical source. Interference can create envelope beating. Treat received levels as uncalibrated digital amplitude. Keep overlapping components and uncertain associations. Filenames and hidden labels are unavailable. Do not invent evidence, measurements, probabilities, or physical identities. Report directly measured observable morphology even when its source is unknown; uncertain source identity alone is not a reason to withhold a measured observation. Return one JSON object only, without markdown. Use only requested schema keys. Use at most three frequency_track points per component and do not copy the full measurement table into the report.'''

REPORT_SCHEMA = '''Return {"components":[{"id":"C1","type":"tone|chirp|transient|broadband|modulation","frequency_domain":"acoustic|modulation","time_s":[start,end],"frequency_hz":[low,high],"frequency_track":[[start,frequency_at_start],[end,frequency_at_end]],"carrier_band_hz":null,"source_identity":"unknown","evidence_ids":["E1"]}],"relations":[{"type":"overlaps_time|harmonic_candidate|unresolved_association","component_ids":["C1","C2"],"evidence_ids":["E1"]}],"claims":[{"text":"a concise observable finding","component_ids":["C1"],"evidence_ids":["E1"],"kind":"observation"}],"unresolved_questions":["Which source produced these components?"],"abstained":false}. Use at most 6 components and 3 brief claims. For chirps always supply at least two ordered frequency_track points, retaining rising versus falling direction. Evidence IDs must actually appear in the supplied measurement record; for image-only reports use empty evidence_ids. Intervals use the displayed original-recording seconds. Carrier bands are mandatory for modulation. Do not equate a component count with a physical-source count. An empty report has zero coverage, not perfect accuracy.'''


def report_schema_errors(report):
    """Host JSON/type validation, distinct from CUDA measurement verification."""
    if not isinstance(report, dict):
        return ['report_must_be_object']
    errors = []
    for field in ['components', 'relations', 'claims', 'unresolved_questions']:
        if not isinstance(report.get(field), list):
            errors.append(f'{field}_must_be_list')
    if type(report.get('abstained')) is not bool:
        errors.append('abstained_must_be_bool')
    if errors:
        return errors
    def number(value):
        return type(value) in (int, float) and math.isfinite(value)
    def pair(value, strict=False, nonnegative=False):
        return (isinstance(value, list) and len(value)==2 and all(number(x) for x in value)
                and (value[1]>value[0] if strict else value[1]>=value[0])
                and (not nonnegative or value[0]>=0))
    def strings(value):
        return isinstance(value, list) and all(isinstance(x,str) and x for x in value)
    seen = set()
    if len(report['components'])>6:
        errors.append('component_output_budget_exceeded')
    if len(report['claims'])>3:
        errors.append('claim_output_budget_exceeded')
    for index, component in enumerate(report['components']):
        prefix=f'components[{index}]'
        if not isinstance(component,dict):
            errors.append(prefix+'_must_be_object'); continue
        cid=component.get('id')
        if not isinstance(cid,str) or not cid or cid in seen:
            errors.append(prefix+'_id_missing_or_duplicate')
        else:
            seen.add(cid)
        kind=component.get('type')
        if kind not in ['tone','chirp','transient','broadband','modulation']:
            errors.append(prefix+'_unknown_morphology')
        expected_domain='modulation' if kind=='modulation' else 'acoustic'
        if component.get('frequency_domain')!=expected_domain:
            errors.append(prefix+'_invalid_frequency_domain')
        if not pair(component.get('time_s'),strict=True):
            errors.append(prefix+'_invalid_time_interval')
        if not pair(component.get('frequency_hz'),nonnegative=True):
            errors.append(prefix+'_invalid_frequency_interval')
        if not strings(component.get('evidence_ids')):
            errors.append(prefix+'_evidence_ids_must_be_string_list')
        source=component.get('source_identity','unknown')
        if source is not None and not isinstance(source,str):
            errors.append(prefix+'_source_identity_must_be_string_or_null')
        band=component.get('carrier_band_hz')
        if (kind=='modulation' or band is not None) and not pair(band,strict=True,nonnegative=True):
            errors.append(prefix+'_invalid_or_missing_carrier_band')
        track=component.get('frequency_track')
        if track is not None or kind=='chirp':
            if (not isinstance(track,list) or (kind=='chirp' and len(track)<2)
                or any(not isinstance(point,list) or len(point)!=2 or not all(number(x) for x in point) for point in track)):
                errors.append(prefix+'_invalid_frequency_track')
            elif any(track[k+1][0]<=track[k][0] for k in range(len(track)-1)):
                errors.append(prefix+'_frequency_track_not_ordered')
    for index, relation in enumerate(report['relations']):
        if (not isinstance(relation,dict) or not isinstance(relation.get('type'),str)
            or not strings(relation.get('component_ids')) or len(relation.get('component_ids',[]))<2
            or not strings(relation.get('evidence_ids'))):
            errors.append(f'relations[{index}]_invalid_structure')
    for index, claim in enumerate(report['claims']):
        if (not isinstance(claim,dict) or not isinstance(claim.get('text'),str)
            or not isinstance(claim.get('kind'),str) or not strings(claim.get('component_ids'))
            or not strings(claim.get('evidence_ids'))):
            errors.append(f'claims[{index}]_invalid_structure')
    if any(not isinstance(question,str) for question in report['unresolved_questions']):
        errors.append('unresolved_questions_must_contain_strings')
    return errors


def compact_state(state: dict, quantitative: bool = True, graph: bool = True) -> dict:
    return compact_measurements(state,quantitative=quantitative,graph=graph)


def restricted_state(state, method):
    with GPU_LOCK:
        return _restricted_state(state,method)


def _restricted_state(state, method):
    """Ablation of the visible information, preserving the full audit separately."""
    if method in ['G1','no_demon']:
        state = dict(state, components=[c for c in state.get('components',[]) if c.get('frequency_domain')!='modulation'],
                     measurements=[m for m in state.get('measurements',[]) if m.get('frequency_domain')!='modulation'],relations=[])
    if method=='forced_exclusivity':
        retained=[]
        for component in state.get('components',[]):
            conflicting=False
            for previous in retained:
                if component.get('frequency_domain') != previous.get('frequency_domain'): continue
                times=torch.tensor([component['time_s'],previous['time_s']],device='cuda')
                frequencies=torch.tensor([component['frequency_hz'],previous['frequency_hz']],device='cuda')
                time_overlap=times[:,1].min()>times[:,0].max()
                frequency_overlap=frequencies[:,1].min()>=frequencies[:,0].max()-2
                if bool(time_overlap & frequency_overlap): conflicting=True; break
            if not conflicting: retained.append(component)
        state=dict(state,components=retained,relations=[])
    if method=='no_graph':
        flat=[]
        for measurement in state.get('measurements',[]):
            if not isinstance(measurement.get('observed_components'),list):
                raise ValueError('no_graph requires raw observed_components for every measurement; stale state is unsupported')
            flat.extend(measurement['observed_components'])
        state=dict(state,components=flat,relations=[])
    return state


def action_menu(state, allow_demon=True, fixed_resolution=False):
    start, end = state['time_s']
    duration = end-start
    midpoint = (start+end)/2
    long_window = 1. if fixed_resolution else min(4., duration/2)
    menu = [
        {'name':'inspect_region','time_s':[start,end],'frequency_hz':[0.,800.],'window_s':long_window},
        {'name':'inspect_region','time_s':[midpoint,end],'frequency_hz':[0.,1800.],'window_s':1. if fixed_resolution else .25},
        {'name':'measure_peaks','time_s':[start,end],'frequency_hz':[0.,1800.],'window_s':1.},
    ]
    if allow_demon:
        for band in [[500.,2000.],[2000.,5000.],[5000.,8000.]]:
            if band[1] <= state['sample_rate_hz']/2-100:
                menu.append({'name':'inspect_modulation','time_s':[start,end],
                             'carrier_band_hz':band,'window_s':min(4.,duration/2)})
    components = state.get('components', [])
    if len(components) >= 2:
        menu.append({'name':'inspect_overlap','component_ids':[components[0]['id'],components[1]['id']]})
        acoustic = [c for c in components if c.get('frequency_domain')=='acoustic' and c.get('type') in ['tone','chirp']]
        if len(acoustic) >= 2:
            menu.append({'name':'test_harmonic_relation','component_ids':[c['id'] for c in acoustic[:6]]})
    acoustic_components=[c for c in components if c.get('frequency_domain')=='acoustic']
    if acoustic_components:
        menu.append({'name':'track_component','component_id':acoustic_components[0]['id'],
                     'time_s':[start,end],'window_s':1.})
    menu.append({'name':'compare_intervals','intervals_s':[[start,midpoint],[midpoint,end]],
                 'frequency_hz':[0.,1000.],'window_s':1.})
    return menu


def verify_report(report, state):
    with GPU_LOCK:
        return _verify_report(report,state)


def _verify_report(report, state):
    """Internal consistency only; NOT independent supported-claim correctness.

    A component must cite an existing detector measurement and agree with a
    cited component's numeric range, domain, and carrier band. Unsupported
    source claims are withheld. The unmodified original is always retained.
    """
    schema_errors=report_schema_errors(report)
    if schema_errors:
        return {'components':[], 'relations':[], 'claims':[], 'abstained':True,
                'unresolved_questions':['Model generation did not satisfy the report schema.']}, [{'status':'schema_invalid','errors':schema_errors}]
    verified = dict(report)
    accepted, audit = [], []
    source_components = state.get('components', [])
    for component in report.get('components', []):
        reasons = []
        if not isinstance(component, dict):
            audit.append({'status':'withheld', 'reasons':['not an object']})
            continue
        matches = []
        for candidate in source_components:
            if not set(component.get('evidence_ids', [])) & set(candidate.get('evidence_ids', [])):
                continue
            if component.get('frequency_domain') != candidate.get('frequency_domain'):
                continue
            if component.get('type') != candidate.get('type'):
                continue
            try:
                frequency = component['frequency_hz']
                reference = candidate['frequency_hz']
                tolerance = 2*float(candidate.get('bin_spacing_hz', 1.))
                interval = component['time_s']
                expected = candidate['time_s']
                boundary = max(.5, float(candidate.get('boundary_uncertainty_s', .5)))
                frequencies = torch.tensor([frequency, reference], device='cuda', dtype=torch.float32)
                boundaries = torch.tensor([interval, expected], device='cuda', dtype=torch.float32)
                agrees = bool(((frequencies[0]-frequencies[1]).abs()<=tolerance).all())
                agrees = agrees and bool(((boundaries[0]-boundaries[1]).abs()<=boundary).all())
                if component.get('type')=='chirp' or (
                    component.get('type') in ['tone','modulation'] and component.get('frequency_track') is not None):
                    from .metrics import _interpolate_track
                    track=component.get('frequency_track',[])
                    if len(track)<2:
                        agrees=False
                    else:
                        track_tensor=torch.tensor(track,device='cuda')
                        ordered=bool((track_tensor[1:,0]>track_tensor[:-1,0]).all())
                        sample_times=torch.linspace(max(interval[0],expected[0]),min(interval[1],expected[1]),17,device='cuda')
                        track_error=(_interpolate_track(component,sample_times)-_interpolate_track(candidate,sample_times)).abs().mean()
                        agrees=agrees and ordered and bool(track_error<=tolerance)
                if component.get('frequency_domain') == 'modulation':
                    agrees = agrees and component.get('carrier_band_hz') == candidate.get('carrier_band_hz')
                if agrees:
                    matches.append(candidate['id'])
            except (KeyError, TypeError, ValueError, RuntimeError):
                continue
        if not matches:
            reasons.append('no cited component with matching morphology, domain, coordinates, and carrier')
        if component.get('source_identity', 'unknown') not in ['unknown', None]:
            reasons.append('physical source identity has no independent evidence')
        if not reasons:
            accepted.append(dict(component, verification_status='measurement_consistent'))
        audit.append({'component_id':component.get('id'), 'status':'withheld' if reasons else 'measurement_consistent',
                      'reasons':reasons, 'matched_measurement_components':matches})
    verified['components'] = accepted
    accepted_ids = {x.get('id') for x in accepted}
    verified['claims'] = [dict(c, verification_status='component_references_consistent_text_not_verified')
                          for c in report.get('claims', []) if isinstance(c,dict)
                          and c.get('kind')=='observation' and c.get('component_ids')
                          and set(c['component_ids']).issubset(accepted_ids)]
    verified['relations'] = [r for r in report.get('relations', []) if isinstance(r,dict)
                             and set(r.get('component_ids',[])).issubset(accepted_ids)]
    verified['abstained'] = not bool(accepted)
    return verified, audit


def run_episode(tools, model, method, output_dir, budget=6, random_seed=0, max_report_tokens=2048, render_cmap='magma'):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    allow_demon = method not in ['G1','no_demon']
    with GPU_LOCK:
        torch.cuda.synchronize()
        before = torch.cuda.Event(enable_timing=True)
        after = torch.cuda.Event(enable_timing=True)
        before.record()
        views = tools.overview(include_demon=allow_demon)
        after.record(); torch.cuda.synchronize()
        preprocess_seconds = before.elapsed_time(after)/1000
    quantitative = method not in ['G1','no_quantitative']
    graph = method != 'no_graph'
    image_paths = []
    for name, view in views.items():
        if view is None or (name == 'demon' and not allow_demon):
            continue
        image_paths.append(render_view(view, output / f'global_{name}.png', f'{name.upper()} overview',cmap=render_cmap))
    state = tools.report_state()
    initial_state = state
    state=restricted_state(state,method)
    action_logs, generations, used = [], [], set()
    do_actions = method not in ['G1','G2']
    for step in range(budget if do_actions else 0):
        menu = action_menu(state, allow_demon=allow_demon, fixed_resolution=method=='fixed_resolution')
        if method not in ['G3','G4','random']:
            menu.append({'name':'finalize_or_abstain'})
        unused = [(i,a) for i,a in enumerate(menu) if json.dumps(a,sort_keys=True) not in used]
        if not unused:
            break
        if method in ['G3']:
            index, action = unused[0]
            reason = 'Frozen fixed schedule.'
        elif method in ['G4']:
            # Explicit heuristic, no claim of learned uncertainty/information gain.
            observed_tools={a['action'].get('name') for a in action_logs if a.get('status')=='completed'}
            overlap_conflicts=sum(r.get('type')=='unresolved_association' for r in state.get('relations',[]))
            tonal_count=sum(c.get('type') in ['tone','chirp'] for c in state.get('components',[]))
            def priority(action):
                name=action['name']
                value=1. if name not in observed_tools else .25
                if name=='inspect_overlap': value+=min(3.,float(overlap_conflicts))
                if name=='test_harmonic_relation': value+=min(2.,max(0.,tonal_count-1.))
                if name=='inspect_modulation':
                    band=action.get('carrier_band_hz')
                    seen=any(m.get('carrier_band_hz')==band for m in state.get('measurements',[]))
                    value+=2. if not seen else 0.
                if name=='inspect_region': value+=.5
                return value
            index, action = max(unused, key=lambda pair:priority(pair[1]))
            reason = 'Frozen novelty and unresolved cross-view/harmonic-conflict priority; not learned information gain.'
        elif method == 'random':
            with GPU_LOCK:
                generator = torch.Generator(device='cuda').manual_seed(random_seed+step)
                selected = int(torch.randint(len(unused), (), device='cuda', generator=generator))
            index, action = unused[selected]
            reason = 'Seeded random valid-action control.'
        else:
            prompt = OBSERVATION_RULES + '\nChoose the next measurement to resolve an uncertainty. '
            prompt += 'Return {"action_index":0,"reason":"short observable purpose"}. '
            prompt += 'Choose only an index in AVAILABLE. Finalize when no useful measurement would resolve the remaining uncertainties; report unresolved source identity.\n'
            prompt += json.dumps({'state':compact_state(state,quantitative,graph),
                                  'history':prompt_numbers([{'action':a['action'],'status':a.get('status')} for a in action_logs]),
                                  'AVAILABLE':prompt_numbers([{**a,'action_index':i} for i,a in unused])}, separators=(',',':'))
            generated = model.generate(prompt, image_paths[-3:], 160,
                                       json_schema=action_json_schema([i for i,_ in unused]))
            generations.append(dict(generated, phase='action', step=step, prompt=prompt))
            value = generated.get('parsed') or {}
            chosen = value.get('action_index')
            match = [pair for pair in unused if type(chosen) is int and pair[0] == chosen]
            if not match:
                action_logs.append({'step':step,'action':value,'status':'invalid_model_action', 'attempt_counted':True})
                continue
            index, action = match[0]
            reason = str(value.get('reason',''))
        used.add(json.dumps(action,sort_keys=True))
        known_views = set(tools.latest_views)
        with GPU_LOCK:
            tool_start = torch.cuda.Event(enable_timing=True)
            tool_end = torch.cuda.Event(enable_timing=True)
            tool_start.record()
            try:
                returned = tools.execute(action)
                status = 'completed' if returned.get('status') not in ['error','unavailable'] else returned['status']
            except (ValueError,KeyError,TypeError,RuntimeError) as error:
                returned = {'error':str(error)}
                status = 'failed'
            tool_end.record(); torch.cuda.synchronize()
        action_logs.append({'step':step,'action':action,'reason':reason,'status':status,
                            'gpu_seconds':tool_start.elapsed_time(tool_end)/1000,'result':returned})
        for evidence_id in tools.latest_views.keys()-known_views:
            view = tools.latest_views[evidence_id]
            if view is not None:
                image_paths.append(render_view(view,output/f'action_{step}_{evidence_id}.png',f'{action["name"]}: {evidence_id}',cmap=render_cmap))
        state = tools.report_state()
        state=restricted_state(state,method)
        if action['name']=='finalize_or_abstain':
            break
    prompt = OBSERVATION_RULES+'\n'+REPORT_SCHEMA+'\n'
    prompt += json.dumps(compact_state(state, quantitative, graph),separators=(',',':'))
    generated = model.generate(prompt, image_paths[:2] + image_paths[-1:] if len(image_paths)>2 else image_paths,
                               max_report_tokens,json_schema=report_json_schema())
    generations.append(dict(generated, phase='report', prompt=prompt))
    raw = generated.get('parsed')
    schema_errors=report_schema_errors(raw)
    valid_raw=raw if not schema_errors else None
    verified, audit = verify_report(valid_raw,state)
    # G2/G3/G4/F share the same verifier; G1 has no quantitative tool evidence.
    report = valid_raw if method in ['G1','no_verifier'] else verified
    result = {'method':method,'scene_id':state['scene_id'],'action_budget':budget,
              'prompt_serialization':PROMPT_PROTOCOL,
              'rendering':{'colormap':render_cmap,'dynamic_range_db':60,'frequency_max_pool_bins':256,'quantitative_measurements_from_unpooled_PSD':True},
              'actions_attempted':len(action_logs), 'actions':action_logs,
              'raw_report':raw, 'report':report, 'verification_audit':audit,
              'verification_meaning':'Internal measurement consistency, not independent correctness.',
              'initial_state':initial_state, 'final_state':tools.report_state(),
              'generations':generations,'preprocess_gpu_seconds':preprocess_seconds,
              'model_gpu_seconds':sum(g['gpu_seconds'] for g in generations),
              'tool_gpu_seconds':sum(a.get('gpu_seconds',0.) for a in action_logs),
              'wall_seconds':time.perf_counter()-started,
              'input_tokens':sum(g['input_tokens'] for g in generations),
              'output_tokens':sum(g['output_tokens'] for g in generations),
              'schema_valid':not schema_errors, 'schema_errors':schema_errors,
              'reference_type':'AI-generated morphology labels for real data; exact simulator labels for synthetic data'}
    (output/'episode.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
