"""Blinded local AI reference labels, separate from the evaluated controller.

These labels are machine opinions. Consensus and schema checks are not independent
physical validation, and undiscovered events are not established negatives.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .signal import require_cuda, lofar, demon
from .rendering import render_view
from .json_protocol import parse_json
from .metrics import event_metrics

EVENT_TYPES = {"tone", "chirp", "broadband", "transient", "modulation"}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_complete_event_rows(raw:str)->tuple[dict|None,dict]:
    """Only close unclosed outer containers around already complete event rows.

    No values, keys, event rows, strings, commas, or colons are invented. A
    truncated event row and the malformed {"events"} case remain invalid.
    """
    try:
        parsed=parse_json(raw)
        return parsed,dict(raw_json_valid=True,syntax_repair_applied=False,numeric_values_changed=False)
    except ValueError as original:
        original_error=str(original)
    start=raw.find('{')
    if start<0:return None,dict(raw_json_valid=False,syntax_repair_applied=False,error=original_error)
    text=raw[start:].strip()
    if text.endswith('```'):text=text[:-3].rstrip()
    stack=[];quoted=False;escape=False;mismatch=False
    for char in text:
        if quoted:
            if escape:escape=False
            elif char=='\\':escape=True
            elif char=='"':quoted=False
        elif char=='"':quoted=True
        elif char in '{[':stack.append(char)
        elif char in '}]':
            if not stack or (stack[-1],char) not in [('{','}'),('[',']')]:mismatch=True;break
            stack.pop()
    if quoted or mismatch or stack not in [['{'],['{','[']] or not text.endswith(']'):
        return None,dict(raw_json_valid=False,syntax_repair_applied=False,error=original_error)
    suffix=''.join('}' if x=='{' else ']' for x in reversed(stack))
    try:
        candidate=json.loads(text+suffix)
        if set(candidate)!={'events'} or not isinstance(candidate['events'],list):raise ValueError('Unexpected schema')
        for row in candidate['events']:
            if not isinstance(row,list) or len(row)!=5 or not isinstance(row[4],str):raise ValueError('Incomplete event row')
            if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in row[:4]):raise ValueError('Incomplete event values')
    except (json.JSONDecodeError,ValueError,TypeError):
        return None,dict(raw_json_valid=False,syntax_repair_applied=False,error=original_error)
    return candidate,dict(raw_json_valid=False,syntax_repair_applied=True,appended_delimiters=suffix,
                          numeric_values_changed=False,raw_error=original_error,
                          rule='close_only_outer_containers_around_complete_event_rows')


def annotation_prompt(start_s: float, duration_s: float, views: list[dict], max_events=12) -> str:
    available = [{key: value for key, value in view.items()
                  if key in ("name", "frequency_domain", "window_s", "bin_spacing_hz",
                             "carrier_band_hz", "frequency_range_hz", "available")}
                 for view in views]
    return f"""You are making an AI reference annotation of observable passive underwater acoustic structure.
You see fixed raw spectral displays with axes, not detector candidates. Analyze only these images.
The time coordinates are ORIGINAL RECORDING seconds, from {start_s:.6f} to {start_s+duration_s:.6f}.
The low-frequency acoustic plots cover only 0--1800 acoustic Hz. DEMON plots show envelope
modulation Hz in the named carrier band. Do not confuse acoustic with modulation frequency.
The two acoustic resolutions show the SAME events: do not duplicate them. The two DEMON
bands can contain separate modulation events; name the carrier for each.
Visible thin persistent lines are tones, sloping lines chirps, brief wide vertical regions
transients, sustained wide regions broadband, and periodic envelope ridges modulation.
Keep overlapping components separate. Do not identify vessels, machinery, propellers,
blade counts, ranges, source power, calibrated SPL, or physical mechanisms from these views.
Do not invent exact precision beyond visible axis resolution. It is acceptable to return
no confident events and describe uncertainty. Broad diffuse background alone is not a
localized event. Represent a harmonic family as individual tone/chirp events plus
harmonic_candidate relations. Unresolved overlap belongs in uncertainties or
unresolved_association relations, not an additional component event.
Return at most {max_events} events as ONE JSON object and no surrounding prose:
{{"events":[{{"id":"a0","type":"tone|chirp|broadband|transient|modulation",
"frequency_domain":"acoustic|modulation","time_s":[start,end],"frequency_hz":[low,high],
"frequency_track":[[time,frequency],[time,frequency]],"carrier_band_hz":null,
"frequency_uncertainty_hz":4.0,"boundary_uncertainty_s":1.0,
"confidence":"clear|uncertain|unresolved","description":"observable morphology only"}}],
"relations":[{{"source_id":"a0","target_id":"a1","type":"overlaps_time|overlaps_time_frequency|harmonic_candidate|unresolved_association"}}],
"reviewed_regions":[{{"frequency_domain":"acoustic","time_s":[start,end],"frequency_hz":[0,1800],"carrier_band_hz":null}}],
"inspected_no_events":false,"uncertainties":["remaining limitations"]}}
Use numeric values, not placeholder words. Frequency intervals may have equal endpoints
for a single thin ridge. For modulation set type=modulation, frequency_domain=modulation,
and carrier_band_hz exactly [500,2000] or [2000,5000]. Omit frequency_track for diffuse
events. Reviewed regions are your declared coverage, not independently proven negatives.
If events is empty, set inspected_no_events=true only if you actually inspected the images.
Available views: {json.dumps(available, separators=(',', ':'))}
"""


def prepare_fixed_views(waveform: torch.Tensor, sample_rate: int, output_dir,
                        offset_s: float = 0., duration_s: float | None = None,
                        display_policy: str = "percentile_995") -> dict:
    """No candidate detector, labels, graph state, or controller output is read."""
    require_cuda(waveform)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    duration_s = duration_s or waveform.numel()/sample_rate
    specifications = [
        ("lofar_1s", "acoustic", 1., None),
        ("lofar_4s", "acoustic", 4., None),
        ("demon_500_2000", "modulation", 4., (500., 2000.)),
        ("demon_2000_5000", "modulation", 4., (2000., 5000.)),
    ]
    records=[]
    for name, domain, window, band in specifications:
        row=dict(name=name, frequency_domain=domain,window_s=window,
                 carrier_band_hz=list(band) if band else None,
                 frequency_range_hz=[0.,1800.] if domain=="acoustic" else [.5,200.])
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        try:
            start.record()
            view=(lofar(waveform,sample_rate,window_s=window,offset_s=offset_s)
                  if domain=="acoustic" else
                  demon(waveform,sample_rate,band,window_s=window,offset_s=offset_s))
            end.record(); torch.cuda.synchronize()
            row["signal_gpu_seconds"]=start.elapsed_time(end)/1000
            path=directory/(name+".png")
            rendering_start=time.perf_counter()
            title=("LOFAR " if domain=="acoustic" else "DEMON ")+f"{window:g} s window"
            if display_policy=="peak_60db":
                row["display_scale"]=render_reference_view(view,path,title)
            elif display_policy=="percentile_995":
                render_view(view,path,title=title,components=())
            else:
                raise ValueError("Unknown fixed reference display policy")
            row.update(available=True,path=str(path),image_sha256=file_hash(path),
                       bin_spacing_hz=view.metadata["bin_spacing_hz"],
                       view_metadata=view.metadata,render_wall_seconds=time.perf_counter()-rendering_start)
            del view
        except ValueError as exc:
            row.update(available=False,unavailable_reason=str(exc))
        records.append(row)
    result=dict(offset_s=offset_s,duration_s=duration_s,sample_rate=sample_rate,
                views=records,candidate_detections_supplied=False,source_labels_supplied=False,
                waveform_device=str(waveform.device),display_policy=display_policy)
    (directory/"views.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def render_reference_view(view,path,title,max_pool_frequency_bins=None,time_label="Original-recording time (s)"):
    """Unwhitened PSD, peak-preserving fixed 60 dB display range, no detections."""
    import matplotlib.pyplot as plt
    power=view.log_power()
    maximum=power.max();minimum=power.min()
    vmax=maximum.item();vmin=vmax-60.
    constant=bool(maximum==minimum)
    if constant:
        # Constant images have no localized contrast. Draw their value at the
        # low end instead of saturating every pixel as a bright false structure.
        vmin=vmax;vmax=vmin+60.
    frequency_tensor=view.frequencies_hz
    display_bin_width=view.metadata["bin_spacing_hz"]
    if max_pool_frequency_bins and power.shape[0]>max_pool_frequency_bins:
        original_bins=power.shape[0]
        display_bin_width=original_bins/max_pool_frequency_bins*view.metadata["bin_spacing_hz"]
        power=torch.nn.functional.adaptive_max_pool2d(power[None,None],(max_pool_frequency_bins,power.shape[1]))[0,0]
        lower=torch.floor(torch.arange(max_pool_frequency_bins,device=power.device)*original_bins/max_pool_frequency_bins).long()
        upper=torch.ceil((torch.arange(max_pool_frequency_bins,device=power.device)+1)*original_bins/max_pool_frequency_bins).long()-1
        frequency_tensor=(frequency_tensor[lower]+frequency_tensor[upper])/2
    frequencies=frequency_tensor.detach().cpu().numpy()
    times=view.times_s.detach().cpu().numpy()
    fig,axis=plt.subplots(figsize=(8,4),constrained_layout=True)
    mesh=axis.pcolormesh(times,frequencies,power.detach().cpu().numpy(),
                         shading="nearest",cmap="magma",vmin=vmin,vmax=vmax)
    domain=view.metadata["frequency_domain"]
    axis.set(xlabel=time_label,ylabel=f"{domain.capitalize()} frequency (Hz)",title=title)
    axis.ticklabel_format(axis="x",style="plain",useOffset=False)
    if domain=="modulation":
        band=view.metadata["carrier_band_hz"]
        axis.set_title(f"{title} | carrier {band[0]:g}–{band[1]:g} Hz")
    fig.colorbar(mesh,ax=axis,label="PSD (dB re digital amplitude²/Hz)")
    fig.savefig(path,dpi=120);plt.close(fig)
    return dict(policy="peak_60db",vmin_db=vmin,vmax_db=vmax,constant_view=constant,
                raw_psd_unchanged=True,scale_numeric_device=str(power.device),
                display_frequency_bin_width_hz=display_bin_width,
                display_aggregation="GPU_frequency_max_pool" if max_pool_frequency_bins else "none")


def render_modulation_reference(view,path,title="Envelope modulation evidence",time_label="Original-recording time (s)"):
    """Fixed raw DEMON plus its mean PSD, with no detected/candidate peaks."""
    import matplotlib.pyplot as plt
    power=view.log_power();nf,nt=power.shape
    count=min(nf,360)
    pooled=torch.nn.functional.adaptive_max_pool2d(power[None,None],(count,nt))[0,0]
    lower=torch.floor(torch.arange(count,device=power.device)*nf/count).long()
    upper=torch.ceil((torch.arange(count,device=power.device)+1)*nf/count).long()-1
    frequency=(view.frequencies_hz[lower]+view.frequencies_hz[upper])/2
    maximum=power.max().item();minimum=maximum-60.
    if bool(power.max()==power.min()):minimum=maximum;maximum+=60.
    average=10*torch.log10(view.psd.mean(dim=1).clamp_min(1e-20))
    fig,axes=plt.subplots(1,2,figsize=(11,4),width_ratios=[2,1],constrained_layout=True)
    axes[0].pcolormesh(view.times_s.cpu().numpy(),frequency.cpu().numpy(),pooled.cpu().numpy(),
                       cmap="magma",shading="nearest",vmin=minimum,vmax=maximum)
    axes[0].set(xlabel=time_label,ylabel="Modulation frequency (Hz)",title="Time-resolved envelope spectrum")
    axes[0].ticklabel_format(axis="x",style="plain",useOffset=False)
    axes[1].plot(view.frequencies_hz.cpu().numpy(),average.cpu().numpy(),linewidth=1.2)
    axes[1].set(xlabel="Modulation frequency (Hz)",ylabel="Mean PSD (dB re digital amplitude²/Hz)",title="Same data averaged over time")
    axes[1].grid(alpha=.2)
    band=view.metadata["carrier_band_hz"]
    fig.suptitle(f"{title} | acoustic carrier {band[0]:g}–{band[1]:g} Hz")
    fig.savefig(path,dpi=120);plt.close(fig)
    return dict(policy="peak_60db_with_fixed_raw_mean_psd",vmin_db=minimum,vmax_db=maximum,
                display_frequency_bin_width_hz=nf/count*view.metadata["bin_spacing_hz"],
                display_aggregation="GPU_frequency_max_pool",candidate_peaks_supplied=False,
                mean_psd_numeric_device=str(power.device),raw_psd_unchanged=True)


class AnnotationVLM:
    """Native Transformers classes only; explicit CUDA placement, no CPU offload."""
    def __init__(self,spec:dict,config:dict):
        require_cuda()
        from transformers import AutoProcessor, InternVLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
        classes={"InternVLForConditionalGeneration":InternVLForConditionalGeneration,
                 "Qwen2_5_VLForConditionalGeneration":Qwen2_5_VLForConditionalGeneration}
        self.spec=spec
        self.config=config
        options=dict(local_files_only=True,trust_remote_code=False)
        if spec["model_class"]=="InternVLForConditionalGeneration":
            options.update(min_patches=1,max_patches=spec.get("max_patches",2),crop_to_patches=True)
        else:
            options.update(min_pixels=65536,max_pixels=config.get("max_pixels",262144))
        self.processor=AutoProcessor.from_pretrained(spec["model_path"],**options)
        self.model=classes[spec["model_class"]].from_pretrained(
            spec["model_path"],dtype=torch.bfloat16,device_map={"":"cuda:0"},
            attn_implementation="sdpa",local_files_only=True,trust_remote_code=False).eval()
        for name,value in list(self.model.named_parameters())+list(self.model.named_buffers()):
            if value.device.type!="cuda":
                raise RuntimeError(f"Non-CUDA annotation model tensor: {name} on {value.device}")
        self.device_audit=dict(all_parameters_cuda=True,all_buffers_cuda=True,
                               dtype="bfloat16",hf_device_map=self.model.hf_device_map,
                               parameter_count=sum(x.numel() for x in self.model.parameters()))

    @torch.inference_mode()
    def generate(self,prompt:str,images:list[str],max_new_tokens:int|None=None,examples:list[dict]|None=None)->dict:
        wall=time.perf_counter()
        content=[{"type":"image","image":Image.open(p).convert("RGB")} for p in images]
        content.append({"type":"text","text":prompt})
        messages=[]
        for example in examples or []:
            messages.extend([{"role":"user","content":[{"type":"image","image":Image.open(example["image"]).convert("RGB")},{"type":"text","text":example["prompt"]}]},
                             {"role":"assistant","content":[{"type":"text","text":json.dumps(example["answer"],separators=(',',':'))}]}])
        messages.append({"role":"user","content":content})
        inputs=self.processor.apply_chat_template(messages,
                    tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt").to("cuda:0")
        for key,value in inputs.items():
            if isinstance(value,torch.Tensor) and value.is_floating_point() and key.startswith("pixel_values"):
                inputs[key]=value.to(torch.bfloat16)
        maximum=max_new_tokens or self.config.get("max_new_tokens",1536)
        torch.cuda.synchronize()
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats(); start.record()
        outputs=self.model.generate(**inputs,max_new_tokens=maximum,do_sample=False,use_cache=True)
        end.record(); torch.cuda.synchronize()
        ids=outputs[:,inputs["input_ids"].shape[1]:]
        raw=self.processor.batch_decode(ids,skip_special_tokens=True)[0]
        result=dict(raw_text=raw,input_tokens=int(inputs["input_ids"].shape[1]),
                    output_tokens=int(ids.shape[1]),gpu_seconds=start.elapsed_time(end)/1000,
                    wall_seconds=time.perf_counter()-wall,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    max_new_tokens=maximum,at_generation_limit=ids.shape[1]==maximum,decode="greedy",
                    model_id=self.spec["model_id"],model_revision=self.spec["revision"],
                    device_audit=self.device_audit,
                    input_tensor_shapes={k:list(v.shape) for k,v in inputs.items() if isinstance(v,torch.Tensor)})
        result["training_examples"]=[dict(image=x["image"],image_sha256=file_hash(x["image"]),answer=x["answer"],prompt=x["prompt"],provenance=x.get("provenance")) for x in examples or []]
        try:
            result.update(parsed=parse_json(raw),parse_error=None)
        except ValueError as exc:
            result.update(parsed=None,parse_error=str(exc))
        return result


def _pair(value,name,strict=False):
    if not isinstance(value,list) or len(value)!=2 or not all(isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x) for x in value):
        raise ValueError(name+" must contain two finite numbers")
    if value[1]<value[0] or (strict and value[1]==value[0]):
        raise ValueError(name+" is reversed or empty")
    return [float(x) for x in value]


def validate_annotation(parsed:dict|None,start_s:float,duration_s:float,max_events=12)->dict:
    """Structural/range validation only; never certify physical correctness."""
    if parsed is None or not isinstance(parsed.get("events"),list):
        return dict(annotation_status="invalid",events=[],invalid_events=[],errors=["Missing events array"])
    valid,invalid=[],[]
    identifiers=set()
    for index,event in enumerate(parsed["events"]):
        try:
            if not isinstance(event,dict):raise ValueError("Event must be an object")
            if index>=max_events:raise ValueError("Declared event budget exceeded")
            typ=event.get("type"); domain=event.get("frequency_domain")
            if typ not in EVENT_TYPES:raise ValueError("Unknown observable morphology")
            if domain not in {"acoustic","modulation"}:raise ValueError("Unknown frequency domain")
            interval=_pair(event.get("time_s"),"time_s",strict=True)
            frequencies=_pair(event.get("frequency_hz"),"frequency_hz")
            if interval[0]<start_s-.01 or interval[1]>start_s+duration_s+.01:
                raise ValueError("Time outside supplied scene's original coordinates")
            if frequencies[0]<0 or frequencies[1]>(1800 if domain=="acoustic" else 200):
                raise ValueError("Frequency outside supplied display")
            eid=event.get("id")
            if not isinstance(eid,str) or not eid or eid in identifiers:raise ValueError("Missing/duplicate event id")
            if domain=="modulation":
                if typ!="modulation":raise ValueError("Modulation domain requires modulation type")
                if event.get("carrier_band_hz") not in [[500,2000],[2000,5000]]:raise ValueError("Missing/invalid modulation carrier")
            if typ=="modulation" and domain!="modulation":raise ValueError("Modulation type with acoustic frequency")
            for name in ["frequency_uncertainty_hz","boundary_uncertainty_s"]:
                value=event.get(name)
                if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:raise ValueError("Invalid "+name)
            track=event.get("frequency_track")
            if track:
                previous=-float("inf")
                for point in track:
                    if not isinstance(point,list) or len(point)!=2 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in point):raise ValueError("Malformed ridge point")
                    if not interval[0]<=point[0]<=interval[1] or point[0]<=previous:raise ValueError("Ridge time outside support or unordered")
                    if not 0<=point[1]<=(1800 if domain=="acoustic" else 200):raise ValueError("Ridge frequency outside displayed domain")
                    if not frequencies[0]-event["frequency_uncertainty_hz"]<=point[1]<=frequencies[1]+event["frequency_uncertainty_hz"]:raise ValueError("Ridge frequency inconsistent with interval")
                    previous=point[0]
            cleaned=dict(event,time_s=interval,frequency_hz=frequencies,
                         annotation_origin="AI_reference",independent_ground_truth=False)
            valid.append(cleaned);identifiers.add(eid)
        except (ValueError,TypeError) as exc:
            invalid.append(dict(index=index,raw_event=event,error=str(exc)))
    errors=[]
    if not parsed["events"] and parsed.get("inspected_no_events") is not True:
        errors.append("Empty events without explicit inspection declaration")
    relations=[]
    for relation in parsed.get("relations",[]):
        if isinstance(relation,dict) and relation.get("source_id") in identifiers and relation.get("target_id") in identifiers and relation.get("type") in {"overlaps_time","overlaps_time_frequency","harmonic_candidate","unresolved_association"}:
            relations.append(relation)
        else:
            errors.append("Invalid relation endpoint")
    return dict(annotation_status="complete" if not invalid and not errors else "partial_or_invalid",
                events=valid,relations=relations,invalid_events=invalid,errors=errors,
                reviewed_regions=parsed.get("reviewed_regions",[]),
                inspected_no_events=parsed.get("inspected_no_events") is True,
                uncertainties=parsed.get("uncertainties",[]),
                independently_exhaustive=False,independent_ground_truth=False,
                semantic_correctness_assessed=False)


def annotate_scene(model:AnnotationVLM,scene_id:str,view_bundle:dict,output_file,config:dict)->dict:
    prompt=annotation_prompt(view_bundle["offset_s"],view_bundle["duration_s"],view_bundle["views"],config.get("max_events",12))
    available=[x for x in view_bundle["views"] if x["available"]]
    result=model.generate(prompt,[x["path"] for x in available])
    validated=validate_annotation(result["parsed"],view_bundle["offset_s"],view_bundle["duration_s"],config.get("max_events",12))
    record=dict(scene_id=scene_id,annotation_origin="AI_reference",annotation=validated,generation=result,
                prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                configuration_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
                image_inputs=[dict(path=x["path"],sha256=x["image_sha256"]) for x in available],
                scene_interval_s=[view_bundle["offset_s"],view_bundle["offset_s"]+view_bundle["duration_s"]],
                source_labels_supplied=False,candidate_detections_supplied=False,
                main_controller_outputs_supplied=False,simulator_parameters_supplied=False)
    destination=Path(output_file);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
    return record


def reconcile_annotations(first:dict,second:dict,config:dict)->dict:
    """Conservative geometric agreement, with CUDA arithmetic and explicit disputes."""
    device=require_cuda()
    if first["scene_id"]!=second["scene_id"]:raise ValueError("Different scenes cannot be reconciled")
    a,b=first["annotation"]["events"],second["annotation"]["events"]
    # AI uncertainty is not calibrated and cannot make matching arbitrarily lax.
    b_bounded=[dict(x,frequency_uncertainty_hz=min(x.get("frequency_uncertainty_hz",0),config.get("consensus_max_ai_uncertainty_hz",8))) for x in b]
    agreement=event_metrics(a,b_bounded,
            time_iou_threshold=config.get("consensus_time_iou_threshold",.3),
            box_iou_threshold=config.get("consensus_box_iou_threshold",.3),
            frequency_tolerance_hz=config.get("consensus_min_frequency_tolerance_hz",4),exhaustive=True)
    kept=[];ia=set();ib=set()
    for match in agreement["matches"]:
        i,j=match["prediction_index"],match["gold_index"];ia.add(i);ib.add(j)
        av,bv=a[i],b[j]
        times=torch.tensor([av["time_s"],bv["time_s"]],device=device)
        frequencies=torch.tensor([av["frequency_hz"],bv["frequency_hz"]],device=device)
        interval=torch.stack((times[:,0].max(),times[:,1].min())).tolist()
        center=frequencies.mean(dim=0).tolist()
        event=dict(id=f"consensus:{len(kept)}",type=av["type"],frequency_domain=av["frequency_domain"],
                   time_s=interval,frequency_hz=center,
                   frequency_uncertainty_hz=min(config.get("consensus_max_ai_uncertainty_hz",8),max(av.get("frequency_uncertainty_hz",0),bv.get("frequency_uncertainty_hz",0))),
                   ai_reported_frequency_uncertainty_hz=[av.get("frequency_uncertainty_hz",0),bv.get("frequency_uncertainty_hz",0)],
                   matching_tolerance_capped=True,
                   boundary_uncertainty_s=max(av.get("boundary_uncertainty_s",0),bv.get("boundary_uncertainty_s",0)),
                   annotator_event_ids=[av["id"],bv["id"]],annotation_origin="two_AI_geometric_consensus",
                   independent_ground_truth=False)
        if av.get("carrier_band_hz"):event["carrier_band_hz"]=av["carrier_band_hz"]
        if av.get("frequency_track") and bv.get("frequency_track"):
            from .metrics import _interpolate_track
            t=torch.linspace(interval[0],interval[1],33,device=device)
            frequency=(_interpolate_track(av,t)+_interpolate_track(bv,t))/2
            event["frequency_track"]=torch.stack((t,frequency),dim=1).tolist()
        kept.append(event)
    disputed_a=[x for i,x in enumerate(a) if i not in ia]
    disputed_b=[x for i,x in enumerate(b) if i not in ib]
    return dict(scene_id=first["scene_id"],events=kept,
                annotation_origin="two_distinct_local_AI_reference_models",
                reference_status="complete" if all(x["annotation"]["annotation_status"]=="complete" for x in [first,second]) else "partial",
                ai_interannotator_agreement=agreement,
                disputed_events={"annotator_a":disputed_a,"annotator_b":disputed_b},
                disputed_regions=[{k:x[k] for k in ["time_s","frequency_hz","frequency_domain","carrier_band_hz"] if k in x} for x in disputed_a+disputed_b],
                independently_exhaustive=False,independent_ground_truth=False,
                negative_region_certification=False,
                shared_backbone_with_controller="Qwen2.5-VL shares the Qwen family; InternVL3.5 uses an InternVL vision encoder and Qwen3 text decoder. Distinct checkpoints are not statistically independent judges.",
                event_count_a=len(a),event_count_b=len(b),consensus_event_count=len(kept))
