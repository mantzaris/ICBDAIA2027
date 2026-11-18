#!/usr/bin/env python3
"""Fixed CUDA views, separate blinded AI annotation passes, and reconciliation."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import torch
from sonar.annotators import AnnotationVLM,annotate_scene,prepare_fixed_views,reconcile_annotations,file_hash
from sonar.data import load_waveform,read_manifest
from sonar.metrics import event_metrics,aggregate_event_metrics
from sonar.synthetic import KINDS


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def write_jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(x,sort_keys=True,allow_nan=False)+"\n" for x in rows))


def control_records(config):
    settings=config["calibration_controls"];records=[]
    for i in range(settings["count"]):
        seed=settings["seed_start"]+i
        sid="ac_"+hashlib.sha256(f"annotation-control:{seed}".encode()).hexdigest()[:16]
        records.append(dict(scene_id=sid,stratum="synthetic",split=settings["generator_split"],
                            seed=seed,kind=KINDS[i%len(KINDS)],duration_seconds=settings["duration_s"],
                            sample_rate=settings["sample_rate_hz"],snr_db=10,
                            group_id="annotation-control:"+str(seed),annotation_control=True,
                            control_role="annotation_calibration" if i<12 else "reference_error_audit"))
    return records


def prepare(args,config):
    records=[]
    for manifest in args.manifest or ["data/manifests/real_scenes.jsonl","data/manifests/mixture_scenes.jsonl"]:
        records.extend(read_manifest(manifest))
    if args.include_controls:
        controls=control_records(config);path=Path("data/manifests/annotation_controls.jsonl")
        if path.exists() and read_manifest(path)!=controls:
            raise RuntimeError("Existing annotation controls differ from frozen config")
        write_jsonl(path,controls);records.extend(controls)
    if args.split:records=[r for r in records if r["split"]==args.split]
    if args.limit:records=records[:args.limit]
    output=Path(args.output)
    jobs=read_manifest(output/"jobs.jsonl") if (output/"jobs.jsonl").exists() else []
    by_id={row["scene_id"]:row for row in jobs}
    for record in records:
        sid=record["scene_id"];directory=Path(args.view_root or config.get("view_root","data/annotation_views"))/sid
        digest=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
        if sid in by_id:
            if by_id[sid]["source_record_sha256"]!=digest:
                raise RuntimeError("Changed source record for cached annotation views: "+sid)
            if Path(by_id[sid]["bundle_path"]).exists():
                existing=json.loads(Path(by_id[sid]["bundle_path"]).read_text())
                if existing.get("display_policy","percentile_995")!=config.get("display_policy","percentile_995"):
                    raise RuntimeError("Refusing to reuse views from a different display policy")
                continue
        if (directory/"views.json").exists():
            existing=json.loads((directory/"views.json").read_text())
            if existing.get("display_policy","percentile_995")!=config.get("display_policy","percentile_995"):
                raise RuntimeError("Refusing to overwrite earlier-pilot images; choose a separate view root")
        started=time.perf_counter()
        waveform,rate,origin,truth=load_waveform(record)
        prepare_fixed_views(waveform,rate,directory,offset_s=origin,duration_s=waveform.numel()/rate,
                            display_policy=config.get("display_policy","percentile_995"))
        if record.get("annotation_control"):
            truth["scene_id"]=sid;truth["control_role"]=record["control_role"]
            write_json(Path("data/annotation_control_truth")/(sid+".json"),truth)
        by_id[sid]=dict(scene_id=sid,bundle_path=str(directory/"views.json"),
                       source_record_sha256=digest,split=record["split"],stratum=record["stratum"],
                       group_id=record.get("group_id",record.get("parent_group",sid)),
                       annotation_control=record.get("annotation_control",False),
                       control_role=record.get("control_role"),prepare_wall_seconds=time.perf_counter()-started)
        write_jsonl(output/"jobs.jsonl",list(by_id.values()))
        print("PREPARED",sid,flush=True)
        del waveform,truth;gc.collect()
    write_json(output/"preparation_status.json",dict(total_jobs=len(by_id),new_request_records=len(records),
                configuration_sha256=file_hash(args.config),annotation_origin="AI_reference",
                independent_ground_truth=False))


def selected_jobs(args):
    jobs=read_manifest(Path(args.output)/"jobs.jsonl")
    if args.split:jobs=[j for j in jobs if j["split"]==args.split]
    if args.controls_only:jobs=[j for j in jobs if j.get("annotation_control")]
    if args.scene_id:jobs=[j for j in jobs if j["scene_id"] in args.scene_id]
    if args.limit:jobs=jobs[:args.limit]
    return jobs


def run(args,config):
    spec=config["models"][args.annotator];directory=Path(args.output)/args.annotator
    directory.mkdir(parents=True,exist_ok=True)
    jobs=selected_jobs(args);completed=[];pending=[]
    for job in jobs:
        target=directory/(job["scene_id"]+".json")
        if target.exists():
            record=json.loads(target.read_text())
            if record["generation"]["model_revision"]!=spec["revision"]:
                raise RuntimeError("Refusing to mix model revisions")
            if record.get("configuration_sha256")!=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest():
                raise RuntimeError("Refusing to mix annotation configurations")
            completed.append(record)
        else:pending.append(job)
    if pending:
        model=AnnotationVLM(spec,config)
        for job in pending:
            sid=job["scene_id"];bundle=json.loads(Path(job["bundle_path"]).read_text())
            record=annotate_scene(model,sid,bundle,directory/(sid+".json"),config)
            completed.append(record)
            print("ANNOTATED",args.annotator,sid,record["annotation"]["annotation_status"],len(record["annotation"]["events"]),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    write_json(Path(args.output)/(args.annotator+"_status.json"),dict(
        annotator=args.annotator,model_id=spec["model_id"],revision=spec["revision"],
        requested_scenes=len(jobs),completed_scenes=len(completed),
        complete_schema_scenes=sum(r["annotation"]["annotation_status"]=="complete" for r in completed),
        invalid_or_partial_scenes=sum(r["annotation"]["annotation_status"]!="complete" for r in completed),
        total_output_tokens=sum(r["generation"]["output_tokens"] for r in completed),
        total_gpu_seconds=sum(r["generation"]["gpu_seconds"] for r in completed),
        total_generation_wall_seconds=sum(r["generation"]["wall_seconds"] for r in completed),
        independent_ground_truth=False,main_controller_outputs_supplied=False))


def reconcile(args,config):
    output=Path(args.output);consensus=[];agreement_rows=[];controls=[];missing=[]
    control_scores={key:[] for key in ["internvl","qwen25","consensus"]}
    for job in selected_jobs(args):
        sid=job["scene_id"];paths=[output/name/(sid+".json") for name in ["internvl","qwen25"]]
        if not all(p.exists() for p in paths):missing.append(sid);continue
        first,second=[json.loads(p.read_text()) for p in paths]
        result=reconcile_annotations(first,second,config)
        result["annotators"]=[dict(model_id=r["generation"]["model_id"],model_revision=r["generation"]["model_revision"],
                                   annotation_file=str(path),annotation_sha256=file_hash(path)) for r,path in zip([first,second],paths)]
        result["scene_interval_s"]=first["scene_interval_s"];result["stratum"]=job["stratum"]
        result["annotation_control"]=job.get("annotation_control",False)
        write_json(output/"consensus"/(sid+".json"),result)
        consensus.append(result);agreement_rows.append(result["ai_interannotator_agreement"])
        if job.get("annotation_control"):
            truth=json.loads((Path("data/annotation_control_truth")/(sid+".json")).read_text())
            row=dict(scene_id=sid,kind=truth["kind"],control_role=job["control_role"],metrics={})
            for name,events in [("internvl",first["annotation"]["events"]),("qwen25",second["annotation"]["events"]),("consensus",result["events"])]:
                scored=events if truth.get("modulation_exhaustive",False) else [e for e in events if e.get("frequency_domain")!="modulation"]
                metric=event_metrics(scored,truth["events"],frequency_tolerance_hz=4.,exhaustive=True)
                metric["excluded_unenumerated_modulation_predictions"]=len(events)-len(scored)
                row["metrics"][name]=metric;control_scores[name].append(metric)
            controls.append(row)
    write_jsonl(output/"consensus.jsonl",consensus)
    write_json(output/"agreement_summary.json",dict(
        completed_pairs=len(consensus),missing_pairs=missing,ai_event_agreement=aggregate_event_metrics(agreement_rows),
        consensus_events=sum(len(x["events"]) for x in consensus),
        disputed_events=sum(len(x["disputed_events"][side]) for x in consensus for side in ["annotator_a","annotator_b"]),
        agreement_is_not_physical_accuracy=True,independent_real_ground_truth=False,
        exhaustive_negative_regions_verified=False))
    write_json(output/"synthetic_reference_error_audit.json",dict(
        controls=controls,summary={name:aggregate_event_metrics(rows) for name,rows in control_scores.items()},
        interpretation="Exact synthetic mechanism controls; this does not establish real-data source correctness",
        controls_distinct_from_main_benchmark=True))
    print("RECONCILED",len(consensus),"pairs; missing",len(missing),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["prepare","run","reconcile"])
    parser.add_argument("--config",default="configs/ai_annotations.json")
    parser.add_argument("--output",default="results/ai_annotations")
    parser.add_argument("--view-root")
    parser.add_argument("--manifest",action="append")
    parser.add_argument("--include-controls",action="store_true")
    parser.add_argument("--annotator",choices=["internvl","qwen25"])
    parser.add_argument("--split")
    parser.add_argument("--limit",type=int)
    parser.add_argument("--controls-only",action="store_true")
    parser.add_argument("--scene-id",action="append")
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    if args.stage=="run" and args.annotator is None:parser.error("run requires --annotator")
    if not torch.cuda.is_available():raise RuntimeError("CUDA is mandatory for acoustic/reference work")
    {"prepare":prepare,"run":run,"reconcile":reconcile}[args.stage](args,config)


if __name__=="__main__":main()
