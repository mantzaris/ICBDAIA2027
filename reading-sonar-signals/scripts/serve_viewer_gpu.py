#!/usr/bin/env python3
"""Optional localhost CUDA reinspection for the offline research viewer.

Never exposes a public listener, runs a model, or changes frozen outputs. Only
selected bundle scenes and bounded acoustic/modulation inspections are accepted.
The index page and saved case replay remain usable after this process stops.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import sys
import threading
from urllib.parse import unquote, urlsplit
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from build_interactive_viewer import export_view_cuda, write_json, digest


def validate_action(action, interval, sample_rate):
    if not isinstance(action,dict) or action.get("name") not in {"inspect_region","inspect_modulation"}:
        raise ValueError("Only bounded region or modulation inspection is available")
    bounds=action.get("time_s")
    if not isinstance(bounds,list) or len(bounds)!=2 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in bounds):
        raise ValueError("Specify two finite original-recording time coordinates")
    if not interval[0]<=bounds[0]<bounds[1]<=interval[1]+1e-6 or bounds[1]-bounds[0]>30.00001:
        raise ValueError("Inspection must lie within the selected thirty-second scene")
    window=action.get("window_s")
    if window not in {.25,1.,4.} or window>bounds[1]-bounds[0]:
        raise ValueError("Window must be 0.25, 1, or 4 seconds and fit the selected interval")
    clean={"name":action["name"],"time_s":bounds,"window_s":window}
    if action["name"]=="inspect_region":
        frequencies=action.get("frequency_hz")
        if not isinstance(frequencies,list) or len(frequencies)!=2 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in frequencies):
            raise ValueError("Specify two finite acoustic frequency bounds")
        if not 0<=frequencies[0]<frequencies[1]<=min(1800.,.45*sample_rate):
            raise ValueError("Acoustic frequency bounds exceed the available LOFAR band")
        clean["frequency_hz"]=frequencies
    else:
        carrier=action.get("carrier_band_hz")
        if carrier not in ([500,2000],[2000,5000]):
            raise ValueError("Carrier must be 500–2000 Hz or 2000–5000 Hz")
        if carrier[1]>sample_rate/2-100:
            raise ValueError("Requested carrier is unavailable at this sample rate")
        clean["carrier_band_hz"]=carrier
    return clean


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=ROOT)
    parser.add_argument("--viewer",type=Path,default=Path("viewer"))
    parser.add_argument("--port",type=int,default=8766)
    parser.add_argument("--max-calls-per-case",type=int,default=6)
    args=parser.parse_args()
    if not 1<=args.max_calls_per_case<=12:
        raise ValueError("Bounded reinspection allows 1–12 calls per case")
    root=args.project_root.resolve();directory=(root/args.viewer).resolve()
    payload=json.loads((directory/"viewer_bundle.json").read_text())
    cases={case["scene_id"]:case for case in payload["cases"]}
    # Model loading and numerical CPU fallbacks are deliberately absent.
    import torch
    from sonar.signal import require_cuda
    from sonar.data import load_waveform
    from sonar.evidence import AcousticTools
    require_cuda();os.chdir(root)
    lock=threading.Lock();counts={sid:0 for sid in cases}

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self,*pos,**kw):
            super().__init__(*pos,directory=str(directory),**kw)

        def send_json(self,status,value):
            content=json.dumps(value,allow_nan=False).encode()
            self.send_response(status);self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(content)));self.send_header("Cache-Control","no-store")
            self.end_headers();self.wfile.write(content)

        def allowed_host(self):
            host=self.headers.get("Host","").split(":")[0]
            return host in {"127.0.0.1","localhost"}

        def do_GET(self):
            if not self.allowed_host():
                self.send_json(403,{"error":"Only localhost access is permitted"});return
            path=urlsplit(self.path).path
            if path=="/api/status":
                self.send_json(200,{"service":"sonar-local-cuda-viewer","cuda":True,
                    "device":torch.cuda.get_device_name(),"max_calls_per_case":args.max_calls_per_case,
                    "benchmark_mutation":False});return
            resolved=(directory/unquote(path).lstrip("/")).resolve()
            if resolved!=directory and directory not in resolved.parents:
                self.send_json(403,{"error":"Path is outside the local viewer"});return
            super().do_GET()

        def do_POST(self):
            if not self.allowed_host() or urlsplit(self.path).path!="/api/reinspect":
                self.send_json(403,{"error":"Unavailable local endpoint"});return
            origin=self.headers.get("Origin")
            if origin and urlsplit(origin).hostname not in {"127.0.0.1","localhost"}:
                self.send_json(403,{"error":"Cross-origin reinspection is disabled"});return
            try:
                size=int(self.headers.get("Content-Length","0"))
                if not 0<size<=8192:raise ValueError("Request size is invalid")
                if self.headers.get("Content-Type","").split(";")[0]!="application/json":raise ValueError("JSON is required")
                request=json.loads(self.rfile.read(size))
                sid=request.get("scene_id")
                if sid not in cases:raise ValueError("Scene is not part of this selected local bundle")
                case=cases[sid]
                action=validate_action(request.get("action"),case["time_s"],case["record"]["sample_rate"])
            except (ValueError,TypeError,KeyError,json.JSONDecodeError) as error:
                self.send_json(400,{"error":str(error)});return
            if not lock.acquire(blocking=False):
                self.send_json(429,{"error":"One CUDA reinspection is already running"});return
            try:
                if counts[sid]>=args.max_calls_per_case:
                    self.send_json(429,{"error":"This case has exhausted its bounded exploratory call allowance"});return
                counts[sid]+=1
                request_id=uuid.uuid4().hex[:12]
                folder=directory/"reinspections"/sid/request_id;folder.mkdir(parents=True)
                audit={"request_id":request_id,"scene_id":sid,"action":action,
                    "created_at":datetime.now(timezone.utc).isoformat(),"scope":"exploratory_only_excluded_from_frozen_benchmark",
                    "bundle_sha256":digest(directory/"viewer_bundle.json"),"script_sha256":digest(Path(__file__)),
                    "signal_code_sha256":digest(root/"src/sonar/signal.py"),"evidence_code_sha256":digest(root/"src/sonar/evidence.py")}
                write_json(folder/"request.json",audit)
                record=dict(case["record"])
                if case.get("audio"):
                    audio=(directory/case["audio"]["url"]).resolve()
                    if directory not in audio.parents or digest(audio)!=case["audio"]["sha256"]:
                        raise ValueError("Bundled source audio provenance check failed")
                    record["path"]=str(audio)
                    audit["audio_sha256"]=case["audio"]["sha256"]
                waveform,rate,origin,_=load_waveform(record)
                acoustic=AcousticTools(waveform,rate,sid,offset_s=origin)
                before,after=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                before.record();result=acoustic.execute(action);after.record();torch.cuda.synchronize()
                exported=None
                if result.get("id") in acoustic.latest_views:
                    exported=export_view_cuda(acoustic.latest_views[result["id"]],folder,"exploratory")
                    for raster in exported["rasters"].values():
                        raster["url"]=Path(raster["path"]).relative_to(directory).as_posix()
                response=dict(audit,result=result,view=exported,gpu_seconds=before.elapsed_time(after)/1000,
                              remaining_calls=args.max_calls_per_case-counts[sid],numeric_device="cuda")
                write_json(folder/"result.json",response)
                del acoustic,waveform
                self.send_json(200,response)
            except Exception as error:
                if "folder" in locals():write_json(folder/"failure.json",{"error":str(error),"scope":"exploratory_only"})
                self.send_json(500,{"error":str(error),"scope":"exploratory_only"})
            finally:
                lock.release()

    server=ThreadingHTTPServer(("127.0.0.1",args.port),Handler)
    print(json.dumps({"url":f"http://127.0.0.1:{args.port}/","scope":"local_only_exploratory","cuda":torch.cuda.get_device_name()}),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=="__main__":main()
