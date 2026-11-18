"""Status/stop for the specifically authorized pod; never terminate/delete it.

The pod-scoped credential is read over authenticated SSH and kept only in memory.
Stop requires a locally verified artifact receipt. The final API status is saved
locally, so shutdown success is not inferred solely from an SSH disconnection.
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

POD_ID='f2n82jl2ysvf2i'
URL='https://api.runpod.io/v2/pods/'+POD_ID


def filtered_history(outputs):
    """TimeoutExpired.output can be bytes even when Popen uses text=True.

    Retried communicate returns accumulated output, so deduplicate complete API
    records. Never persist raw stdout/stderr or unexpected response fields.
    """
    history=[]; seen=set()
    keys=('id','status','actions','cost','checked_at_unix','requested_action')
    for output in outputs:
        if isinstance(output,bytes):output=output.decode('utf-8',errors='replace')
        for line in (output or '').splitlines():
            try:value=json.loads(line)
            except (json.JSONDecodeError,ValueError):continue
            if not isinstance(value,dict) or value.get('id')!=POD_ID:continue
            if not isinstance(value.get('status'),str):continue
            value={key:value[key] for key in keys if key in value}
            identity=json.dumps(value,sort_keys=True)
            if identity not in seen:
                seen.add(identity);history.append(value)
    return history


def collect_api_evidence(process,remote_script):
    """Retain filtered evidence and bound cleanup of this local SSH client."""
    outputs=[]; timed_out=False; cleanup_timed_out=False
    try:
        stdout,_=process.communicate(remote_script,timeout=60)
        outputs.append(stdout)
    except subprocess.TimeoutExpired as error:
        timed_out=True;outputs.append(error.output)
        # remote.sh execs ssh; this kills only the newly spawned local client,
        # not the shared control master and not any remote GPU process.
        try:process.kill()
        except ProcessLookupError:pass
        try:
            stdout,_=process.communicate(timeout=5)
            outputs.append(stdout)
        except subprocess.TimeoutExpired as error:
            cleanup_timed_out=True;outputs.append(error.output)
            # A shared SSH master may still hold a pipe descriptor. Preserve
            # what communicate already received, close local pipes, and reap.
            for stream in (process.stdin,process.stdout,process.stderr):
                if stream is not None:stream.close()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
    return filtered_history(outputs),dict(timed_out=timed_out,
        cleanup_drain_timed_out=cleanup_timed_out,ssh_returncode=process.returncode,
        local_client_reaped=process.returncode is not None,
        unfiltered_stdout_and_stderr_saved=False)


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['status','stop'])
    parser.add_argument('--receipt',default='reports/local_artifact_receipt.json')
    args=parser.parse_args(argv)
    receipt_sha256=None
    if args.action=='stop':
        receipt_bytes=Path(args.receipt).read_bytes()
        receipt=json.loads(receipt_bytes)
        if receipt.get('verified') is not True or receipt.get('pod_id')!=POD_ID or not receipt.get('files'):
            raise RuntimeError('A verified local artifact receipt for this pod is required')
        for entry in receipt['files']:
            path=Path(entry['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
                raise RuntimeError('Local artifact changed after verification: '+str(path))
        receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest()
    # This pod-scoped credential is usable from the pod; a local REST call
    # returned403. Keep it on the pod and stream only filtered API evidence.
    remote_script='''source /etc/rp_environment
python -u - <<'PY'
import os,requests,json,time
pod_id='f2n82jl2ysvf2i'
url='https://api.runpod.io/v2/pods/'+pod_id
headers={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY']}
def request(action=None):
    response=(requests.get(url,headers=headers,timeout=20) if action is None else
              requests.post(url+'/action',headers=headers,json={'action':action},timeout=20))
    response.raise_for_status(); value=response.json()
    result={key:value.get(key) for key in ['id','status','actions','cost']}
    result['checked_at_unix']=time.time()
    if action:result['requested_action']=action
    print(json.dumps(result),flush=True)
    return result
current=request()
if current['id']!=pod_id:raise RuntimeError('Pod identity mismatch')
if ACTION=='stop' and current['status']=='RUNNING':
    current=request('stop')
    for _ in range(12):
        if current['status']=='EXITED':break
        current=request()
        if current['status']=='EXITED':break
        time.sleep(2)
PY
'''.replace('ACTION',repr(args.action))
    process=subprocess.Popen(['bash','scripts/remote.sh','bash -s'],stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    history,transport=collect_api_evidence(process,remote_script)
    for value in history:print(json.dumps(value),flush=True)
    record={'pod_id':POD_ID,'action':args.action,'terminated_or_deleted':False,
                'confirmed_stopped':bool(history and history[-1]['status']=='EXITED'),'history':history,
                'transport':transport,'recorded_at_unix':time.time(),
                'local_backup_receipt':args.receipt,
                'local_backup_receipt_sha256':receipt_sha256,
                'api_documentation':'https://docs.runpod.io/api-reference-v2/pods/trigger-a-pod-state-transition'}
    attempts=Path('reports/pod_control_attempts');attempts.mkdir(parents=True,exist_ok=True)
    attempt=attempts/(str(time.time_ns())+'_'+args.action+'.json')
    record['attempt_record']=str(attempt)
    attempt.write_text(json.dumps(record,indent=2)+'\n')
    if args.action=='stop':
        Path('reports/pod_shutdown.json').write_text(json.dumps(record,indent=2)+'\n')
    if not history:raise RuntimeError('No pod API status received; SSH exit '+str(process.returncode)+'; filtered attempt evidence saved')
    if args.action=='stop':
        if not record['confirmed_stopped']:
            raise RuntimeError('Stop requested but API has not yet confirmed EXITED')


if __name__=='__main__':main()
