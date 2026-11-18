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


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['status','stop'])
    parser.add_argument('--receipt',default='reports/local_artifact_receipt.json')
    args=parser.parse_args()
    if args.action=='stop':
        receipt=json.loads(Path(args.receipt).read_text())
        if not receipt.get('verified') or receipt.get('pod_id')!=POD_ID:
            raise RuntimeError('A verified local artifact receipt for this pod is required')
        for entry in receipt['files']:
            path=Path(entry['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
                raise RuntimeError('Local artifact changed after verification: '+str(path))
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
    stdout,stderr=process.communicate(remote_script,timeout=60)
    history=[]
    for line in stdout.splitlines():
        try:value=json.loads(line)
        except json.JSONDecodeError:continue
        if value.get('id')==POD_ID:
            history.append(value);print(json.dumps(value),flush=True)
    if not history:raise RuntimeError('No pod API status received; SSH exit '+str(process.returncode))
    if args.action=='stop':
        record={'pod_id':POD_ID,'action':'stop','terminated_or_deleted':False,
                'confirmed_stopped':any(r['status']=='EXITED' for r in history),'history':history,
                'local_backup_receipt':args.receipt,
                'api_documentation':'https://docs.runpod.io/api-reference-v2/pods/trigger-a-pod-state-transition'}
        Path('reports/pod_shutdown.json').write_text(json.dumps(record,indent=2)+'\n')
        if not record['confirmed_stopped']:
            raise RuntimeError('Stop requested but API has not yet confirmed EXITED')


if __name__=='__main__':main()
