"""Copy only completed GPU-agent episodes, then verify bytes against the remote snapshot."""
import argparse
import hashlib
import json
import shlex
import subprocess
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
    cfg=json.loads((ROOT/'.local/gpu_agent_connection.json').read_text())
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','StrictHostKeyChecking=yes',
         '-o','UserKnownHostsFile='+cfg['known_hosts_path'],'-p',str(cfg['port'])]
    host=cfg['user']+'@'+cfg['host'];remote=cfg['project_directory']
    code='''import hashlib,json
from pathlib import Path
files=[]
completion={};progress={}
for run in ['gpu_agent_benchmark','gpu_agent_guard']:
    base=Path('results')/run
    for phase in ['smoke','evaluation']:
        directory=base/phase
        for p in directory.glob('*__*.json'):
            q=p.with_suffix('.jsonl')
            item=json.loads(p.read_text())
            assert hashlib.sha256(q.read_bytes()).hexdigest()==item['events_sha256']
            files.extend([p,q])
        for name in ['completion.json','run_definition.json']:
            if (directory/name).exists(): files.append(directory/name)
    for name in ['evaluation_launch.json']:
        if (base/name).exists(): files.append(base/name)
    queue=base/'queue.json'
    if queue.exists() and json.loads(queue.read_text())['status'] not in ['waiting_for_primary','running']:
        files.append(queue)
    completion[run]=(base/'evaluation/completion.json').exists()
    if (base/'evaluation/progress.json').exists():
        progress[run]=json.loads((base/'evaluation/progress.json').read_text())
print(json.dumps({'complete':all(completion.values()),'phase_completion':completion,'progress':progress,
 'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}))
'''
    command='cd '+shlex.quote(remote)+' && python3 -c '+shlex.quote(code)
    snapshot=json.loads(subprocess.check_output(ssh+[host,command],text=True))
    if args.final:assert snapshot['complete'],'Evaluation is still running'
    filelist=ROOT/'.cache/gpu_agent_backup_files.txt';filelist.parent.mkdir(parents=True,exist_ok=True)
    filelist.write_text('\n'.join(snapshot['files'])+'\n')
    subprocess.run(['rsync','-a','--files-from='+str(filelist),'-e',shlex.join(ssh),host+':'+remote+'/',str(ROOT)+'/'],check=True)
    for name,expected in snapshot['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    report={'checked_utc':datetime.now(timezone.utc).isoformat(),'evaluation_complete':snapshot['complete'],
            'phase_completion':snapshot['phase_completion'],
            'completed_episodes':{run:p['completed_episodes'] for run,p in snapshot['progress'].items()},'verified_files':len(snapshot['files']),
            'files':snapshot['files'],'partial_live_event_files_copied':False}
    (ROOT/'manifests/verified_gpu_agent_backup.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))


if __name__=='__main__':main()
