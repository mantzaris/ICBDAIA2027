"""Generate/check an SHA256 receipt for every retained research artifact."""
import argparse
import hashlib
import json
from pathlib import Path
import time

ROOTS=['reports','results','checkpoints','data','figures','viewer','configs','provenance']
EXCLUDE={'reports/local_artifact_receipt.json','reports/remote_artifact_receipt.json','reports/pod_shutdown.json'}


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify')
    parser.add_argument('--output',default='reports/remote_artifact_receipt.json')
    args=parser.parse_args()
    if args.verify:
        source=json.loads(Path(args.verify).read_text());missing=[];changed=[]
        for item in source['files']:
            path=Path(item['path'])
            if not path.is_file():missing.append(str(path))
            elif digest(path)!=item['sha256']:changed.append(str(path))
        result=dict(source,verified=not(missing or changed),verified_at_unix=time.time(),missing=missing,changed=changed)
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'verified':result['verified'],'files':len(result['files']),'missing':missing,'changed':changed}))
        if not result['verified']:raise SystemExit(1)
    else:
        paths=set()
        for root in ROOTS:
            paths.update(p for p in Path(root).rglob('*') if p.is_file() and p.as_posix() not in EXCLUDE
                         and not p.as_posix().startswith('data/raw/'))
        result={'pod_id':'f2n82jl2ysvf2i','created_at_unix':time.time(),'verified':False,
                'files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(paths)]}
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'files':len(paths),'bytes':sum(r['bytes'] for r in result['files']),'receipt':args.output}))


if __name__=='__main__':main()
