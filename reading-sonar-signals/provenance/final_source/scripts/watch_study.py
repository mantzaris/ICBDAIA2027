"""Read-only local progress monitor for the long remote queue and preservation."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
REMOTE = r"""cd /workspace/reading-sonar-signals
.venv/bin/python - <<'PY'
from pathlib import Path
import json,time
def read(path):return json.loads(Path(path).read_text())
outer=read('reports/remaining_study/active.json')
phase=outer['phase']; started=outer['started_at_unix']; run=None
if phase.startswith('original_'):
 active=read('reports/study_phases/active.json');phase=active['phase'];started=active['started_at_unix']
 if phase in {'development','main','ablation_inference','budget2','budget4','robustness','sensitivity'}:run='results/'+phase
elif phase=='synthetic_ablation_supplement':run='results/synthetic_ablation_inference'
elif phase=='capped_ablation_inference':run='results/ablation_inference_capped'
result={'phase':phase,'elapsed_minutes':round((time.time()-started)/60,1),'queue_complete':Path('reports/remaining_study/COMPLETED.json').is_file()}
if run and Path(run,'run.json').exists():
 config=read(Path(run,'run.json'));latest={};ledger=Path(run,'ledger.jsonl')
 reuse=Path(run,'reuse_provenance.json')
 if reuse.exists():
  for row in read(reuse)['reused_episodes']:
   latest[(row['scene_id'],row['method'])]={'status':'complete'}
 if ledger.exists():
  for line in ledger.read_text().splitlines():
   try:row=json.loads(line)
   except json.JSONDecodeError:continue
   latest[(row['scene_id'],row['method'])]=row
 result.update(finished=len(latest),expected=config['scene_count']*len(config['methods']),execution_failures=sum(row['status']=='failed' for row in latest.values()))
print(json.dumps(result))
PY
"""


def main():
    while True:
        status_path = ROOT / "reports/automatic_preservation_status.json"
        try:
            preservation = json.loads(status_path.read_text()) if status_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            preservation = {"phase": "preservation_status_read_retry"}
        if preservation.get("phase") == "GPU_finished_preserved_and_pod_stopped":
            print(json.dumps({"phase": "GPU_work_preserved_pod_stopped", "archive": preservation["archive"]}), flush=True)
            break
        if preservation.get("phase") == "supervisor_needs_attention_no_unverified_stop":
            print(json.dumps(preservation), flush=True)
            break
        try:
            process = subprocess.run(["bash", str(ROOT / "scripts/remote.sh"), "bash -s"],
                input=REMOTE, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
            value = json.loads(process.stdout) if process.returncode == 0 else {"phase": "connection_check_needed", "returncode": process.returncode}
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
            value = {"phase": "progress_read_retry", "error_type": type(error).__name__, "error": str(error)}
        value["observed_at"] = datetime.now(timezone.utc).isoformat()
        value["preservation_phase"] = preservation.get("phase")
        (ROOT / "reports/progress_latest.json").write_text(json.dumps(value, indent=2) + "\n")
        with (ROOT / "reports/progress_history.jsonl").open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        print(json.dumps(value), flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
