"""Local preservation supervisor for the specifically authorized long GPU job.

Wait for the successful queue, copy to an isolated archive, verify every remote
receipt hash, check the offline viewer and GPU deliverables, then stop the pod.
An incomplete/failed queue or validation never triggers stop. This process does
not finalize manuscript prose or claim that the original scientific hypotheses
were established. No control credential leaves the pod.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "runpod_archive" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
REMOTE_ROOT = "/workspace/reading-sonar-signals"
STATUS = ROOT / "reports/automatic_preservation_status.json"
HOLD = ROOT / "reports/hold_gpu_shutdown"
LOCAL_VALIDATIONS = ["viewer/browser_check.json", "viewer/browser_check.png",
                     "reports/final_gpu_verification.json", "reports/final_remote_preservation_guard.json",
                     "reports/local_raw_scene_preservation.json"]
SSH = "ssh -S /tmp/runpod-ssh.YZ5pS8/session -p 45012 -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/tmp/runpod-ssh.YZ5pS8/known_hosts"


def status(phase, **values):
    result = dict(phase=phase, updated_at=datetime.now(timezone.utc).isoformat(), **values)
    STATUS.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def remote(script, timeout=180):
    return subprocess.run(["bash", str(ROOT / "scripts/remote.sh"), "bash -s"],
        input="set -euo pipefail\ncd " + REMOTE_ROOT + "\n" + script,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=timeout)


def run(argv, cwd=ROOT):
    subprocess.run([str(value) for value in argv], cwd=cwd, check=True)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check_hold():
    if HOLD.exists():
        raise RuntimeError("Local shutdown hold exists; do not stop. Resume this supervisor only after additional work and removing reports/hold_gpu_shutdown")


def preserve_local_scenes():
    """Copy already local inputs, binding every byte to the frozen scene manifest."""
    relative_manifest = "data/manifests/real_scenes.jsonl"
    manifest = ARCHIVE / relative_manifest
    freeze = json.loads((ARCHIVE / "configs/final_protocol_freeze.json").read_text())
    if digest(manifest) != freeze["sha256"][relative_manifest]:
        raise RuntimeError("Raw-scene manifest differs from the frozen protocol")
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len(records) != 210 or len({row["path"] for row in records}) != 210:
        raise RuntimeError("Expected the complete 210-scene raw inventory")
    inventory = []
    for row in records:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:3] != ("data", "raw", "scenes"):
            raise RuntimeError("Unexpected raw-scene input path")
        source, destination = ROOT / relative, ARCHIVE / relative
        if source.is_symlink() or digest(source) != row["sha256"]:
            raise RuntimeError("Local raw scene differs from frozen input: " + str(relative))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if digest(destination) != row["sha256"]:
            raise RuntimeError("Raw scene copy failed verification: " + str(relative))
        inventory.append({"path": str(destination), "project_relative_path": relative.as_posix(),
            "sha256": row["sha256"], "bytes": destination.stat().st_size,
            "origin": "preexisting_local_scene_verified_against_frozen_manifest"})
    proof = {"status": "passed", "checked_at": datetime.now(timezone.utc).isoformat(),
             "manifest_sha256": digest(manifest), "files": inventory,
             "scope": "All 210 real scene WAVs, including figure inputs and mixture parents. Full original downloads also remain in the working project."}
    (ARCHIVE / "reports/local_raw_scene_preservation.json").write_text(json.dumps(proof, indent=2) + "\n")
    return inventory


def main():
    probe = """.venv/bin/python - <<'PY'
from pathlib import Path
import json,hashlib,subprocess
from datetime import datetime
marker=Path('reports/remaining_study/COMPLETED.json')
launcher_path=Path('reports/remaining_study_launcher.json')
launcher=json.loads(launcher_path.read_text())
proc=Path('/proc')/str(launcher['pid'])/'stat'
alive=proc.exists() and proc.read_text().split(') ',1)[1].split()[0]!='Z'
completion=json.loads(marker.read_text()) if marker.is_file() else None
bound=bool(completion and completion['status']=='all_queued_experiments_and_exports_finished'
    and completion['script_sha256']==launcher['script_sha256']
    and datetime.fromisoformat(completion['completed_at']).timestamp()>=launcher['started_at_unix'])
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
print(json.dumps({'queue_complete':marker.is_file(),'queue_writer_alive':alive,
    'completion_matches_launcher':bound,
    'marker_sha256':hashlib.sha256(marker.read_bytes()).hexdigest() if completion else None,
    'launcher_sha256':hashlib.sha256(launcher_path.read_bytes()).hexdigest(),
    'script_matches_launcher':hashlib.sha256(Path(launcher['script']).read_bytes()).hexdigest()==launcher['script_sha256'],
    'active_gpu_process_ids':[line.strip() for line in gpu.stdout.splitlines() if line.strip()]}))
PY
"""
    while True:
        try:
            state = json.loads(remote(probe, timeout=30).stdout)
            status("waiting_for_complete_GPU_queue", **state)
            if state["queue_complete"] and not state["queue_writer_alive"] and not state["active_gpu_process_ids"]:
                if not state["completion_matches_launcher"] or not state["script_matches_launcher"]:
                    raise RuntimeError("Completion marker does not match the launched queue")
                approved_state = state
                break
        except (subprocess.SubprocessError, ValueError) as error:
            status("waiting_for_connection", error=str(error))
            state = None
        if state and not state["queue_writer_alive"] and not state["queue_complete"]:
            raise RuntimeError("GPU queue exited without completion; inspect retained phase logs and repair before restarting this supervisor")
        time.sleep(30)
    check_hold()
    status("creating_remote_receipt")
    result = remote(".venv/bin/python scripts/artifact_receipt.py\n", timeout=300)
    print(result.stdout, flush=True)
    ARCHIVE.mkdir(parents=True, exist_ok=False)
    roots = ["reports", "results", "checkpoints", "data", "figures", "viewer", "configs",
             "provenance", "annotations", "src", "scripts", "tests"]
    command = ["rsync", "-azH", "--checksum", "--exclude=data/raw/", "--exclude=__pycache__/",
               "--exclude=.pytest_cache/", "--exclude=*.pyc", "--exclude=*.pyo"]
    command += ["--include=" + name + "/***" for name in roots]
    command += ["--include=README*", "--include=requirements*.txt",
                "--include=pyproject.toml", "--include=.gitignore",
                "--include=underwater_acoustic_agent_research_plan.md", "--exclude=*", "-e", SSH,
                "root@195.26.233.13:" + REMOTE_ROOT + "/", str(ARCHIVE) + "/"]
    status("copying_to_isolated_local_archive", archive=str(ARCHIVE))
    run(command)
    status("verifying_remote_artifact_hashes")
    run([sys.executable, ARCHIVE / "scripts/artifact_receipt.py", "--verify",
         "reports/remote_artifact_receipt.json", "--output", "reports/local_artifact_receipt.json"], cwd=ARCHIVE)
    remote_paths = {item["path"] for item in json.loads((ARCHIVE / "reports/remote_artifact_receipt.json").read_text())["files"]}
    preserved_validation_paths = {}
    for relative in LOCAL_VALIDATIONS:
        if relative in remote_paths:
            preserved = ARCHIVE / "provenance/preserved_remote_validation" / relative
            preserved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ARCHIVE / relative, preserved)
            preserved_validation_paths[relative] = preserved
    status("preserving_local_raw_scene_inputs")
    local_raw_entries = preserve_local_scenes()
    status("checking_offline_viewer_and_complete_GPU_outputs")
    run(["node", ARCHIVE / "scripts/check_viewer_browser.mjs", ARCHIVE / "viewer"])
    run([sys.executable, ARCHIVE / "scripts/verify_final_deliverables.py", "--project-root", ARCHIVE,
         "--gpu-only", "--output", "reports/final_gpu_verification.json"])
    # Copy generated material into the working project, retaining local prose
    # and administrative reports. The isolated archive stays unchanged.
    for name in ["results", "checkpoints", "data", "figures", "viewer", "configs", "provenance", "annotations"]:
        source = ARCHIVE / name
        if source.is_dir():
            (ROOT / name).mkdir(exist_ok=True)
            options = ["--exclude=raw/"] if name == "data" else []
            run(["rsync", "-a", *options, str(source) + "/", str(ROOT / name) + "/"])
    run(["rsync", "-a", "--ignore-existing", str(ARCHIVE / "reports") + "/", str(ROOT / "reports") + "/"])
    for name in ["study_phases", "remaining_study", "synthetic_ablation_phases"]:
        source = ARCHIVE / "reports" / name
        if source.is_dir():
            (ROOT / "reports" / name).mkdir(exist_ok=True)
            run(["rsync", "-a", str(source) + "/", str(ROOT / "reports" / name) + "/"])
    shutil.copy2(ARCHIVE / "reports/final_gpu_verification.json", ROOT / "reports/final_gpu_verification.json")
    check_hold()
    # Reject any new, changed, or removed remote artifact during the copy/check
    # interval. Receipt creation timestamps are not research-artifact changes.
    status("rechecking_remote_inventory_and_GPU_idle_state")
    remote(".venv/bin/python scripts/artifact_receipt.py\n", timeout=300)
    current_receipt = json.loads(remote("cat reports/remote_artifact_receipt.json\n", timeout=60).stdout)
    original_receipt = json.loads((ARCHIVE / "reports/remote_artifact_receipt.json").read_text())
    inventory = lambda value: sorted((item["path"], item["bytes"], item["sha256"]) for item in value["files"])
    if inventory(current_receipt) != inventory(original_receipt):
        raise RuntimeError("Remote artifacts changed during preservation; keep the pod running and start a fresh archive after writers finish")
    final_state = json.loads(remote(probe, timeout=30).stdout)
    if final_state != approved_state:
        raise RuntimeError("Queue identity or GPU activity changed during preservation")
    guard = {"checked_at": datetime.now(timezone.utc).isoformat(), "remote_inventory_unchanged": True,
             "files": len(current_receipt["files"]), "queue_and_GPU_state": final_state}
    (ARCHIVE / "reports/final_remote_preservation_guard.json").write_text(json.dumps(guard, indent=2) + "\n")
    receipt = json.loads((ARCHIVE / "reports/local_artifact_receipt.json").read_text())
    if not receipt["verified"]:
        raise RuntimeError("Archive verification did not pass")
    for entry in receipt["files"]:
        entry["remote_relative_path"] = entry["path"]
        entry["path"] = str(preserved_validation_paths.get(entry["path"], ARCHIVE / entry["path"]))
        entry["origin"] = "remote_receipt_verified_copy"
    for relative in LOCAL_VALIDATIONS:
        path = ARCHIVE / relative
        receipt["files"].append({"path": str(path), "bytes": path.stat().st_size,
            "sha256": digest(path), "origin": "local_administrative_validation"})
    receipt["files"].extend(local_raw_entries)
    receipt.update(archive=str(ARCHIVE), remote_receipt_sha256=digest(ARCHIVE / "reports/remote_artifact_receipt.json"),
                   manuscript_finalization="Separate CPU-only work may continue after GPU stop")
    stop_receipt = ROOT / "reports/local_artifact_receipt.json"
    stop_receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    status("preserved_and_verified_requesting_authorized_stop", files=len(receipt["files"]))
    check_hold()
    run([sys.executable, ROOT / "scripts/pod_control.py", "stop", "--receipt", stop_receipt])
    stopped = json.loads((ROOT / "reports/pod_shutdown.json").read_text())
    if not stopped["confirmed_stopped"]:
        raise RuntimeError("API did not confirm the stopped state")
    status("GPU_finished_preserved_and_pod_stopped", archive=str(ARCHIVE),
           confirmed_stopped=True, manuscript_finalization="Continue locally; no submission performed")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        status("supervisor_needs_attention_no_unverified_stop", error=str(error))
        raise
