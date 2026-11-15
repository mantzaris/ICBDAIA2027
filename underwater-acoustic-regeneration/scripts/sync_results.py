"""Pull phase artifacts over supplied direct SSH; no Git pushes or remote deletion."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--connection", type=Path, default=ROOT / ".local/connection.json")
    parser.add_argument("--include-processed", action="store_true")
    args = parser.parse_args()
    connection = json.loads(args.connection.read_text())
    ssh = ["ssh", "-i", connection["key_path"], "-p", str(connection["port"]),
           "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
           "-o", "UserKnownHostsFile=" + connection["known_hosts_path"]]
    destination = connection["user"] + "@" + connection["host"]
    directories = ["manifests", "provenance", "reports", "logs", "results", "checkpoints"]
    if args.include_processed:
        directories.append("data/processed")
    for directory in directories:
        remote_path = connection["project_directory"] + "/./" + directory + "/"
        subprocess.run(["rsync", "-a", "--relative", "--partial", "--exclude=source_retrieval.json",
                        "--exclude=repositories.json", "--exclude=status.json", "-e", shlex.join(ssh),
                        destination + ":" + remote_path, str(ROOT) + "/"], check=True)
    subprocess.run(["rsync", "-a", "-e", shlex.join(ssh), destination + ":" + connection["project_directory"] +
                    "/requirements.gpu.lock", str(ROOT) + "/"], check=True)
    # Checkpoints are immutable within each completed phase. Hash every downloaded
    # checkpoint and compare with the remote copy; retry a changed active file later.
    paths = [str(path.relative_to(ROOT)) for path in (ROOT / "checkpoints").rglob("*.pt")]
    if paths:
        from acoustic_regeneration.registry import file_hash
        code = "import hashlib,json; from pathlib import Path; "
        code += "paths=" + repr(paths) + "; "
        code += "print(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}))"
        command = "cd " + shlex.quote(connection["project_directory"]) + " && .venv/bin/python -c " + shlex.quote(code)
        response = subprocess.run(ssh + [destination, command], capture_output=True, text=True, check=True)
        remote_hashes = json.loads(response.stdout)
        local_hashes = {path: file_hash(ROOT / path) for path in paths}
        if local_hashes != remote_hashes:
            raise RuntimeError("Checkpoint changed during sync or hash mismatch; repeat phase sync")
        (ROOT / "manifests/verified_checkpoint_backup.json").write_text(json.dumps(local_hashes, indent=2) + "\n")
        print("Verified", len(paths), "checkpoint backups")


if __name__ == "__main__":
    main()
