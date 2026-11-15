"""Use the user's existing SSH connection; never provision resources or push Git."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-code", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config = json.loads((ROOT / ".local/connection.json").read_text())
    ssh = ["ssh", "-i", config["key_path"], "-p", str(config["port"]), "-o", "IdentitiesOnly=yes",
           "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + config["known_hosts_path"]]
    host = config["user"] + "@" + config["host"]
    if args.sync_code:
        subprocess.run(["rsync", "-a", "--exclude=__pycache__", "-e", shlex.join(ssh),
                        *[str(ROOT / name) for name in ("src", "scripts", "tests", "preregistration")],
                        host + ":" + config["project_directory"] + "/"], check=True)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if command:
        subprocess.run(ssh + [host, "cd " + shlex.quote(config["project_directory"]) + " && " + shlex.join(command)], check=True)


if __name__ == "__main__":
    main()
