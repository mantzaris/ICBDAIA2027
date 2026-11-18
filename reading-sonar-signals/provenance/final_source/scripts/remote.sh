#!/usr/bin/env bash
set -euo pipefail
# Run from the project root. Credentials remain in the user's SSH agent.
runpod_socket=${RUNPOD_SSH_SOCKET:-/tmp/runpod-ssh.YZ5pS8/session}
runpod_known_hosts=${RUNPOD_KNOWN_HOSTS:-/tmp/runpod-ssh.YZ5pS8/known_hosts}
exec ssh -S "$runpod_socket" -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile="$runpod_known_hosts" -p 45012 root@195.26.233.13 "$@"
