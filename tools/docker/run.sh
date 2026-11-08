#!/usr/bin/env bash
# Serve the repo and run the headless runner against this machine's GPU — inside the
# si-webgpu container or directly on a Linux host with Chrome installed.
#   tools/docker/run.sh tests | bench | info | gpuinfo | eval --expr '...'   (extra arguments go to tools/headless.mjs)
# Environment: PORT (default 8799), CHROME (default /opt/google/chrome/chrome), CHROME_FLAGS.
set -euo pipefail
cd "$(dirname "$0")/../.."
PORT=${PORT:-8799}
CHROME=${CHROME:-/opt/google/chrome/chrome}
FLAGS="${CHROME_FLAGS:-}"
# root (in a container): Chrome's sandbox needs user namespaces the container does not grant
if [ "$(id -u)" = 0 ]; then FLAGS="--no-sandbox $FLAGS"; fi
node tools/serve.mjs "$PORT" >/tmp/si-serve.log 2>&1 &
SERVER=$!
trap 'kill $SERVER' EXIT
sleep 1
node tools/headless.mjs "$@" --base "http://localhost:$PORT" --chrome "$CHROME" --flags "$FLAGS"
