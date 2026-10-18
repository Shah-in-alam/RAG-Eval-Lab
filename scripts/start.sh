#!/usr/bin/env bash
# Start the API (port 8000) and the web dev server (port 5173).
set -e
cd "$(dirname "$0")/.."
python -m uvicorn api.server:app --port 8000 &
API=$!
trap 'kill $API' EXIT
[ -d web/node_modules ] || npm --prefix web install
npm --prefix web run dev
