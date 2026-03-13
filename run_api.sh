#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -d ".venv" ]; then
  # Local developer path.
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH=.
python -m uvicorn src.api.server:app --host 0.0.0.0 --port "${PORT:-8000}"
