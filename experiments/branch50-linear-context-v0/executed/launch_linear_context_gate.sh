#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/mo/DEV/experiments/helix-branch50-linear-context-v0
PYTHON=/home/mo/DEV/experiments/helix-branch49-5080-scaling-v0/.venv/bin/python
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
COURT="$ROOT/artifacts/linear-context-court-$STAMP.json"

cd "$ROOT"

"$PYTHON" profile_linear_context.py \
  --contexts 512 1024 2048 \
  --batch-size 1 \
  --seed 42 \
  --output "$COURT"

"$PYTHON" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if p["admission"] == "PASS" else 2)' \
  "$COURT"

"$PYTHON" run_branch50_linear_context_trial.py \
  --seq-len 1024 \
  --batch-size 6 \
  --grad-accum 7 \
  --steps 100 \
  --eval-every 20 \
  --validation-batches 4
