#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="$REPO_ROOT/experiments/branch50-linear-context-v0/run_branch50_quality_promotion.py"
PYTHON="${HELIX_PYTHON:-/home/mo/DEV/experiments/helix-branch49-5080-scaling-v0/.venv/bin/python}"
RUN_ROOT="${HELIX_BRANCH50_RUN_ROOT:-/home/mo/DEV/experiments/helix-branch50-linear-context-v0}"
TARGETS="${HELIX_TARGET_CAUSAL_TARGETS:-100000000}"
SEED="${HELIX_SEED:-42}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_ROOT="$RUN_ROOT/logs/quality-promotion-v0"

mkdir -p "$LOG_ROOT"

export HELIX_BRANCH50_RUN_ROOT="$RUN_ROOT"
export PYTHONHASHSEED=0

wait_for_gpu() {
  local attempt
  for attempt in $(seq 1 30); do
    if nvidia-smi -L >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "REFUSED: RTX 5080 driver unavailable after 30 bounded probes" >&2
  return 74
}

wait_for_gpu
"$PYTHON" "$RUNNER" \
  --seq-len 512 \
  --batch-size 12 \
  --grad-accum 7 \
  --target-causal-targets "$TARGETS" \
  --eval-every 100 \
  --checkpoint-every 500 \
  --validation-base-blocks 48 \
  --seed "$SEED" \
  2>&1 | tee "$LOG_ROOT/seq512-seed${SEED}-${STAMP}.log"

wait_for_gpu
"$PYTHON" "$RUNNER" \
  --seq-len 1024 \
  --batch-size 6 \
  --grad-accum 7 \
  --target-causal-targets "$TARGETS" \
  --eval-every 100 \
  --checkpoint-every 500 \
  --validation-base-blocks 48 \
  --seed "$SEED" \
  2>&1 | tee "$LOG_ROOT/seq1024-seed${SEED}-${STAMP}.log"
