#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 PAIR_ROOT" >&2
  exit 64
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
RUNTIME_ROOT=/home/mo/DEV/experiments/helix-lighteval-runtime-v0
PYTHON="$RUNTIME_ROOT/.venv/bin/python"
PAIR_ROOT="$1"

if [[ -e "$PAIR_ROOT" ]]; then
  echo "REFUSED: pair root already exists: $PAIR_ROOT" >&2
  exit 2
fi

mkdir -p "$PAIR_ROOT/logs"
export HF_HOME="$RUNTIME_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$RUNTIME_ROOT/cache/datasets"
export TRANSFORMERS_CACHE="$RUNTIME_ROOT/cache/transformers"
export NLTK_DATA="$RUNTIME_ROOT/cache/nltk"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=0

cd "$REPOSITORY_ROOT"

run_checkpoint() {
  local checkpoint_id="$1"
  "$PYTHON" "$SCRIPT_DIR/run_evaluation.py" \
    --checkpoint-id "$checkpoint_id" \
    --mode full \
    --installed-freeze "$RUNTIME_ROOT/installed.freeze.txt" \
    --output-dir "$PAIR_ROOT/$checkpoint_id" \
    2>&1 | tee "$PAIR_ROOT/logs/$checkpoint_id.log"
}

run_checkpoint branch50_lr1p5e4_ffn2p5_full_best
run_checkpoint branch53_lr2e4_ffn2p5_full_best
"$PYTHON" "$SCRIPT_DIR/compare_results.py" --pair-root "$PAIR_ROOT" \
  2>&1 | tee "$PAIR_ROOT/logs/comparison.log"

find "$PAIR_ROOT" -type f \
  ! -path "$PAIR_ROOT/MANIFEST.sha256" \
  ! -path "$PAIR_ROOT/projections/*" \
  -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$PAIR_ROOT/MANIFEST.sha256"
echo "LIGHTEVAL_LOCAL_PACKET=PASS"
"$PYTHON" "$SCRIPT_DIR/project_mlflow.py" \
  --pair-root "$PAIR_ROOT" \
  --tracking-uri "${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}" \
  2>&1 | tee "$PAIR_ROOT/logs/mlflow_projection.log"
find "$PAIR_ROOT/projections" -type f \
  ! -path "$PAIR_ROOT/projections/MANIFEST.sha256" \
  -print0 | LC_ALL=C sort -z | \
  xargs -0 sha256sum > "$PAIR_ROOT/projections/MANIFEST.sha256"
echo "LIGHTEVAL_PAIRED_FULL=PASS"
