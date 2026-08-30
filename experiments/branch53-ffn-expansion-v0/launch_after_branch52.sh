#!/usr/bin/env bash
set -euo pipefail

readonly BRANCH52_PID="1350401"
readonly BRANCH52_RUN_ROOT="/home/mo/DEV/experiments/helix-branch52-activation-checkpointing-v0/artifacts/quality-vram-ablation-v0/branch52-ablation-activation-checkpointing-b8-a8-400m-s512-b8-a8-t400000000-20260830T165308Z"
readonly BRANCH52_TERMINAL="${BRANCH52_RUN_ROOT}/terminal.json"
readonly BRANCH52_COMMAND_MARKER="activation-checkpointing-b8-a8-400m"
readonly BRANCH53_ROOT="/home/mo/DEV/experiments/helix-branch53-ffn-expansion-v0"
readonly SOURCE_ROOT="${BRANCH53_ROOT}/source"
readonly RUNNER="${SOURCE_ROOT}/experiments/branch51-quality-vram-v0/run_branch51_quality_vram_ablation.py"
readonly PROMOTION="${SOURCE_ROOT}/experiments/branch53-ffn-expansion-v0/evidence/ffn3p0-promotion.json"
readonly PYTHON="/home/mo/DEV/experiments/helix-branch49-5080-scaling-v0/.venv/bin/python"
readonly QUEUE_ROOT="${BRANCH53_ROOT}/queue"
readonly LOG_ROOT="${BRANCH53_ROOT}/logs"
readonly CLAIM="${QUEUE_ROOT}/launch.claimed"
readonly PID_FILE="${QUEUE_ROOT}/branch53.pid"
readonly LOG_FILE="${LOG_ROOT}/branch53-ffn3p0-b8a8-400m.log"

mkdir -p "${QUEUE_ROOT}" "${LOG_ROOT}"

if [[ -r "/proc/${BRANCH52_PID}/cmdline" ]]; then
  current_command="$(tr '\0' ' ' < "/proc/${BRANCH52_PID}/cmdline")"
  if [[ "${current_command}" != *"${BRANCH52_COMMAND_MARKER}"* ]]; then
    printf '%s REFUSED pid_reuse pid=%s\n' "$(date --iso-8601=seconds)" "${BRANCH52_PID}" >> "${QUEUE_ROOT}/queue.log"
    exit 1
  fi
  exit 0
fi

if [[ -d "${CLAIM}" ]]; then
  exit 0
fi

if [[ ! -f "${BRANCH52_TERMINAL}" ]]; then
  exit 0
fi

jq -e '
  .status == "PASS" and
  .data_offset.causal_targets_seen >= 400000000 and
  .nonfinite_events == 0 and
  .skipped_batches == 0 and
  .checkpoint_health == "PASS" and
  .checkpoint_readback == "PASS" and
  .best_checkpoint_readback == "PASS" and
  .mlflow_health == "PASS" and
  .numerical_health == "PASS"
' "${BRANCH52_TERMINAL}" >/dev/null

mkdir "${CLAIM}"

if setsid env \
  PYTHONUNBUFFERED=1 \
  HELIX_BRANCH53_RUN_ROOT="${BRANCH53_ROOT}" \
  "${PYTHON}" "${RUNNER}" \
    --ablation-id ffn3p0-b8a8-400m \
    --batch-size 8 \
    --grad-accum 8 \
    --target-causal-targets 400000000 \
    --eval-every 100 \
    --checkpoint-every 500 \
    --validation-batches 16 \
    --learning-rate 0.00015 \
    --warmup-microbatches 2280 \
    --scheduler-policy linear_warmup_then_constant \
    --scheduler-min-lr-ratio 1.0 \
    --weight-decay 0.05 \
    --grad-clip 1.0 \
    --dropout 0.05 \
    --attention-dropout 0.05 \
    --ffn-expansion 3.0 \
    --n-loops 3 \
    --activation-checkpointing \
    --promotion-manifest "${PROMOTION}" \
    >> "${LOG_FILE}" 2>&1 &
then
  child_pid="$!"
  printf '%s\n' "${child_pid}" > "${PID_FILE}"
  printf '%s LAUNCHED pid=%s log=%s\n' "$(date --iso-8601=seconds)" "${child_pid}" "${LOG_FILE}" >> "${QUEUE_ROOT}/queue.log"
else
  rmdir "${CLAIM}"
  printf '%s LAUNCH_FAILED\n' "$(date --iso-8601=seconds)" >> "${QUEUE_ROOT}/queue.log"
  exit 1
fi
