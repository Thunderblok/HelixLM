#!/usr/bin/env bash

set -u

artifact_root="${BRANCH50_ABLATION_ARTIFACT_ROOT:-/home/mo/DEV/experiments/helix-branch50-linear-context-v0/artifacts/ablation-300m-v0}"
log_path="${BRANCH50_ABLATION_WATCHDOG_LOG:-${artifact_root}/watchdog.log}"
reminder_path="${BRANCH50_ABLATION_REMINDER_PATH:-/tmp/BRANCH50_ABLATION_LEDGER_REVIEW_DUE}"
fresh_spool_seconds="${BRANCH50_WATCHDOG_FRESH_SPOOL_SECONDS:-900}"

timestamp="$(date --iso-8601=seconds)"
now_epoch="${BRANCH50_WATCHDOG_NOW_EPOCH:-$(date +%s)}"

mkdir -p "${artifact_root}"

stable_process_probe() {
  local pattern="$1"
  local line

  for _ in 1 2 3; do
    if [[ -n "${BRANCH50_WATCHDOG_PS_FIXTURE:-}" ]]; then
      line="$(awk -v pattern="${pattern}" '$0 ~ pattern {print; exit}' "${BRANCH50_WATCHDOG_PS_FIXTURE}")"
    else
      line="$(ps -eo pid=,args= 2>/dev/null | awk -v pattern="${pattern}" '$0 ~ pattern {print; exit}')"
    fi
    if [[ -n "${line}" ]]; then
      printf '%s\n' "${line}"
      return 0
    fi
    [[ -n "${BRANCH50_WATCHDOG_PS_FIXTURE:-}" ]] && break
    sleep 1
  done

  return 1
}

file_mtime_epoch() {
  local path="$1"

  stat -c '%Y' "${path}" 2>/dev/null || printf '0\n'
}

spool_age_seconds() {
  local spool="$1"
  local mtime

  mtime="$(file_mtime_epoch "${spool}")"
  printf '%s\n' "$((now_epoch - mtime))"
}

classify_run_liveness() {
  local process_line="$1"
  local queue_line="$2"
  local successor_line="$3"
  local latest_spool="$4"
  local latest_terminal="$5"
  local age

  if [[ -n "${latest_terminal}" && -f "${latest_terminal}" ]]; then
    printf '%s\n' "TERMINAL terminal_present"
    return 0
  fi

  if [[ -z "${latest_spool}" || ! -f "${latest_spool}" ]]; then
    if [[ -n "${process_line}${queue_line}${successor_line}" ]]; then
      printf '%s\n' "RUNNING live_process_or_supervisor_without_spool"
    else
      printf '%s\n' "UNKNOWN no_spool_no_terminal_no_visible_process"
    fi
    return 0
  fi

  age="$(spool_age_seconds "${latest_spool}")"
  if (( age <= fresh_spool_seconds )); then
    printf '%s\n' "RUNNING fresh_spool_without_terminal"
    return 0
  fi

  if [[ -n "${process_line}${queue_line}${successor_line}" ]]; then
    printf '%s\n' "RUNNING stale_spool_but_process_or_supervisor_visible"
  else
    printf '%s\n' "LOST stale_spool_no_terminal_no_matching_process_or_supervisor"
  fi
}

gpu_probe() {
  local gpu_line

  if [[ "${BRANCH50_WATCHDOG_NVIDIA_SMI_STATUS:-}" == "fail" ]]; then
    printf '%s\n' "GPU_PROBE_STATUS=GPU_PROBE_UNAVAILABLE"
    printf '%s\n' "GPU_PROBE_ERROR=fixture_forced_failure"
    return 0
  fi

  if gpu_line="$(nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>&1)"; then
    printf '%s\n' "GPU_PROBE_STATUS=PASS"
    printf 'GPU=%s\n' "${gpu_line}"
  else
    printf '%s\n' "GPU_PROBE_STATUS=GPU_PROBE_UNAVAILABLE"
    printf 'GPU_PROBE_ERROR=%s\n' "${gpu_line//$'\n'/ }"
  fi
}

process_line="$(stable_process_probe '[r]un_branch50_300m_ablation[.]py' || true)"
queue_line="$(stable_process_probe '[b]ranch50_ablation_queue[.]sh' || true)"
successor_line="$(stable_process_probe '[r]un_branch50_overnight_campaign[.]py' || true)"
latest_spool="$(find "${artifact_root}" -type f -name mlflow_spool.jsonl -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
latest_validation=""
latest_event=""
latest_terminal=""
if [[ -n "${latest_spool}" && -f "${latest_spool}" ]]; then
  latest_event="$(tail -n 1 "${latest_spool}")"
  latest_validation="$(awk '/"phase": "validation"/ {line=$0} END {print line}' "${latest_spool}")"
  run_root="$(dirname "${latest_spool}")"
  if [[ -f "${run_root}/terminal.json" ]]; then
    latest_terminal="${run_root}/terminal.json"
  fi
fi
liveness="$(classify_run_liveness "${process_line}" "${queue_line}" "${successor_line}" "${latest_spool}" "${latest_terminal}")"
run_liveness="${liveness%% *}"
run_liveness_reason="${liveness#* }"
admitted_source="${BRANCH50_ADMITTED_SOURCE:-/home/mo/DEV/experiments/helix-branch50-linear-context-v0/source-admitted-ec85}"
successor_source="${BRANCH50_SUCCESSOR_SOURCE:-/home/mo/DEV/experiments/helix-branch50-linear-context-v0/source-campaign-admitted-d708}"

{
  printf 'WATCHDOG_AT=%s\n' "${timestamp}"
  if [[ -n "${process_line}" ]]; then
    printf 'RUNNER_OBSERVATION=visible %s\n' "${process_line}"
  else
    printf 'RUNNER_OBSERVATION=not_observed\n'
  fi

  if [[ -n "${queue_line}" ]]; then
    printf 'QUEUE_OBSERVATION=visible %s\n' "${queue_line}"
  else
    printf 'QUEUE_OBSERVATION=not_observed\n'
  fi

  if [[ -n "${successor_line}" ]]; then
    printf 'SUCCESSOR_SUPERVISOR_OBSERVATION=visible %s\n' "${successor_line}"
  else
    printf 'SUCCESSOR_SUPERVISOR_OBSERVATION=not_observed\n'
  fi

  printf 'RUN_LIVENESS=%s\n' "${run_liveness}"
  printf 'RUN_LIVENESS_REASON=%s\n' "${run_liveness_reason}"

  if [[ -d "${admitted_source}/.git" || -f "${admitted_source}/.git" ]]; then
    printf 'ADMITTED_SOURCE_HEAD=%s\n' "$(git -C "${admitted_source}" rev-parse HEAD 2>/dev/null || printf unavailable)"
    printf 'ADMITTED_SOURCE_TREE=%s\n' "$(git -C "${admitted_source}" rev-parse 'HEAD^{tree}' 2>/dev/null || printf unavailable)"
    printf 'ADMITTED_SOURCE_DIRTY=%s\n' "$(test -n "$(git -C "${admitted_source}" status --porcelain 2>/dev/null)" && printf true || printf false)"
  else
    printf 'ADMITTED_SOURCE=unavailable\n'
  fi

  if [[ -d "${successor_source}/.git" || -f "${successor_source}/.git" ]]; then
    printf 'SUCCESSOR_SOURCE_HEAD=%s\n' "$(git -C "${successor_source}" rev-parse HEAD 2>/dev/null || printf unavailable)"
    printf 'SUCCESSOR_SOURCE_TREE=%s\n' "$(git -C "${successor_source}" rev-parse 'HEAD^{tree}' 2>/dev/null || printf unavailable)"
    printf 'SUCCESSOR_SOURCE_DIRTY=%s\n' "$(test -n "$(git -C "${successor_source}" status --porcelain 2>/dev/null)" && printf true || printf false)"
  else
    printf 'SUCCESSOR_SOURCE=unavailable\n'
  fi

  if [[ -n "${latest_spool}" && -f "${latest_spool}" ]]; then
    printf 'LATEST_SPOOL=%s\n' "${latest_spool}"
    printf 'LATEST_EVENT=%s\n' "${latest_event}"
    if [[ -n "${latest_validation}" ]]; then
      printf 'LATEST_VALIDATION=%s\n' "${latest_validation}"
    else
      printf 'LATEST_VALIDATION=unavailable\n'
    fi
    if [[ -n "${latest_terminal}" ]]; then
      printf 'LATEST_TERMINAL=%s\n' "${latest_terminal}"
    else
      printf 'LATEST_TERMINAL=not_emitted\n'
    fi
  else
    printf 'LATEST_SPOOL=unavailable\n'
  fi

  gpu_probe
  df -Pk /home/mo/DEV | tail -n 1 | awk '{printf "DISK_KB_TOTAL=%s DISK_KB_USED=%s DISK_KB_FREE=%s DISK_USE=%s\n", $2, $3, $4, $5}'
  printf '%s\n' 'LEDGER_ACTION=review meaningful checkpoint or terminal evidence; commit and push only when evidence changed'
  printf '%s\n' '---'
} >> "${log_path}"

reminder_staged="${reminder_path}.tmp"
{
  printf 'WATCHDOG_AT=%s\n' "${timestamp}"
  if [[ -n "${process_line}" ]]; then
    printf 'RUNNER_OBSERVATION=visible %s\n' "${process_line}"
  else
    printf 'RUNNER_OBSERVATION=not_observed\n'
  fi
  if [[ -n "${queue_line}" ]]; then
    printf 'QUEUE_OBSERVATION=visible %s\n' "${queue_line}"
  else
    printf 'QUEUE_OBSERVATION=not_observed\n'
  fi
  if [[ -n "${successor_line}" ]]; then
    printf 'SUCCESSOR_SUPERVISOR_OBSERVATION=visible %s\n' "${successor_line}"
  else
    printf 'SUCCESSOR_SUPERVISOR_OBSERVATION=not_observed\n'
  fi
  printf 'RUN_LIVENESS=%s\n' "${run_liveness}"
  printf 'RUN_LIVENESS_REASON=%s\n' "${run_liveness_reason}"
  printf 'LATEST_SPOOL=%s\n' "${latest_spool:-unavailable}"
  printf 'LATEST_EVENT=%s\n' "${latest_event:-unavailable}"
  printf 'LATEST_VALIDATION=%s\n' "${latest_validation:-unavailable}"
  printf 'LATEST_TERMINAL=%s\n' "${latest_terminal:-not_emitted}"
  printf '%s\n' 'LEDGER_ACTION=bank only meaningful new evidence; verify, Lore-commit, and push Branch 50'
} > "${reminder_staged}"
mv -f "${reminder_staged}" "${reminder_path}"
