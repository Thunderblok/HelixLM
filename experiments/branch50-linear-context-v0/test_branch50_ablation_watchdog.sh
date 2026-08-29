#!/usr/bin/env bash

set -euo pipefail

script="${1:-$(git rev-parse --show-toplevel)/ops/branch50_ablation_watchdog.sh}"

make_spool() {
  local root="$1"
  local run_name="$2"
  local mtime_epoch="$3"
  local run_root="${root}/${run_name}"

  mkdir -p "${run_root}"
  printf '%s\n' '{"event": "metric", "phase": "train", "step": 100, "ts": 1000}' > "${run_root}/mlflow_spool.jsonl"
  printf '%s\n' '{"event": "metric", "phase": "validation", "step": 100, "ts": 1001}' >> "${run_root}/mlflow_spool.jsonl"
  touch -d "@${mtime_epoch}" "${run_root}/mlflow_spool.jsonl"
}

run_watchdog_case() {
  local root="$1"
  local now_epoch="$2"
  local ps_fixture="$3"
  local log_path="${root}/watchdog.log"
  local reminder_path="${root}/reminder"

  BRANCH50_ABLATION_ARTIFACT_ROOT="${root}/artifacts" \
  BRANCH50_ABLATION_WATCHDOG_LOG="${log_path}" \
  BRANCH50_ABLATION_REMINDER_PATH="${reminder_path}" \
  BRANCH50_ADMITTED_SOURCE="${root}/missing-admitted-source" \
  BRANCH50_SUCCESSOR_SOURCE="${root}/missing-successor-source" \
  BRANCH50_WATCHDOG_NOW_EPOCH="${now_epoch}" \
  BRANCH50_WATCHDOG_PS_FIXTURE="${ps_fixture}" \
  BRANCH50_WATCHDOG_NVIDIA_SMI_STATUS=fail \
  bash "${script}"

  cat "${log_path}"
}

court_fresh_spool_pid_negative_is_running() {
  local temporary ps_fixture output

  temporary="$(mktemp -d)"
  ps_fixture="${temporary}/ps.txt"
  touch "${ps_fixture}"
  mkdir -p "${temporary}/artifacts"
  make_spool "${temporary}/artifacts" "branch50-ablation-fresh-s512-b12-a7-t100-20260829T000000Z" 1000
  output="$(run_watchdog_case "${temporary}" 1010 "${ps_fixture}")"

  grep -q '^RUNNER_OBSERVATION=not_observed$' <<<"${output}"
  grep -q '^RUN_LIVENESS=RUNNING$' <<<"${output}"
  grep -q '^RUN_LIVENESS_REASON=fresh_spool_without_terminal$' <<<"${output}"
  grep -q '^LATEST_TERMINAL=not_emitted$' <<<"${output}"
  grep -q '^GPU_PROBE_STATUS=GPU_PROBE_UNAVAILABLE$' <<<"${output}"
}

court_stale_spool_absent_process_is_lost() {
  local temporary ps_fixture output

  temporary="$(mktemp -d)"
  ps_fixture="${temporary}/ps.txt"
  touch "${ps_fixture}"
  mkdir -p "${temporary}/artifacts"
  make_spool "${temporary}/artifacts" "branch50-ablation-stale-s512-b12-a7-t100-20260829T000000Z" 1000
  output="$(run_watchdog_case "${temporary}" 3000 "${ps_fixture}")"

  grep -q '^RUNNER_OBSERVATION=not_observed$' <<<"${output}"
  grep -q '^QUEUE_OBSERVATION=not_observed$' <<<"${output}"
  grep -q '^SUCCESSOR_SUPERVISOR_OBSERVATION=not_observed$' <<<"${output}"
  grep -q '^RUN_LIVENESS=LOST$' <<<"${output}"
  grep -q '^RUN_LIVENESS_REASON=stale_spool_no_terminal_no_matching_process_or_supervisor$' <<<"${output}"
}

main() {
  court_fresh_spool_pid_negative_is_running
  printf '%s\n' 'court_fresh_spool_pid_negative_is_running=PASS'
  court_stale_spool_absent_process_is_lost
  printf '%s\n' 'court_stale_spool_absent_process_is_lost=PASS'
  printf '%s\n' 'BRANCH50_ABLATION_WATCHDOG_HOSTILE_COURTS=PASS'
}

main "$@"
