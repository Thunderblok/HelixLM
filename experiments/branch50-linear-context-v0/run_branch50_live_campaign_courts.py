#!/usr/bin/env python3
"""Run the live Branch-50 resume, rotation, refusal, and stop courts.

This is an execution court, not a training candidate.  It uses the admitted
Branch-50 runner in four short, sequential GPU runs and emits one custody
receipt outside the source worktree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch


REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "experiments" / "branch50-linear-context-v0" / "run_branch50_300m_ablation.py"
EXPERIMENT_ROOT = REPO.parent
ARTIFACT_ROOT = EXPERIMENT_ROOT / "artifacts" / "live-campaign-courts-v0"
ABLATION_ROOT = EXPERIMENT_ROOT / "artifacts" / "ablation-300m-v0"
CAUSAL_TARGETS_PER_STEP = 12 * 7 * 511
TOTAL_STEPS = 12
SPLIT_STEPS = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def assert_clean_source() -> None:
    dirty = git_value("status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise SystemExit("REFUSED: live courts require a clean committed Branch-50 source")


def assert_exclusive_gpu_lease() -> None:
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="], check=True, capture_output=True, text=True
    )
    conflicts = []
    for line in result.stdout.splitlines():
        if "run_branch50_300m_ablation.py" not in line:
            continue
        conflicts.append(line.strip())
    if conflicts:
        raise SystemExit(
            "REFUSED: another Branch-50 runner owns the GPU:\n" + "\n".join(conflicts)
        )
    gpu_processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if gpu_processes:
        raise SystemExit(
            "REFUSED: another compute process owns the GPU:\n" + gpu_processes
        )


def run_dirs(ablation_id: str) -> set[Path]:
    return set(ABLATION_ROOT.glob(f"branch50-ablation-{ablation_id}-s512-b12-a7-t*"))


def run_runner(
    receipt_dir: Path,
    *,
    ablation_id: str,
    max_steps: int,
    target_steps: int = TOTAL_STEPS,
    resume: Path | None = None,
    diminishing: bool = False,
    expect_success: bool = True,
) -> tuple[subprocess.CompletedProcess[str], Path | None]:
    before = run_dirs(ablation_id)
    command = [
        sys.executable,
        str(RUNNER),
        "--ablation-id",
        ablation_id,
        "--target-causal-targets",
        str(target_steps * CAUSAL_TARGETS_PER_STEP),
        "--eval-every",
        "1" if diminishing else "2",
        "--checkpoint-every",
        "2",
        "--validation-batches",
        "1",
        "--max-optimizer-steps",
        str(max_steps),
        "--skip-shard-sha256",
    ]
    if resume is not None:
        command.extend(["--resume", str(resume)])
    if diminishing:
        command.extend(
            [
                "--diminishing-window-evals",
                "1",
                "--diminishing-min-improvement",
                "1000",
                "--diminishing-patience-windows",
                "1",
                "--diminishing-min-optimizer-steps",
                "2",
            ]
        )

    started = time.time()
    result = subprocess.run(
        command,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=1_200,
    )
    label = f"{ablation_id}-max{max_steps}-target{target_steps}"
    (receipt_dir / f"{label}.stdout.log").write_text(result.stdout)
    (receipt_dir / f"{label}.stderr.log").write_text(result.stderr)
    (receipt_dir / f"{label}.command.json").write_text(
        json.dumps(
            {
                "argv": command,
                "returncode": result.returncode,
                "started_at_unix": started,
                "ended_at_unix": time.time(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if expect_success and result.returncode != 0:
        raise RuntimeError(f"live court runner failed: {label}; stderr={result.stderr[-2000:]}")
    if not expect_success and result.returncode == 0:
        raise RuntimeError(f"hostile resume unexpectedly succeeded: {label}")

    created = sorted(run_dirs(ablation_id) - before, key=lambda path: path.stat().st_mtime)
    run_root = created[-1] if created else None
    if expect_success and run_root is None:
        raise RuntimeError(f"runner created no run root: {label}")
    return result, run_root


def load_terminal(run_root: Path) -> dict[str, Any]:
    terminal_path = run_root / "terminal.json"
    if not terminal_path.is_file():
        raise RuntimeError(f"missing terminal: {terminal_path}")
    return json.loads(terminal_path.read_text())


def compare_exact_resume(
    uninterrupted: dict[str, Any], resumed: dict[str, Any]
) -> dict[str, Any]:
    fields = (
        "steps",
        "final_model_root",
        "data_offset",
        "best_val_loss",
        "best_val_step",
        "last_val_loss",
        "validation_history",
        "scheduler",
        "skipped_batches",
        "nonfinite_events",
        "checkpoint_readback",
        "best_checkpoint_readback",
        "mlflow_errors",
    )
    mismatches = {
        field: {"uninterrupted": uninterrupted.get(field), "resumed": resumed.get(field)}
        for field in fields
        if uninterrupted.get(field) != resumed.get(field)
    }
    return {"status": "PASS" if not mismatches else "FAIL", "mismatches": mismatches}


def verify_rotation(run_root: Path) -> dict[str, Any]:
    checkpoint_dir = run_root / "checkpoints"
    latest_path = checkpoint_dir / "latest.pt"
    previous_path = checkpoint_dir / "previous.pt"
    staged_path = checkpoint_dir / "staged.pt"
    if not latest_path.is_file() or not previous_path.is_file() or staged_path.exists():
        return {
            "status": "FAIL",
            "latest_exists": latest_path.is_file(),
            "previous_exists": previous_path.is_file(),
            "staged_exists": staged_path.exists(),
        }
    latest = torch.load(latest_path, map_location="cpu", weights_only=True)
    latest_step = int(latest["step"])
    del latest
    previous = torch.load(previous_path, map_location="cpu", weights_only=True)
    previous_step = int(previous["step"])
    del previous
    passed = latest_step == 12 and previous_step == 10
    return {
        "status": "PASS" if passed else "FAIL",
        "latest_step": latest_step,
        "previous_step": previous_step,
        "latest_sha256": sha256(latest_path),
        "previous_sha256": sha256(previous_path),
        "staged_exists": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("REFUSED: pass --execute to consume the exclusive GPU lease")
    assert_clean_source()
    assert_exclusive_gpu_lease()

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    receipt_dir = ARTIFACT_ROOT / f"branch50-live-courts-{stamp}"
    receipt_dir.mkdir(parents=True, exist_ok=False)

    _, uninterrupted_root = run_runner(
        receipt_dir,
        ablation_id="live-resume-uninterrupted",
        max_steps=TOTAL_STEPS,
    )
    _, split_root = run_runner(
        receipt_dir,
        ablation_id="live-resume-split",
        max_steps=SPLIT_STEPS,
    )
    assert uninterrupted_root is not None and split_root is not None
    split_checkpoint = split_root / "checkpoints" / "latest.pt"
    _, resumed_root = run_runner(
        receipt_dir,
        ablation_id="live-resume-split",
        max_steps=TOTAL_STEPS,
        resume=split_checkpoint,
    )
    assert resumed_root is not None

    hostile, _ = run_runner(
        receipt_dir,
        ablation_id="live-resume-split",
        max_steps=TOTAL_STEPS + 1,
        target_steps=TOTAL_STEPS + 1,
        resume=split_checkpoint,
        expect_success=False,
    )
    hostile_text = hostile.stdout + "\n" + hostile.stderr
    hostile_pass = "REFUSED: resume scheduler does not match ablation" in hostile_text

    _, stop_root = run_runner(
        receipt_dir,
        ablation_id="live-diminishing-stop",
        max_steps=TOTAL_STEPS,
        diminishing=True,
    )
    assert stop_root is not None

    uninterrupted = load_terminal(uninterrupted_root)
    resumed = load_terminal(resumed_root)
    stopped = load_terminal(stop_root)
    resume_court = compare_exact_resume(uninterrupted, resumed)
    rotation_court = verify_rotation(resumed_root)
    stop_pass = (
        stopped.get("status") == "STOPPED_DIMINISHING_RETURN"
        and stopped.get("promotion_eligible") is False
        and int(stopped.get("steps", 0)) == 2
        and stopped.get("stop_state", {}).get("diminishing_decision", {}).get("should_stop")
        is True
    )

    receipt = {
        "schema": "helix.branch50.live-campaign-courts.v0",
        "status": "PASS"
        if resume_court["status"] == "PASS"
        and rotation_court["status"] == "PASS"
        and hostile_pass
        and stop_pass
        else "FAIL",
        "source_head": git_value("rev-parse", "HEAD"),
        "source_tree": git_value("rev-parse", "HEAD^{tree}"),
        "runner_sha256": sha256(RUNNER),
        "python": sys.version,
        "uninterrupted_run_root": str(uninterrupted_root),
        "split_run_root": str(split_root),
        "resumed_run_root": str(resumed_root),
        "diminishing_run_root": str(stop_root),
        "run_evidence": {
            "uninterrupted": {
                "mlflow_run_id": uninterrupted.get("mlflow_run_id"),
                "terminal_sha256": sha256(uninterrupted_root / "terminal.json"),
                "checkpoint_sha256": uninterrupted.get("checkpoint_sha256"),
            },
            "resumed": {
                "mlflow_run_id": resumed.get("mlflow_run_id"),
                "terminal_sha256": sha256(resumed_root / "terminal.json"),
                "checkpoint_sha256": resumed.get("checkpoint_sha256"),
            },
            "diminishing": {
                "mlflow_run_id": stopped.get("mlflow_run_id"),
                "terminal_sha256": sha256(stop_root / "terminal.json"),
                "checkpoint_sha256": stopped.get("checkpoint_sha256"),
            },
        },
        "exact_resume_court": resume_court,
        "rotation_court": rotation_court,
        "hostile_scheduler_mismatch_refusal": "PASS" if hostile_pass else "FAIL",
        "diminishing_stop_court": {
            "status": "PASS" if stop_pass else "FAIL",
            "terminal_status": stopped.get("status"),
            "promotion_eligible": stopped.get("promotion_eligible"),
            "steps": stopped.get("steps"),
            "stop_state": stopped.get("stop_state"),
        },
    }
    receipt_path = receipt_dir / "court-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**receipt, "receipt_sha256": sha256(receipt_path)}, indent=2, sort_keys=True))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
