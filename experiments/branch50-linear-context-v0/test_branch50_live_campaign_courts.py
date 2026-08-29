#!/usr/bin/env python3
"""Runtime-aligned courts for the Branch-50 live-court orchestrator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile

import torch


MODULE_PATH = Path(__file__).with_name("run_branch50_live_campaign_courts.py")
SPEC = importlib.util.spec_from_file_location("branch50_live_courts", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def court_exact_resume_detects_mutation() -> None:
    baseline = {
        "steps": 12,
        "final_model_root": "root-a",
        "data_offset": {"samples_seen": 1},
        "best_val_loss": 4.0,
        "best_val_step": 12,
        "last_val_loss": 4.0,
        "validation_history": [{"step": 12, "val_loss": 4.0}],
        "scheduler": {"type": "linear_warmup_then_constant"},
        "skipped_batches": 0,
        "nonfinite_events": 0,
        "checkpoint_readback": "PASS",
        "best_checkpoint_readback": "PASS",
        "mlflow_errors": [],
    }
    same = dict(baseline)
    assert MODULE.compare_exact_resume(baseline, same)["status"] == "PASS"
    mutated = dict(baseline)
    mutated["final_model_root"] = "root-b"
    result = MODULE.compare_exact_resume(baseline, mutated)
    assert result["status"] == "FAIL"
    assert "final_model_root" in result["mismatches"]


def court_rotation_requires_exact_steps_and_no_staged_file() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        run_root = Path(temporary)
        checkpoint_dir = run_root / "checkpoints"
        checkpoint_dir.mkdir()
        torch.save({"step": 12}, checkpoint_dir / "latest.pt")
        torch.save({"step": 10}, checkpoint_dir / "previous.pt")
        assert MODULE.verify_rotation(run_root)["status"] == "PASS"
        (checkpoint_dir / "staged.pt").write_bytes(b"hostile")
        assert MODULE.verify_rotation(run_root)["status"] == "FAIL"


def court_bounded_browser_gpu_process_is_not_training_contention() -> None:
    chrome = {
        "pid": 101,
        "used_memory_mib": 152,
        "cmdline": "/opt/google/chrome/chrome --type=gpu-process",
    }
    assert MODULE.conflicting_gpu_processes([chrome]) == []

    oversized = dict(chrome, used_memory_mib=257)
    assert MODULE.conflicting_gpu_processes([oversized]) == [oversized]

    python = {
        "pid": 202,
        "used_memory_mib": 1,
        "cmdline": "/usr/bin/python train.py",
    }
    assert MODULE.conflicting_gpu_processes([python]) == [python]

    wrong_chrome_role = dict(chrome, cmdline="/opt/google/chrome/chrome --renderer")
    assert MODULE.conflicting_gpu_processes([wrong_chrome_role]) == [wrong_chrome_role]


def court_gpu_process_rows_refuse_malformed_observation() -> None:
    assert MODULE.parse_gpu_process_rows("101, 152") == [
        {"pid": 101, "used_memory_mib": 152, "cmdline": MODULE.read_process_cmdline(101)}
    ]
    try:
        MODULE.parse_gpu_process_rows("malformed")
    except SystemExit as error:
        assert "malformed nvidia-smi compute row" in str(error)
    else:
        raise AssertionError("malformed GPU observation did not turn the court RED")


def main() -> None:
    courts = [
        court_exact_resume_detects_mutation,
        court_rotation_requires_exact_steps_and_no_staged_file,
        court_bounded_browser_gpu_process_is_not_training_contention,
        court_gpu_process_rows_refuse_malformed_observation,
    ]
    for court in courts:
        court()
        print(f"{court.__name__}=PASS")
    print("BRANCH50_LIVE_CAMPAIGN_STATIC_COURTS=PASS")


if __name__ == "__main__":
    main()
