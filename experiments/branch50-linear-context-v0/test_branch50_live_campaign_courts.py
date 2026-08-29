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


def main() -> None:
    courts = [
        court_exact_resume_detects_mutation,
        court_rotation_requires_exact_steps_and_no_staged_file,
    ]
    for court in courts:
        court()
        print(f"{court.__name__}=PASS")
    print("BRANCH50_LIVE_CAMPAIGN_STATIC_COURTS=PASS")


if __name__ == "__main__":
    main()
