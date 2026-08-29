#!/usr/bin/env python3
"""Hostile courts for the Branch-50 final completion verifier."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("verify_branch50_campaign_completion.py")
SPEC = importlib.util.spec_from_file_location("branch50_completion", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value))


def fixture(root: Path) -> tuple[Path, Path, Path]:
    campaign = root / "campaign"
    run = root / "run"
    checkpoints = run / "checkpoints"
    campaign.mkdir()
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / "terminal.pt"
    best = checkpoints / "best-model.pt"
    checkpoint.write_bytes(b"terminal")
    best.write_bytes(b"best")
    spool = run / "mlflow_spool.jsonl"
    spool.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {"event": "run_started", "run_id": "mlflow-run"},
                {"event": "run_finished", "status": "FINISHED"},
            )
        )
        + "\n"
    )
    terminal = {
        "status": "PASS",
        "promotion_eligible": True,
        "run_root": str(run.resolve()),
        "full_corpus_pass": True,
        "expected_full_corpus_raw_tokens": MODULE.EXPECTED_FULL_CORPUS_RAW_TOKENS,
        "full_corpus_plan": {"raw_tokens": MODULE.EXPECTED_FULL_CORPUS_RAW_TOKENS},
        "seq_len": MODULE.EXPECTED_SEQ_LEN,
        "parameter_count": {
            "total": MODULE.EXPECTED_PARAMETER_COUNT,
            "trainable": MODULE.EXPECTED_PARAMETER_COUNT,
        },
        "skipped_batches": 0,
        "nonfinite_events": 0,
        "mlflow_errors": [],
        "checkpoint_readback": "PASS",
        "best_checkpoint_readback": "PASS",
        "validation_history": [{"step": 1, "loss": 4.0}],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": MODULE.sha256_file(checkpoint),
        "best_checkpoint": str(best),
        "mlflow_spool": str(spool),
        "mlflow_run_id": "mlflow-run",
        "data_offset": {
            "raw_tokens_seen": MODULE.EXPECTED_FULL_CORPUS_RAW_TOKENS,
            "causal_targets_seen": 1_501_062_500,
        },
        "best_val_loss": 4.0,
        "best_val_step": 1,
    }
    write_json(run / "terminal.json", terminal)
    stages = {name: {} for name in MODULE.REQUIRED_STAGES}
    for name in (
        "live_campaign_courts",
        "compare_primary_300m",
        "compare_operational_100m",
        "build_combined_pilot_manifest",
        "build_full_corpus_manifest",
    ):
        stages[name]["returncode"] = 0
    for name in (
        "operational_control_100m",
        "scheduler_cosine_100m",
        "checkpoint_cadence_100m",
        "combined_pilot_100m",
    ):
        stages[name].update(terminal_status="PASS", promotion_eligible=True)
    stages["full_corpus"].update(run_root=str(run.resolve()))
    state = {
        "status": "PASS",
        "source_head": "head",
        "source_tree": "tree",
        "full_run_root": str(run.resolve()),
        "stages": stages,
    }
    write_json(campaign / "campaign-state.json", state)
    live_path = root / "live.json"
    write_json(
        live_path,
        {
            "status": "PASS",
            "source_head": "head",
            "source_tree": "tree",
            "exact_resume_court": {"status": "PASS"},
            "rotation_court": {"status": "PASS"},
            "hostile_scheduler_mismatch_refusal": "PASS",
            "diminishing_stop_court": {"status": "PASS"},
        },
    )
    lighteval_path = root / "lighteval.json"
    write_json(
        lighteval_path,
        {
            "status": "PREPARED",
            "lighteval_version": "0.13.0",
            "transformers_version": "5.8.1",
            "trust_remote_code": False,
            "checkpoint_preflight": {
                "resolved_config": {
                    "seq_len": MODULE.EXPECTED_SEQ_LEN,
                    "parameter_count": MODULE.EXPECTED_PARAMETER_COUNT,
                },
                "checkpoint": {
                    "path": str(best),
                    "sha256": MODULE.sha256_file(best),
                },
            },
        },
    )
    return campaign, live_path, lighteval_path


def court_complete_identity_chain_passes() -> None:
    with tempfile.TemporaryDirectory() as temp:
        campaign, live, lighteval = fixture(Path(temp))
        packet = MODULE.verify(campaign, live, lighteval)
        assert packet["status"] == "VERIFIED", packet
        assert packet["promotion_recommendation"] == "PROMOTE_FOR_EVALUATION", packet


def court_missing_live_stage_turns_red() -> None:
    with tempfile.TemporaryDirectory() as temp:
        campaign, live, lighteval = fixture(Path(temp))
        state_path = campaign / "campaign-state.json"
        state = MODULE.load_object(state_path)
        del state["stages"]["live_campaign_courts"]
        write_json(state_path, state)
        try:
            MODULE.verify(campaign, live, lighteval)
        except RuntimeError as error:
            assert "stages missing" in str(error)
        else:
            raise AssertionError("missing live stage did not turn RED")


def court_best_checkpoint_substitution_turns_red() -> None:
    with tempfile.TemporaryDirectory() as temp:
        campaign, live, lighteval = fixture(Path(temp))
        manifest = MODULE.load_object(lighteval)
        substituted = Path(temp) / "substituted.pt"
        substituted.write_bytes(b"other")
        manifest["checkpoint_preflight"]["checkpoint"].update(
            path=str(substituted), sha256=MODULE.sha256_file(substituted)
        )
        write_json(lighteval, manifest)
        try:
            MODULE.verify(campaign, live, lighteval)
        except RuntimeError as error:
            assert "Lighteval checkpoint mismatch" in str(error)
        else:
            raise AssertionError("checkpoint substitution did not turn RED")


def court_nonpromotional_stop_is_verified_hold() -> None:
    with tempfile.TemporaryDirectory() as temp:
        campaign, live, lighteval = fixture(Path(temp))
        state_path = campaign / "campaign-state.json"
        state = MODULE.load_object(state_path)
        state["status"] = "STOPPED_DIMINISHING_RETURN"
        write_json(state_path, state)
        terminal_path = Path(state["full_run_root"]) / "terminal.json"
        terminal = MODULE.load_object(terminal_path)
        terminal.update(status="STOPPED_DIMINISHING_RETURN", promotion_eligible=False)
        write_json(terminal_path, terminal)
        packet = MODULE.verify(campaign, live, lighteval)
        assert packet["promotion_recommendation"] == "HOLD_DIMINISHING_RETURN", packet


def main() -> None:
    courts = (
        court_complete_identity_chain_passes,
        court_missing_live_stage_turns_red,
        court_best_checkpoint_substitution_turns_red,
        court_nonpromotional_stop_is_verified_hold,
    )
    for court in courts:
        court()
        print(f"{court.__name__}=PASS")
    print("BRANCH50_CAMPAIGN_COMPLETION_COURTS=PASS")


if __name__ == "__main__":
    main()
