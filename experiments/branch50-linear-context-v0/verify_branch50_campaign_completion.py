#!/usr/bin/env python3
"""Bind Branch-50 campaign, live courts, checkpoint, MLflow, and Lighteval truth."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_SEQ_LEN = 512
EXPECTED_PARAMETER_COUNT = 53_592_340
EXPECTED_FULL_CORPUS_RAW_TOKENS = 1_504_000_000
REQUIRED_STAGES = (
    "old_queue_terminal",
    "live_campaign_courts",
    "operational_control_100m",
    "scheduler_cosine_100m",
    "checkpoint_cadence_100m",
    "compare_primary_300m",
    "compare_operational_100m",
    "build_combined_pilot_manifest",
    "combined_pilot_100m",
    "build_full_corpus_manifest",
    "full_corpus",
)


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_root(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify(
    campaign_root: Path, live_receipt_path: Path, lighteval_manifest_path: Path
) -> dict[str, Any]:
    campaign_root = campaign_root.resolve()
    state_path = campaign_root / "campaign-state.json"
    state = load_object(state_path)
    stages = state.get("stages")
    require(isinstance(stages, dict), "campaign has no stage map")
    missing = [name for name in REQUIRED_STAGES if name not in stages]
    require(not missing, f"campaign stages missing: {missing}")

    for name in (
        "live_campaign_courts",
        "compare_primary_300m",
        "compare_operational_100m",
        "build_combined_pilot_manifest",
        "build_full_corpus_manifest",
    ):
        require(stages[name].get("returncode") == 0, f"stage did not pass: {name}")
    for name in (
        "operational_control_100m",
        "scheduler_cosine_100m",
        "checkpoint_cadence_100m",
        "combined_pilot_100m",
    ):
        require(stages[name].get("terminal_status") == "PASS", f"non-PASS stage: {name}")
        require(stages[name].get("promotion_eligible") is True, f"non-promotable stage: {name}")

    live = load_object(live_receipt_path.resolve())
    require(live.get("status") == "PASS", "live campaign courts are not PASS")
    require(live.get("source_head") == state.get("source_head"), "live/source head mismatch")
    require(live.get("source_tree") == state.get("source_tree"), "live/source tree mismatch")
    require(live.get("exact_resume_court", {}).get("status") == "PASS", "resume court failed")
    require(live.get("rotation_court", {}).get("status") == "PASS", "rotation court failed")
    require(live.get("hostile_scheduler_mismatch_refusal") == "PASS", "hostile resume court failed")
    require(live.get("diminishing_stop_court", {}).get("status") == "PASS", "diminishing court failed")

    full_run_root = Path(str(state.get("full_run_root", ""))).resolve()
    require(full_run_root.is_dir(), "full run root is unavailable")
    require(stages["full_corpus"].get("run_root") == str(full_run_root), "full stage/root mismatch")
    terminal_path = full_run_root / "terminal.json"
    terminal = load_object(terminal_path)
    campaign_status = state.get("status")
    require(campaign_status in {"PASS", "STOPPED_DIMINISHING_RETURN"}, "campaign is nonterminal")
    require(terminal.get("status") == campaign_status, "campaign/full terminal status mismatch")
    expected_promotion = campaign_status == "PASS"
    require(terminal.get("promotion_eligible") is expected_promotion, "promotion eligibility mismatch")
    require(terminal.get("run_root") == str(full_run_root), "terminal run-root mismatch")
    require(terminal.get("full_corpus_pass") is True, "full-corpus mode was not executed")
    require(
        int(terminal.get("expected_full_corpus_raw_tokens", -1))
        == EXPECTED_FULL_CORPUS_RAW_TOKENS,
        "full-corpus raw-token contract mismatch",
    )
    require(terminal.get("full_corpus_plan", {}).get("raw_tokens") == EXPECTED_FULL_CORPUS_RAW_TOKENS, "full-corpus plan mismatch")
    require(int(terminal.get("seq_len", -1)) == EXPECTED_SEQ_LEN, "sequence length drift")
    params = terminal.get("parameter_count", {})
    require(params.get("total") == EXPECTED_PARAMETER_COUNT, "parameter-count drift")
    require(params.get("trainable") == EXPECTED_PARAMETER_COUNT, "trainable-count drift")
    require(terminal.get("skipped_batches") == 0, "skipped batches observed")
    require(terminal.get("nonfinite_events") == 0, "nonfinite events observed")
    require(not terminal.get("mlflow_errors"), "MLflow errors observed")
    require(terminal.get("checkpoint_readback") == "PASS", "terminal checkpoint readback failed")
    require(terminal.get("best_checkpoint_readback") == "PASS", "best checkpoint readback failed")
    require(terminal.get("validation_history"), "fixed validation history is empty")

    checkpoint_root = (full_run_root / "checkpoints").resolve()
    checkpoint = Path(str(terminal.get("checkpoint", ""))).resolve()
    best_checkpoint = Path(str(terminal.get("best_checkpoint", ""))).resolve()
    require(within(checkpoint, checkpoint_root) and checkpoint.is_file(), "terminal checkpoint custody failed")
    require(within(best_checkpoint, checkpoint_root) and best_checkpoint.is_file(), "best checkpoint custody failed")
    require(sha256_file(checkpoint) == terminal.get("checkpoint_sha256"), "terminal checkpoint hash mismatch")

    spool_path = Path(str(terminal.get("mlflow_spool", full_run_root / "mlflow_spool.jsonl"))).resolve()
    require(within(spool_path, full_run_root) and spool_path.is_file(), "MLflow spool custody failed")
    events = [json.loads(line) for line in spool_path.read_text().splitlines() if line.strip()]
    started = [event for event in events if event.get("event") == "run_started"]
    finished = [event for event in events if event.get("event") == "run_finished"]
    require(len(started) == 1 and started[0].get("run_id") == terminal.get("mlflow_run_id"), "MLflow start identity mismatch")
    require(len(finished) == 1, "MLflow finish observation missing")
    require(not any(event.get("event") == "error" for event in events), "MLflow spool error event")

    lighteval = load_object(lighteval_manifest_path.resolve())
    preflight = lighteval.get("checkpoint_preflight", {}).get("checkpoint", {})
    require(lighteval.get("status") == "PREPARED", "Lighteval lane is not prepared")
    require(lighteval.get("lighteval_version") == "0.13.0", "Lighteval version drift")
    require(lighteval.get("transformers_version") == "5.8.1", "Transformers version drift")
    require(lighteval.get("trust_remote_code") is False, "remote code trust enabled")
    require(lighteval.get("checkpoint_preflight", {}).get("resolved_config", {}).get("seq_len") == EXPECTED_SEQ_LEN, "Lighteval sequence mismatch")
    require(lighteval.get("checkpoint_preflight", {}).get("resolved_config", {}).get("parameter_count") == EXPECTED_PARAMETER_COUNT, "Lighteval parameter mismatch")
    require(Path(str(preflight.get("path", ""))).resolve() == best_checkpoint, "Lighteval checkpoint mismatch")
    require(preflight.get("sha256") == sha256_file(best_checkpoint), "Lighteval best-checkpoint hash mismatch")

    recommendation = "PROMOTE_FOR_EVALUATION" if expected_promotion else "HOLD_DIMINISHING_RETURN"
    packet = {
        "schema": "helix.branch50.campaign-completion.v0",
        "status": "VERIFIED",
        "promotion_recommendation": recommendation,
        "campaign_root": str(campaign_root),
        "campaign_state_sha256": sha256_file(state_path),
        "source_head": state["source_head"],
        "source_tree": state["source_tree"],
        "full_run_root": str(full_run_root),
        "full_terminal_sha256": sha256_file(terminal_path),
        "terminal_checkpoint_sha256": terminal["checkpoint_sha256"],
        "best_checkpoint_sha256": sha256_file(best_checkpoint),
        "live_receipt_sha256": sha256_file(live_receipt_path.resolve()),
        "lighteval_manifest_sha256": sha256_file(lighteval_manifest_path.resolve()),
        "mlflow_run_id": terminal["mlflow_run_id"],
        "campaign_status": campaign_status,
        "causal_targets_seen": terminal.get("data_offset", {}).get("causal_targets_seen"),
        "raw_tokens_seen": terminal.get("data_offset", {}).get("raw_tokens_seen"),
        "best_val_loss": terminal.get("best_val_loss"),
        "best_val_step": terminal.get("best_val_step"),
    }
    packet["packet_root"] = canonical_root(packet)
    return packet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--live-receipt", type=Path, required=True)
    parser.add_argument("--lighteval-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"REFUSED: output already exists: {args.output}")
    try:
        packet = verify(args.campaign_root, args.live_receipt, args.lighteval_manifest)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        raise SystemExit(f"REFUSED: {error}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps(packet, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
