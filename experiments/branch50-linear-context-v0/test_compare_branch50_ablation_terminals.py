#!/usr/bin/env python3
"""Hostile courts for the preregistered Branch-50 comparison law."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("compare_branch50_ablation_terminals.py")
SPEC = importlib.util.spec_from_file_location("branch50_compare", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def record(ablation_id: str, loss: float, changed: list[str], throughput: float) -> dict:
    identity = {
        "source_identity": {"head": "h", "tree": "t"},
        "manifest_roots": {"train": "a", "val": "b"},
        "seq_len": 512,
        "batch_size": 12,
        "grad_accum": 7,
        "target_causal_targets": 300_000_000,
        "target_raw_tokens": 300_625_920,
        "eval_every": 100,
        "validation_batches": 16,
        "seed": 42,
        "steps": 6990,
    }
    return {
        **identity,
        "run_root": f"/{ablation_id}",
        "ablation_id": ablation_id,
        "mlflow_run_id": f"run-{ablation_id}",
        "contract_root": f"contract-{ablation_id}",
        "changed_knobs": changed,
        "knobs": {"learning_rate": 1.5e-4},
        "validation": {6990: {"loss": loss, "ppl": 1.0, "targets": 1}},
        "final_val_loss": loss,
        "final_val_ppl": 1.0,
        "best_val_loss": loss,
        "best_val_ppl": 1.0,
        "best_val_step": 6990,
        "last_train": {"causal_targets_per_second": throughput},
        "peak_vram_bytes": 1,
        "checkpoint_sha256": "sha",
        "final_model_root": "model",
    }


def court_material_tie_prefers_fewer_changes() -> None:
    control = record("control", 4.300, [], 20_000.0)
    candidate = record("candidate", 4.297, ["learning_rate"], 21_000.0)
    packet = MODULE.build_packet([control, candidate], material_loss_delta=0.005)
    assert packet["selected_ablation_id"] == "control", packet


def court_material_win_promotes_candidate() -> None:
    control = record("control", 4.300, [], 21_000.0)
    candidate = record("candidate", 4.290, ["learning_rate"], 20_000.0)
    packet = MODULE.build_packet([control, candidate], material_loss_delta=0.005)
    assert packet["selected_ablation_id"] == "candidate", packet


def court_identity_mutation_turns_red() -> None:
    control = record("control", 4.300, [], 20_000.0)
    candidate = record("candidate", 4.290, ["learning_rate"], 20_000.0)
    candidate["seed"] = 43
    try:
        MODULE.build_packet([control, candidate], material_loss_delta=0.005)
    except SystemExit as error:
        assert "identity mismatch" in str(error)
    else:
        raise AssertionError("identity mutation did not turn comparison court RED")


def legacy_terminal(root: Path) -> tuple[dict, dict]:
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir()
    checkpoint = checkpoint_root / "terminal.pt"
    checkpoint.write_bytes(b"checkpoint")
    best = checkpoint_root / "best-model.pt"
    best.write_bytes(b"best")
    contract = {
        "ablation_id": "legacy",
        "source_identity": {
            "source_head": MODULE.LEGACY_ADMITTED_SOURCE_HEAD,
            "source_tree": MODULE.LEGACY_ADMITTED_SOURCE_TREE,
            "source_dirty": "false",
            "model_source_diff": "false",
        },
        "target_causal_targets": 100,
        "seq_len": 512,
        "batch_size": 12,
        "grad_accum": 7,
        "knobs": {"learning_rate": 0.00015},
    }
    terminal = {
        "status": "PASS",
        "run_root": str(root.resolve()),
        "ablation_id": "legacy",
        "checkpoint_readback": "PASS",
        "best_checkpoint_readback": "PASS",
        "skipped_batches": 0,
        "nonfinite_events": 0,
        "mlflow_errors": [],
        "checkpoint_model_root": "final-root",
        "final_model_root": "final-root",
        "initial_model_root": "initial-root",
        "checkpoint_optimizer_state_entries": 1,
        "steps": 2,
        "causal_targets_per_optimizer_step": 42_924,
        "aligned_target_causal_targets": 85_848,
        "target_causal_targets": 100,
        "raw_tokens_per_optimizer_step": 43_008,
        "data_offset": {"causal_targets_seen": 85_848, "raw_tokens_seen": 86_016},
        "seq_len": 512,
        "batch_size": 12,
        "grad_accum": 7,
        "learning_rate": 0.00015,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": MODULE.sha256_file(checkpoint),
        "best_checkpoint": str(best),
    }
    (root / "terminal.json").write_text(json.dumps(terminal))
    return terminal, contract


def court_exact_legacy_terminal_is_derived_not_mutated() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        terminal, contract = legacy_terminal(root)
        admission = MODULE.admit_terminal(root, terminal, contract)
        assert admission["mode"] == "derived_exact_legacy_pass", admission
        assert admission["target_raw_tokens"] == 86_016, admission
        assert admission["target_raw_tokens_source"].startswith("derived_legacy"), admission
        assert "promotion_eligible" not in terminal


def court_legacy_source_mutation_turns_red() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        terminal, contract = legacy_terminal(root)
        contract["source_identity"]["source_head"] = "mutated"
        try:
            MODULE.admit_terminal(root, terminal, contract)
        except SystemExit as error:
            assert "exact legacy source" in str(error)
        else:
            raise AssertionError("legacy source mutation did not turn admission RED")


def court_legacy_health_mutation_turns_red() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        terminal, contract = legacy_terminal(root)
        terminal["nonfinite_events"] = 1
        try:
            MODULE.admit_terminal(root, terminal, contract)
        except SystemExit as error:
            assert "unhealthy terminal" in str(error)
        else:
            raise AssertionError("legacy health mutation did not turn admission RED")


def court_legacy_run_root_escape_turns_red() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        terminal, contract = legacy_terminal(root)
        terminal["run_root"] = str(root.parent)
        try:
            MODULE.admit_terminal(root, terminal, contract)
        except SystemExit as error:
            assert "run-root custody mismatch" in str(error)
        else:
            raise AssertionError("legacy run-root mutation did not turn admission RED")


def court_legacy_checkpoint_escape_turns_red() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        terminal, contract = legacy_terminal(root)
        escaped = root / "escaped.pt"
        escaped.write_bytes(b"checkpoint")
        terminal["checkpoint"] = str(escaped)
        terminal["checkpoint_sha256"] = MODULE.sha256_file(escaped)
        try:
            MODULE.admit_terminal(root, terminal, contract)
        except SystemExit as error:
            assert "escaped run custody" in str(error)
        else:
            raise AssertionError("legacy checkpoint escape did not turn admission RED")


def court_legacy_ablation_identity_mutation_turns_red() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        terminal, contract = legacy_terminal(root)
        contract["ablation_id"] = "different"
        try:
            MODULE.admit_terminal(root, terminal, contract)
        except SystemExit as error:
            assert "ablation identity mismatch" in str(error)
        else:
            raise AssertionError("legacy ablation identity mutation did not turn admission RED")


def court_legacy_raw_token_geometry_mutation_turns_red() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        terminal, contract = legacy_terminal(root)
        terminal["raw_tokens_per_optimizer_step"] = 65
        terminal["data_offset"]["raw_tokens_seen"] = 130
        try:
            MODULE.admit_terminal(root, terminal, contract)
        except SystemExit as error:
            assert "raw-token accounting mismatch" in str(error)
        else:
            raise AssertionError("self-consistent wrong raw-token geometry did not turn RED")


def court_legacy_causal_geometry_mutation_turns_red() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        terminal, contract = legacy_terminal(root)
        terminal["causal_targets_per_optimizer_step"] = 42_923
        terminal["aligned_target_causal_targets"] = 85_846
        terminal["data_offset"]["causal_targets_seen"] = 85_846
        try:
            MODULE.admit_terminal(root, terminal, contract)
        except SystemExit as error:
            assert "causal/accounting identity mismatch" in str(error)
        else:
            raise AssertionError("self-consistent wrong causal geometry did not turn RED")


def court_explicit_promotion_field_preserves_new_source() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        terminal, contract = legacy_terminal(root)
        terminal["promotion_eligible"] = True
        contract["source_identity"]["source_head"] = "new-source"
        contract["source_identity"]["source_tree"] = "new-tree"
        contract["target_raw_tokens"] = 86_016
        (root / "terminal.json").write_text(json.dumps(terminal))
        admission = MODULE.admit_terminal(root, terminal, contract)
        assert admission["mode"] == "explicit_promotion_eligible", admission
        assert admission["target_raw_tokens_source"] == "contract", admission


def court_explicit_nonpromotion_turns_red() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        terminal, contract = legacy_terminal(root)
        terminal["promotion_eligible"] = False
        (root / "terminal.json").write_text(json.dumps(terminal))
        try:
            MODULE.admit_terminal(root, terminal, contract)
        except SystemExit as error:
            assert "non-promotable terminal" in str(error)
        else:
            raise AssertionError("explicit nonpromotion did not turn admission RED")


def main() -> None:
    courts = [
        court_material_tie_prefers_fewer_changes,
        court_material_win_promotes_candidate,
        court_identity_mutation_turns_red,
        court_exact_legacy_terminal_is_derived_not_mutated,
        court_legacy_source_mutation_turns_red,
        court_legacy_health_mutation_turns_red,
        court_legacy_run_root_escape_turns_red,
        court_legacy_checkpoint_escape_turns_red,
        court_legacy_ablation_identity_mutation_turns_red,
        court_legacy_raw_token_geometry_mutation_turns_red,
        court_legacy_causal_geometry_mutation_turns_red,
        court_explicit_promotion_field_preserves_new_source,
        court_explicit_nonpromotion_turns_red,
    ]
    for court in courts:
        court()
        print(f"{court.__name__}=PASS")
    print("BRANCH50_ABLATION_COMPARISON_COURTS=PASS")


if __name__ == "__main__":
    main()
