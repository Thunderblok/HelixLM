#!/usr/bin/env python3
"""Hostile courts for the preregistered Branch-50 comparison law."""

from __future__ import annotations

import importlib.util
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


def main() -> None:
    courts = [
        court_material_tie_prefers_fewer_changes,
        court_material_win_promotes_candidate,
        court_identity_mutation_turns_red,
    ]
    for court in courts:
        court()
        print(f"{court.__name__}=PASS")
    print("BRANCH50_ABLATION_COMPARISON_COURTS=PASS")


if __name__ == "__main__":
    main()
