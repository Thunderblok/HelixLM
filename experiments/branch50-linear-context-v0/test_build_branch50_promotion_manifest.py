#!/usr/bin/env python3
"""Hostile courts for Branch-50 promotion-manifest construction."""

from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("build_branch50_promotion_manifest.py")
SPEC = importlib.util.spec_from_file_location("branch50_promote", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


BASELINE = {
    "learning_rate": 1.5e-4,
    "warmup_microbatches": 2_000,
    "scheduler_policy": "linear_warmup_then_constant",
    "scheduler_min_lr_ratio": 1.0,
    "checkpoint_every": 500,
    "weight_decay": 0.05,
    "grad_clip": 1.0,
    "dropout": 0.05,
    "attention_dropout": 0.05,
    "ffn_expansion": 2.5,
}


def packet(run_id: str, knobs: dict) -> dict:
    return {
        "schema": "helix.branch50.ablation-comparison.v0",
        "status": "PASS",
        "packet_root": f"packet-{run_id}",
        "selected_knobs": knobs,
        "ordered_candidates": [{"mlflow_run_id": run_id}],
    }


def court_merges_only_operational_keys() -> None:
    primary_knobs = {**BASELINE, "learning_rate": 1e-4}
    operational_knobs = {
        **BASELINE,
        "scheduler_policy": "cosine_decay",
        "scheduler_min_lr_ratio": 0.1,
        "checkpoint_every": 250,
    }
    manifest = MODULE.build_manifest(
        packet("primary", primary_knobs), packet("operational", operational_knobs)
    )
    assert manifest["promotion_stage"] == "COMBINED_PILOT"
    assert manifest["selected_knobs"]["learning_rate"] == 1e-4
    assert manifest["selected_knobs"]["scheduler_policy"] == "cosine_decay"
    assert manifest["changed_knobs"] == ["learning_rate", "checkpoint_every", "scheduler"]


def court_full_promotion_requires_matching_combined_pilot() -> None:
    primary = packet("primary", BASELINE)
    operational = packet("operational", BASELINE)
    terminal = {
        "status": "PASS",
        "promotion_eligible": True,
        "checkpoint_readback": "PASS",
        "best_checkpoint_readback": "PASS",
        "skipped_batches": 0,
        "nonfinite_events": 0,
        "mlflow_errors": [],
        "mlflow_run_id": "combined",
    }
    manifest = MODULE.build_manifest(
        primary,
        operational,
        combined_terminal=terminal,
        combined_contract={"knobs": BASELINE},
    )
    assert manifest["promotion_stage"] == "FULL_CORPUS"
    assert manifest["evidence_run_ids"] == ["primary", "operational", "combined"]
    try:
        MODULE.build_manifest(
            primary,
            operational,
            combined_terminal=terminal,
            combined_contract={"knobs": {**BASELINE, "learning_rate": 1e-4}},
        )
    except SystemExit as error:
        assert "knobs do not match" in str(error)
    else:
        raise AssertionError("mismatched combined pilot did not turn court RED")


def main() -> None:
    courts = [
        court_merges_only_operational_keys,
        court_full_promotion_requires_matching_combined_pilot,
    ]
    for court in courts:
        court()
        print(f"{court.__name__}=PASS")
    print("BRANCH50_PROMOTION_MANIFEST_COURTS=PASS")


if __name__ == "__main__":
    main()
