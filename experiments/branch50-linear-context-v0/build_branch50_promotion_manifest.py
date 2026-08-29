#!/usr/bin/env python3
"""Build a Branch-50 promotion manifest from matched comparison packets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


OPERATIONAL_KEYS = (
    "scheduler_policy",
    "scheduler_min_lr_ratio",
    "checkpoint_every",
)


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SystemExit(f"REFUSED: expected JSON object: {path}")
    return value


def canonical_root(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def changed_knobs(baseline: dict[str, Any], selected: dict[str, Any]) -> list[str]:
    changed = [
        key
        for key, baseline_value in baseline.items()
        if not key.startswith("scheduler_") and selected[key] != baseline_value
    ]
    if (
        selected["scheduler_policy"] != baseline["scheduler_policy"]
        or selected["scheduler_min_lr_ratio"] != baseline["scheduler_min_lr_ratio"]
    ):
        changed.append("scheduler")
    return changed


def evidence_ids(packet: dict[str, Any]) -> list[str]:
    rows = packet.get("ordered_candidates")
    if not isinstance(rows, list) or not rows:
        raise SystemExit("REFUSED: comparison packet has no candidate evidence")
    run_ids = [row.get("mlflow_run_id") for row in rows if isinstance(row, dict)]
    if any(not isinstance(run_id, str) or not run_id for run_id in run_ids):
        raise SystemExit("REFUSED: comparison packet has invalid MLflow evidence")
    return run_ids


def build_manifest(
    primary: dict[str, Any],
    operational: dict[str, Any],
    *,
    combined_terminal: dict[str, Any] | None = None,
    combined_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    for label, packet in (("primary", primary), ("operational", operational)):
        if packet.get("schema") != "helix.branch50.ablation-comparison.v0":
            raise SystemExit(f"REFUSED: {label} comparison schema mismatch")
        if packet.get("status") != "PASS":
            raise SystemExit(f"REFUSED: {label} comparison did not pass")
    selected = dict(primary["selected_knobs"])
    operational_selected = operational["selected_knobs"]
    for key in OPERATIONAL_KEYS:
        selected[key] = operational_selected[key]
    baseline = {
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
    if set(selected) != set(baseline):
        raise SystemExit("REFUSED: selected knob schema mismatch")
    run_ids = list(dict.fromkeys(evidence_ids(primary) + evidence_ids(operational)))
    stage = "COMBINED_PILOT"
    if combined_terminal is not None or combined_contract is not None:
        if combined_terminal is None or combined_contract is None:
            raise SystemExit("REFUSED: incomplete combined-pilot evidence")
        if (
            combined_terminal.get("status") != "PASS"
            or combined_terminal.get("promotion_eligible") is not True
            or combined_terminal.get("checkpoint_readback") != "PASS"
            or combined_terminal.get("best_checkpoint_readback") != "PASS"
            or combined_terminal.get("skipped_batches") != 0
            or combined_terminal.get("nonfinite_events") != 0
            or combined_terminal.get("mlflow_errors")
        ):
            raise SystemExit("REFUSED: combined pilot is not promotable")
        if combined_contract.get("knobs") != selected:
            raise SystemExit("REFUSED: combined pilot knobs do not match selection")
        combined_run_id = combined_terminal.get("mlflow_run_id")
        if not isinstance(combined_run_id, str) or not combined_run_id:
            raise SystemExit("REFUSED: combined pilot has no MLflow run ID")
        run_ids.append(combined_run_id)
        stage = "FULL_CORPUS"

    manifest = {
        "schema": "helix.branch50.promotion-decision.v0",
        "status": "PROMOTED",
        "promotion_stage": stage,
        "selected_knobs": selected,
        "changed_knobs": changed_knobs(baseline, selected),
        "evidence_run_ids": run_ids,
        "primary_comparison_root": primary["packet_root"],
        "operational_comparison_root": operational["packet_root"],
        "decision": (
            "Combined settings admitted for a matched 100M pilot by preregistered "
            "fixed-validation and operational comparison courts."
            if stage == "COMBINED_PILOT"
            else "Combined settings admitted for one full corpus pass after a lawful "
            "combined 100M pilot and live recovery courts."
        ),
    }
    manifest["decision_root"] = canonical_root(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-comparison", type=Path, required=True)
    parser.add_argument("--operational-comparison", type=Path, required=True)
    parser.add_argument("--combined-run-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    combined_terminal = None
    combined_contract = None
    if args.combined_run_root:
        combined_terminal = load_object(args.combined_run_root / "terminal.json")
        combined_contract = load_object(args.combined_run_root / "ablation_contract.json")
    manifest = build_manifest(
        load_object(args.primary_comparison),
        load_object(args.operational_comparison),
        combined_terminal=combined_terminal,
        combined_contract=combined_contract,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"REFUSED: output already exists: {args.output}")
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
