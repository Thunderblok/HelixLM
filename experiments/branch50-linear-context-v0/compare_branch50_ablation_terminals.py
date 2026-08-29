#!/usr/bin/env python3
"""Compare lawful Branch-50 ablation terminals under a preregistered rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def canonical_root(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise SystemExit(f"REFUSED: expected JSON object: {path}")
    return value


def load_spool(path: Path) -> list[dict[str, Any]]:
    events = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise SystemExit(f"REFUSED: non-object spool event {path}:{line_number}")
        events.append(value)
    return events


def candidate_record(run_root: Path) -> dict[str, Any]:
    terminal = load_json(run_root / "terminal.json")
    contract = load_json(run_root / "ablation_contract.json")
    events = load_spool(run_root / "mlflow_spool.jsonl")
    if terminal.get("status") != "PASS" or terminal.get("promotion_eligible") is not True:
        raise SystemExit(f"REFUSED: non-promotable terminal: {run_root}")
    if (
        terminal.get("checkpoint_readback") != "PASS"
        or terminal.get("best_checkpoint_readback") != "PASS"
        or terminal.get("skipped_batches") != 0
        or terminal.get("nonfinite_events") != 0
        or terminal.get("mlflow_errors")
    ):
        raise SystemExit(f"REFUSED: unhealthy terminal: {run_root}")

    validation = {}
    last_train = None
    for event in events:
        if event.get("event") != "metrics":
            continue
        metrics = event.get("metrics", {})
        step = int(event.get("step", 0))
        if event.get("phase") == "validation" and "val/loss" in metrics:
            validation[step] = {
                "loss": float(metrics["val/loss"]),
                "ppl": float(metrics["val/ppl"]),
                "targets": int(metrics["val/causal_targets"]),
            }
        if event.get("phase") == "train" and "train/causal_targets_per_second" in metrics:
            last_train = {
                "step": step,
                "causal_targets_per_second": float(metrics["train/causal_targets_per_second"]),
                "raw_tokens_per_second": float(metrics["train/raw_tokens_per_second"]),
                "accum_loss": float(metrics["train/accum_loss"]),
                "accum_ppl": float(metrics["train/accum_ppl"]),
            }
    if not validation or last_train is None:
        raise SystemExit(f"REFUSED: incomplete metric spool: {run_root}")

    final_step = int(terminal["steps"])
    if final_step not in validation:
        raise SystemExit(f"REFUSED: no fixed validation at terminal step: {run_root}")
    if not math.isclose(
        float(terminal["last_val_loss"]), validation[final_step]["loss"], abs_tol=0.0
    ):
        raise SystemExit(f"REFUSED: terminal/spool validation mismatch: {run_root}")

    return {
        "run_root": str(run_root.resolve()),
        "ablation_id": terminal["ablation_id"],
        "mlflow_run_id": terminal["mlflow_run_id"],
        "contract_root": canonical_root(contract),
        "changed_knobs": contract["changed_knobs"],
        "knobs": contract["knobs"],
        "source_identity": contract["source_identity"],
        "manifest_roots": contract["manifest_roots"],
        "seq_len": contract["seq_len"],
        "batch_size": contract["batch_size"],
        "grad_accum": contract["grad_accum"],
        "target_causal_targets": contract["target_causal_targets"],
        "target_raw_tokens": contract["target_raw_tokens"],
        "eval_every": contract["eval_every"],
        "validation_batches": contract["validation_batches"],
        "seed": contract["seed"],
        "steps": final_step,
        "validation": validation,
        "final_val_loss": validation[final_step]["loss"],
        "final_val_ppl": validation[final_step]["ppl"],
        "best_val_loss": float(terminal["best_val_loss"]),
        "best_val_ppl": float(terminal["best_val_ppl"]),
        "best_val_step": int(terminal["best_val_step"]),
        "last_train": last_train,
        "peak_vram_bytes": int(terminal["peak_vram_bytes"]),
        "checkpoint_sha256": terminal["checkpoint_sha256"],
        "final_model_root": terminal["final_model_root"],
    }


def comparison_identity(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "source_identity",
            "manifest_roots",
            "seq_len",
            "batch_size",
            "grad_accum",
            "target_causal_targets",
            "target_raw_tokens",
            "eval_every",
            "validation_batches",
            "seed",
            "steps",
        )
    }


def select_candidate(
    records: list[dict[str, Any]], *, material_loss_delta: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if material_loss_delta < 0:
        raise SystemExit("REFUSED: material loss delta must be nonnegative")
    ordered = sorted(records, key=lambda item: item["final_val_loss"])
    best_loss = ordered[0]["final_val_loss"]
    tied = [item for item in ordered if item["final_val_loss"] - best_loss <= material_loss_delta]
    selected = sorted(
        tied,
        key=lambda item: (
            len(item["changed_knobs"]),
            -item["last_train"]["causal_targets_per_second"],
            item["ablation_id"],
        ),
    )[0]
    return selected, ordered


def build_packet(
    records: list[dict[str, Any]], *, material_loss_delta: float
) -> dict[str, Any]:
    if len(records) < 2:
        raise SystemExit("REFUSED: comparison requires at least two terminals")
    identities = [comparison_identity(record) for record in records]
    if any(identity != identities[0] for identity in identities[1:]):
        raise SystemExit("REFUSED: candidate comparison identity mismatch")
    validation_steps = [sorted(record["validation"]) for record in records]
    if any(steps != validation_steps[0] for steps in validation_steps[1:]):
        raise SystemExit("REFUSED: fixed validation step mismatch")
    selected, ordered = select_candidate(records, material_loss_delta=material_loss_delta)
    packet = {
        "schema": "helix.branch50.ablation-comparison.v0",
        "status": "PASS",
        "selection_law": {
            "primary": "lowest fixed-validation loss at the shared terminal step",
            "material_loss_delta": material_loss_delta,
            "tie_break_1": "fewest changed knobs",
            "tie_break_2": "highest terminal causal-target throughput",
            "tie_break_3": "lexical ablation ID",
            "best_validation": "reported as secondary evidence only",
        },
        "comparison_identity": identities[0],
        "validation_steps": validation_steps[0],
        "selected_ablation_id": selected["ablation_id"],
        "selected_mlflow_run_id": selected["mlflow_run_id"],
        "selected_knobs": selected["knobs"],
        "selected_changed_knobs": selected["changed_knobs"],
        "ordered_candidates": ordered,
    }
    packet["packet_root"] = canonical_root(packet)
    return packet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--material-loss-delta", type=float, default=0.005)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [candidate_record(path.resolve()) for path in args.run_root]
    packet = build_packet(records, material_loss_delta=args.material_loss_delta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"REFUSED: output already exists: {args.output}")
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps(packet, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
