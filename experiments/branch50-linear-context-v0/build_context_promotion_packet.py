#!/usr/bin/env python3
"""Build a deterministic Branch-50 512-vs-1024 promotion packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ALLOWED_CONFIG_DIFFERENCES = {
    "batch_size",
    "max_position_embeddings",
    "seq_len",
}

MATCHED_IDENTITY_FIELDS = {
    "common_runner_sha256",
    "harness_sha256",
    "model_base_head",
    "model_base_tree",
    "model_source_diff",
    "source_dirty",
    "source_head",
    "source_tree",
    "train_manifest_sha256",
    "val_manifest_sha256",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_spool(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"non-object spool row {path}:{line_number}")
        rows.append(value)
    return rows


def phase_metrics(
    rows: list[dict[str, Any]], phase: str
) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for row in rows:
        if row.get("event") != "metrics" or row.get("phase") != phase:
            continue
        step = int(row["step"])
        metrics = row["metrics"]
        if not isinstance(metrics, dict):
            raise ValueError(f"metrics at step {step} are not an object")
        numeric = {key: float(value) for key, value in metrics.items()}
        for key, value in numeric.items():
            if not math.isfinite(value):
                raise ValueError(f"nonfinite metric {phase}:{step}:{key}={value}")
        result[step] = numeric
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in ALLOWED_CONFIG_DIFFERENCES
    }


def load_run(root: Path, expected_seq_len: int) -> dict[str, Any]:
    terminal_path = root / "terminal.json"
    config_path = root / "resolved_config.json"
    spool_path = root / "mlflow_spool.jsonl"
    require(terminal_path.is_file(), f"terminal missing: {terminal_path}")
    require(config_path.is_file(), f"config missing: {config_path}")
    require(spool_path.is_file(), f"spool missing: {spool_path}")

    terminal = read_json(terminal_path)
    config = read_json(config_path)
    spool = read_spool(spool_path)
    train = phase_metrics(spool, "train")
    validation = phase_metrics(spool, "validation")

    require(terminal.get("status") == "PASS", f"terminal not PASS: {root}")
    require(
        terminal.get("checkpoint_readback") == "PASS",
        f"checkpoint readback not PASS: {root}",
    )
    require(not terminal.get("mlflow_errors"), f"MLflow errors present: {root}")
    require(int(terminal["seq_len"]) == expected_seq_len, f"wrong seq_len: {root}")
    require(train, f"training metrics missing: {root}")
    require(validation, f"validation metrics missing: {root}")
    require(
        any(row.get("event") == "run_finished" for row in spool),
        f"run_finished missing: {root}",
    )
    step = int(terminal["steps"])
    require(step in train, f"terminal train metric missing at step {step}: {root}")
    require(
        step in validation,
        f"terminal validation metric missing at step {step}: {root}",
    )

    checkpoint = Path(str(terminal["checkpoint"]))
    require(checkpoint.is_file(), f"checkpoint missing: {checkpoint}")
    checkpoint_actual = sha256(checkpoint)
    require(
        checkpoint_actual == terminal["checkpoint_sha256"],
        f"checkpoint SHA mismatch: {checkpoint}",
    )

    return {
        "terminal": terminal,
        "config": config,
        "train": train,
        "validation": validation,
        "roots": {
            "terminal_sha256": sha256(terminal_path),
            "resolved_config_sha256": sha256(config_path),
            "mlflow_spool_sha256": sha256(spool_path),
            "checkpoint_sha256": checkpoint_actual,
        },
    }


def build_packet(args: argparse.Namespace) -> dict[str, Any]:
    control = load_run(args.control.resolve(), 512)
    candidate = load_run(args.candidate.resolve(), 1024)
    identity = read_json(args.identity.resolve())
    ct = control["terminal"]
    xt = candidate["terminal"]

    identity_runs = identity.get("runs")
    require(isinstance(identity_runs, dict), "identity runs are missing")
    control_identity = identity_runs.get("control")
    candidate_identity = identity_runs.get("candidate")
    require(isinstance(control_identity, dict), "control identity is missing")
    require(isinstance(candidate_identity, dict), "candidate identity is missing")
    require(
        control_identity.get("mlflow_run_id") == ct["mlflow_run_id"],
        "control identity run ID mismatch",
    )
    require(
        candidate_identity.get("mlflow_run_id") == xt["mlflow_run_id"],
        "candidate identity run ID mismatch",
    )
    control_bound_identity = {
        field: control_identity.get(field) for field in MATCHED_IDENTITY_FIELDS
    }
    candidate_bound_identity = {
        field: candidate_identity.get(field) for field in MATCHED_IDENTITY_FIELDS
    }
    require(
        all(value not in (None, "") for value in control_bound_identity.values()),
        "control identity has unbound fields",
    )
    require(
        control_bound_identity == candidate_bound_identity,
        "control and candidate source identities differ",
    )

    matched_terminal_fields = [
        "steps",
        "grad_accum",
        "raw_tokens_per_optimizer_step",
        "causal_targets_per_optimizer_step",
        "base_blocks_per_optimizer_step",
        "target_causal_targets",
        "aligned_target_causal_targets",
        "initial_model_root",
    ]
    mismatches = {
        field: {"control": ct.get(field), "candidate": xt.get(field)}
        for field in matched_terminal_fields
        if ct.get(field) != xt.get(field)
    }
    require(not mismatches, f"terminal contract mismatch: {mismatches}")
    require(
        normalized_config(control["config"])
        == normalized_config(candidate["config"]),
        "resolved configs differ outside seq_len/batch_size/max_position_embeddings",
    )
    require(
        int(ct["batch_size"]) * int(ct["seq_len"])
        == int(xt["batch_size"]) * int(xt["seq_len"]),
        "raw positions per microbatch differ",
    )

    common_validation_steps = sorted(
        set(control["validation"]) & set(candidate["validation"])
    )
    require(common_validation_steps, "no common validation steps")
    require(
        common_validation_steps[-1] == int(ct["steps"]),
        "final common validation step is not the terminal step",
    )
    final_step = common_validation_steps[-1]
    control_final = control["validation"][final_step]
    candidate_final = candidate["validation"][final_step]
    val_loss_delta = candidate_final["val/loss"] - control_final["val/loss"]

    reference_step = args.throughput_reference_step
    require(reference_step in control["train"], "control reference step missing")
    require(reference_step in candidate["train"], "candidate reference step missing")
    control_reference_tps = control["train"][reference_step][
        "train/causal_targets_per_second"
    ]
    candidate_reference_tps = candidate["train"][reference_step][
        "train/causal_targets_per_second"
    ]
    reference_tps_ratio = candidate_reference_tps / control_reference_tps
    control_terminal_tps = control["train"][final_step][
        "train/causal_targets_per_second"
    ]
    candidate_terminal_tps = candidate["train"][final_step][
        "train/causal_targets_per_second"
    ]
    terminal_tps_ratio = candidate_terminal_tps / control_terminal_tps

    quality_not_worse = val_loss_delta <= 0.0
    throughput_floor_pass = reference_tps_ratio >= 0.90
    if quality_not_worse and throughput_floor_pass:
        verdict = "SEED42_PASS_THREE_SEED_REQUIRED"
        recommendation = "RUN_ADDITIONAL_SEEDS"
    elif not quality_not_worse:
        verdict = "RETAIN_512_SEED42_QUALITY_GATE"
        recommendation = "STOP_1024_PROMOTION"
    else:
        verdict = "RETAIN_512_THROUGHPUT_GATE"
        recommendation = "STOP_1024_PROMOTION"

    validation_curve = []
    for step in common_validation_steps:
        control_metrics = control["validation"][step]
        candidate_metrics = candidate["validation"][step]
        validation_curve.append(
            {
                "step": step,
                "control_val_loss": control_metrics["val/loss"],
                "candidate_val_loss": candidate_metrics["val/loss"],
                "candidate_minus_control_val_loss": (
                    candidate_metrics["val/loss"] - control_metrics["val/loss"]
                ),
            }
        )

    return {
        "schema": "BRANCH50_CONTEXT_PROMOTION_V0",
        "court_status": "PASS",
        "verdict": verdict,
        "recommendation": recommendation,
        "seed": int(control["config"]["seed"]),
        "source_identity": control_bound_identity,
        "identity_evidence_sha256": sha256(args.identity.resolve()),
        "matched_contract": {
            field: ct[field] for field in matched_terminal_fields
        },
        "allowed_config_differences": sorted(ALLOWED_CONFIG_DIFFERENCES),
        "control": {
            "seq_len": 512,
            "batch_size": int(ct["batch_size"]),
            "mlflow_run_id": ct["mlflow_run_id"],
            "final_validation_loss": control_final["val/loss"],
            "final_validation_ppl": control_final["val/ppl"],
            "reference_causal_targets_per_second": control_reference_tps,
            "terminal_causal_targets_per_second": control_terminal_tps,
            "peak_vram_bytes": int(ct["peak_vram_bytes"]),
            "evidence_roots": control["roots"],
        },
        "candidate": {
            "seq_len": 1024,
            "batch_size": int(xt["batch_size"]),
            "mlflow_run_id": xt["mlflow_run_id"],
            "final_validation_loss": candidate_final["val/loss"],
            "final_validation_ppl": candidate_final["val/ppl"],
            "reference_causal_targets_per_second": candidate_reference_tps,
            "terminal_causal_targets_per_second": candidate_terminal_tps,
            "peak_vram_bytes": int(xt["peak_vram_bytes"]),
            "evidence_roots": candidate["roots"],
        },
        "comparison": {
            "final_step": final_step,
            "candidate_minus_control_val_loss": val_loss_delta,
            "candidate_quality_not_worse": quality_not_worse,
            "throughput_reference_step": reference_step,
            "reference_throughput_ratio": reference_tps_ratio,
            "terminal_throughput_ratio": terminal_tps_ratio,
            "throughput_floor": 0.90,
            "throughput_floor_pass": throughput_floor_pass,
            "candidate_minus_control_peak_vram_bytes": (
                int(xt["peak_vram_bytes"]) - int(ct["peak_vram_bytes"])
            ),
        },
        "validation_curve": validation_curve,
        "limitations": [
            "One seed only; three-seed promotion is not established.",
            "The declared throughput gate uses the preregistered step-1600 reference metric; terminal cumulative throughput is retained separately and is below the floor.",
            "The court compares next-token validation NLL on the fixed held-out paired-block validation set; it is not a long-context capability benchmark.",
            "Skipped-batch, valid-row-nonfinite, and validation-nonfinite counters were not emitted by this runner and remain unavailable; strict_nan_check was enabled and every logged numeric metric was finite.",
            "No model publication or production effect is authorized.",
        ],
        "numerical_observation": {
            "strict_nan_check": bool(control["config"]["strict_nan_check"]),
            "logged_numeric_nonfinite_events": 0,
            "skipped_batches": "UNAVAILABLE_NOT_INSTRUMENTED",
            "valid_row_nonfinite_events": "UNAVAILABLE_NOT_INSTRUMENTED",
            "validation_nonfinite_batches": "UNAVAILABLE_NOT_INSTRUMENTED",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--throughput-reference-step", type=int, default=1600)
    args = parser.parse_args()
    packet = build_packet(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps(packet, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(
            json.dumps(
                {
                    "schema": "BRANCH50_CONTEXT_PROMOTION_V0",
                    "court_status": "HOLD",
                    "reason": str(error),
                },
                indent=2,
                sort_keys=True,
            )
        )
        sys.exit(2)
