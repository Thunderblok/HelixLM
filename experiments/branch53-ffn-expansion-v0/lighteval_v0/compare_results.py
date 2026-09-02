#!/usr/bin/env python3
"""Compare exactly two complete results under the frozen scoring semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from contract import canonical_json, contract_root, load_contract, sha256_bytes, validate_paired_receipts


ROOT = Path(__file__).resolve().parent
METRIC_KEYS = {"loglikelihood_acc": "acc", "exact_match": "em"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-root", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def extract_scores(results: dict[str, Any], contract: dict[str, Any]) -> dict[str, float]:
    observed = results.get("results", {})
    scores: dict[str, float] = {}
    for task in contract["tasks"]:
        task_result = observed.get(f"{task['name']}|0")
        metric_key = METRIC_KEYS[task["metric"]]
        if not isinstance(task_result, dict) or not isinstance(task_result.get(metric_key), (int, float)):
            raise ValueError(f"missing primary metric: {task['name']}:{metric_key}")
        scores[task["name"]] = float(task_result[metric_key])
    return scores


def compare(pair_root: Path) -> dict[str, Any]:
    contract = load_contract(ROOT / "evaluation_contract.json")
    root = contract_root(contract)
    checkpoint_dirs = [pair_root / checkpoint["id"] for checkpoint in contract["checkpoints"]]
    receipts = [load_json(path / "execution_receipt.json") for path in checkpoint_dirs]
    validate_paired_receipts(receipts, root)
    results = [load_json(path / "results.json") for path in checkpoint_dirs]
    scores = [extract_scores(result, contract) for result in results]
    macros = [fmean(task_scores.values()) for task_scores in scores]
    task_deltas = {name: scores[1][name] - scores[0][name] for name in scores[0]}
    return {
        "schema_version": "helix.lighteval.paired-comparison.v0",
        "status": "complete",
        "contract_root": root,
        "checkpoints": [
            {
                "id": checkpoint["id"],
                "scores": score,
                "macro_mean": macro,
                "execution_receipt_sha256": sha256_bytes(
                    (checkpoint_dir / "execution_receipt.json").read_bytes()
                ),
                "results_sha256": sha256_bytes((checkpoint_dir / "results.json").read_bytes()),
            }
            for checkpoint, checkpoint_dir, score, macro in zip(
                contract["checkpoints"], checkpoint_dirs, scores, macros, strict=True
            )
        ],
        "task_deltas_candidate_minus_control": task_deltas,
        "macro_delta_candidate_minus_control": macros[1] - macros[0],
        "winner_rule": contract["scoring"]["winner_rule"],
        "promotion_decision": "not_made",
        "production_effect": "none",
    }


def main() -> None:
    args = parse_args()
    comparison_path = args.pair_root / "paired_comparison.json"
    if comparison_path.exists():
        raise SystemExit(f"REFUSED: comparison already exists: {comparison_path}")
    comparison = compare(args.pair_root)
    encoded = canonical_json(comparison)
    comparison_path.write_bytes(encoded)
    print(f"LIGHTEVAL_PAIRED_COMPARISON_SHA256={sha256_bytes(encoded)}")
    print("LIGHTEVAL_PAIRED_COMPARISON=PASS")


if __name__ == "__main__":
    main()
