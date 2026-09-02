#!/usr/bin/env python3
"""Hostile courts for paired Lighteval scoring and receipt symmetry."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import compare_results
from contract import contract_root, load_contract


ROOT = Path(__file__).resolve().parent


def write_pair(root: Path, *, missing_metric: bool = False) -> None:
    contract = load_contract(ROOT / "evaluation_contract.json")
    frozen_root = contract_root(contract)
    for index, checkpoint in enumerate(contract["checkpoints"]):
        checkpoint_root = root / checkpoint["id"]
        checkpoint_root.mkdir(parents=True)
        receipt = {
            "checkpoint_id": checkpoint["id"],
            "contract_root": frozen_root,
            "max_samples": None,
            "tasks": [task["name"] for task in contract["tasks"]],
            "status": "complete",
        }
        result_rows = {}
        for task_index, task in enumerate(contract["tasks"]):
            metric = compare_results.METRIC_KEYS[task["metric"]]
            result_rows[f"{task['name']}|0"] = {metric: 0.1 * (index + task_index)}
        if missing_metric:
            result_rows["helix_piqa|0"] = {}
        (checkpoint_root / "execution_receipt.json").write_text(json.dumps(receipt))
        (checkpoint_root / "results.json").write_text(json.dumps({"results": result_rows}))


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        pair_root = Path(directory)
        write_pair(pair_root)
        comparison = compare_results.compare(pair_root)
        assert comparison["promotion_decision"] == "not_made"
        assert comparison["macro_delta_candidate_minus_control"] > 0
        assert len(comparison["task_deltas_candidate_minus_control"]) == 5

    with tempfile.TemporaryDirectory() as directory:
        pair_root = Path(directory)
        write_pair(pair_root, missing_metric=True)
        try:
            compare_results.compare(pair_root)
        except ValueError:
            pass
        else:
            raise AssertionError("missing primary metric did not turn the court RED")

    with tempfile.TemporaryDirectory() as directory:
        pair_root = Path(directory)
        write_pair(pair_root)
        contract = load_contract(ROOT / "evaluation_contract.json")
        candidate = pair_root / contract["checkpoints"][1]["id"] / "execution_receipt.json"
        receipt = json.loads(candidate.read_text())
        hostile = copy.deepcopy(receipt)
        hostile["max_samples"] = 1
        candidate.write_text(json.dumps(hostile))
        try:
            compare_results.compare(pair_root)
        except Exception:
            pass
        else:
            raise AssertionError("partial paired receipt did not turn the court RED")

    print("LIGHTEVAL_PAIRED_SCORING_COURTS=PASS")


if __name__ == "__main__":
    main()
