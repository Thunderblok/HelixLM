#!/usr/bin/env python3
"""Hostile courts for exact converted-parquet subset selection."""

from __future__ import annotations

import copy
from pathlib import Path

from contract import ContractError, load_contract, validate_contract
from dataset_projection import projection_data_files


def main() -> None:
    contract = load_contract(Path(__file__).resolve().parent / "evaluation_contract.json")
    expected_subsets = {
        "helix_arc_easy": "ARC-Easy",
        "helix_hellaswag": "default",
        "helix_openbookqa": "main",
        "helix_piqa": "plain_text",
        "helix_winogrande": "winogrande_xl",
    }
    for task in contract["tasks"]:
        files = projection_data_files(task)
        subset = expected_subsets[task["name"]]
        assert all(f"/{subset}/" in path for paths in files.values() for path in paths)
        assert all(f"@{task['hf_revision']}/" in path for paths in files.values() for path in paths)

    hostile = copy.deepcopy(contract)
    hostile["tasks"][0]["projection_files"]["test"] = []
    try:
        validate_contract(hostile)
    except ContractError:
        pass
    else:
        raise AssertionError("empty projection split did not turn the court RED")

    hostile = copy.deepcopy(contract)
    hostile["tasks"][0]["projection_files"]["test"] = ["../challenge/test.parquet"]
    try:
        validate_contract(hostile)
    except ContractError:
        pass
    else:
        raise AssertionError("projection path escape did not turn the court RED")

    print("LIGHTEVAL_DATASET_PROJECTION_COURTS=PASS")


if __name__ == "__main__":
    main()
