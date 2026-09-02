#!/usr/bin/env python3
"""Hostile courts for the frozen paired Lighteval contract."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = ROOT / "evaluation_contract.json"
MODULE_PATH = ROOT / "contract.py"


def load_module():
    spec = importlib.util.spec_from_file_location("helix_lighteval_contract", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("contract module import unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_red(module, contract: dict, message: str) -> None:
    try:
        module.validate_contract(contract)
    except module.ContractError:
        return
    raise AssertionError(message)


def main() -> None:
    module = load_module()
    contract = json.loads(CONTRACT_PATH.read_text())
    module.validate_contract(contract)
    root = module.contract_root(contract)
    assert len(root) == 64

    runtime_lock = json.loads((ROOT / "runtime.lock.json").read_text())
    frozen = ROOT / "test-installed-freeze.txt"
    frozen.write_text("\n".join(f"package-{index}==1" for index in range(158)) + "\n")
    original_freeze_digest = runtime_lock["installed_freeze_sha256"]
    runtime_lock["installed_freeze_sha256"] = module.sha256_bytes(frozen.read_bytes())
    runtime_lock_path = ROOT / "runtime.lock.json"
    original_runtime_lock = runtime_lock_path.read_text()
    runtime_lock_path.write_text(json.dumps(runtime_lock, indent=2) + "\n")
    try:
        module.validate_runtime_lock(ROOT, frozen)
        frozen.write_text(frozen.read_text() + "unexpected==1\n")
        try:
            module.validate_runtime_lock(ROOT, frozen)
        except module.ContractError:
            pass
        else:
            raise AssertionError("installed freeze drift did not turn the court RED")
    finally:
        runtime_lock_path.write_text(original_runtime_lock)
        frozen.unlink()
    assert original_freeze_digest == "1cd336fa13442d910ad1e7a7ee941f960afa9ae28477caaeed2e5dd15d61d03e"

    for checkpoint in contract["checkpoints"]:
        module.validate_checkpoint_export(checkpoint)

    hostile = copy.deepcopy(contract)
    hostile["evaluator"]["version"] = "0.13.1"
    expect_red(module, hostile, "Lighteval version drift did not turn the court RED")

    hostile = copy.deepcopy(contract)
    hostile["tasks"][0]["hf_revision"] = "main"
    expect_red(module, hostile, "floating dataset revision did not turn the court RED")

    hostile = copy.deepcopy(contract)
    hostile["tasks"].pop()
    expect_red(module, hostile, "missing task did not turn the court RED")

    hostile = copy.deepcopy(contract)
    hostile["sampling"]["max_samples"] = 100
    expect_red(module, hostile, "partial sampling did not turn the court RED")

    hostile = copy.deepcopy(contract)
    hostile["model_adapter"]["trust_remote_code"] = True
    expect_red(module, hostile, "adapter trust drift did not turn the court RED")

    hostile = copy.deepcopy(contract)
    hostile["scoring"]["aggregate"] = "sample_weighted_mean"
    expect_red(module, hostile, "scoring drift did not turn the court RED")

    hostile = copy.deepcopy(contract)
    hostile["checkpoints"][1]["ffn_expansion"] = 3.0
    expect_red(module, hostile, "unmatched FFN geometry did not turn the court RED")

    hostile = copy.deepcopy(contract)
    hostile["checkpoints"][0]["parameter_count"] = 54_771_988
    expect_red(module, hostile, "unmatched parameter count did not turn the court RED")

    complete_receipts = [
        {
            "checkpoint_id": checkpoint["id"],
            "contract_root": root,
            "max_samples": None,
            "tasks": list(module.EXPECTED_TASKS),
            "status": "complete",
        }
        for checkpoint in contract["checkpoints"]
    ]
    module.validate_paired_receipts(complete_receipts, root)

    asymmetric = copy.deepcopy(complete_receipts)
    asymmetric[1]["contract_root"] = "0" * 64
    try:
        module.validate_paired_receipts(asymmetric, root)
    except module.ContractError:
        pass
    else:
        raise AssertionError("asymmetric evaluation did not turn the court RED")

    partial = copy.deepcopy(complete_receipts)
    partial[0]["max_samples"] = 10
    try:
        module.validate_paired_receipts(partial, root)
    except module.ContractError:
        pass
    else:
        raise AssertionError("partial receipt did not turn the court RED")

    print(f"LIGHTEVAL_PAIRED_CONTRACT_ROOT={root}")
    print("LIGHTEVAL_PAIRED_CONTRACT_COURTS=PASS")


if __name__ == "__main__":
    main()
