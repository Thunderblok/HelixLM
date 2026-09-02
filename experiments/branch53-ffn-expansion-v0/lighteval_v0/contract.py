"""Fail-closed contract for the Branch50/Branch53 paired Lighteval court."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "helix.lighteval.paired-evaluation.v0"
EXPECTED_TASKS = (
    "helix_arc_easy",
    "helix_hellaswag",
    "helix_openbookqa",
    "helix_piqa",
    "helix_winogrande",
)
EXPECTED_CHECKPOINTS = (
    "branch50_lr1p5e4_ffn2p5_full_best",
    "branch53_lr2e4_ffn2p5_full_best",
)
SHA256_HEX_LENGTH = 64


class ContractError(RuntimeError):
    """The frozen evaluation identity is incomplete or has drifted."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    try:
        contract = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"contract unavailable: {exc}") from exc
    validate_contract(contract)
    return contract


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("schema version drift")

    evaluator = contract.get("evaluator", {})
    if evaluator.get("name") != "lighteval" or evaluator.get("version") != "0.13.0":
        raise ContractError("Lighteval identity drift")
    if evaluator.get("dependency_lock") != "runtime.lock.json":
        raise ContractError("dependency lock is not bound")

    adapter = contract.get("model_adapter", {})
    required_adapter = {
        "kind": "registered_local_transformers_export_v0",
        "registration_module": "helix_lm.hf_model",
        "config_class": "HelixConfig",
        "model_class": "HelixForCausalLM",
        "tokenizer_class": "HelixTokenizer",
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "device": "cuda",
        "batch_size": 8,
        "max_length": 512,
        "add_special_tokens": True,
        "skip_special_tokens": True,
        "pairwise_tokenization": False,
        "continuous_batching": False,
        "compile": False,
    }
    if adapter != required_adapter:
        raise ContractError("model adapter drift")

    sampling = contract.get("sampling", {})
    if sampling != {
        "few_shots": 0,
        "few_shot_seeds": 1,
        "random_seed": 1234,
        "max_samples": None,
        "dataset_loading_processes": 1,
    }:
        raise ContractError("sampling contract drift or partial evaluation requested")

    scoring = contract.get("scoring", {})
    if scoring != {
        "comparison_unit": "task_primary_metric",
        "aggregate": "unweighted_macro_mean_of_five_task_primary_metrics",
        "sample_weighted_aggregate": False,
        "winner_rule": "report_per_task_deltas_and_macro_delta_without_promotion",
        "tie_rule": "exact_numeric_equality",
        "higher_is_better": True,
    }:
        raise ContractError("scoring semantics drift")

    tasks = contract.get("tasks")
    if not isinstance(tasks, list) or tuple(task.get("name") for task in tasks) != EXPECTED_TASKS:
        raise ContractError("task set or order drift")
    for task in tasks:
        revision = task.get("hf_revision")
        if not _is_git_commit(revision):
            raise ContractError(f"dataset revision is not an exact commit: {task.get('name')}")
        if not _is_git_commit(task.get("source_revision")):
            raise ContractError(f"dataset source revision is not an exact commit: {task.get('name')}")
        if task.get("dataset_projection") != "hf_refs_convert_parquet":
            raise ContractError(f"dataset projection drift: {task.get('name')}")
        projection_files = task.get("projection_files")
        if not isinstance(projection_files, dict) or set(projection_files) != {
            "train",
            "validation",
            "test",
        }:
            raise ContractError(f"dataset projection files incomplete: {task.get('name')}")
        for split, files in projection_files.items():
            if not isinstance(files, list) or not files:
                raise ContractError(f"dataset projection split empty: {task.get('name')}:{split}")
            if any(
                not isinstance(filename, str)
                or filename.startswith("/")
                or ".." in Path(filename).parts
                or not filename.endswith(".parquet")
                for filename in files
            ):
                raise ContractError(f"dataset projection path invalid: {task.get('name')}:{split}")
        splits = task.get("evaluation_splits")
        if not isinstance(splits, list) or len(splits) != 1:
            raise ContractError(f"evaluation split is not singular: {task.get('name')}")
        if task.get("metric") not in {"exact_match", "loglikelihood_acc"}:
            raise ContractError(f"unadmitted metric: {task.get('name')}")

    checkpoints = contract.get("checkpoints")
    if not isinstance(checkpoints, list) or tuple(item.get("id") for item in checkpoints) != EXPECTED_CHECKPOINTS:
        raise ContractError("checkpoint set or order drift")
    for checkpoint in checkpoints:
        if checkpoint.get("ffn_expansion") != 2.5:
            raise ContractError("FFN geometry drifted before downstream gate")
        if checkpoint.get("parameter_count") != 53_592_340:
            raise ContractError("checkpoint parameter count drift")
        for key in ("checkpoint_sha256", "model_root_sha256", "export_manifest_root_sha256"):
            digest = checkpoint.get(key)
            if not isinstance(digest, str) or len(digest) != SHA256_HEX_LENGTH:
                raise ContractError(f"invalid {key}: {checkpoint.get('id')}")
        export_path = checkpoint.get("export_path")
        if not isinstance(export_path, str) or not Path(export_path).is_absolute():
            raise ContractError(f"checkpoint export path is not absolute: {checkpoint.get('id')}")


def contract_root(contract: dict[str, Any]) -> str:
    validate_contract(contract)
    return sha256_bytes(canonical_json(contract))


def validate_runtime_lock(root: Path, installed_freeze: Path) -> dict[str, Any]:
    lock_identity_path = root / "runtime.lock.json"
    try:
        identity = json.loads(lock_identity_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"runtime lock identity unavailable: {exc}") from exc

    if identity.get("schema_version") != "helix.lighteval.runtime-lock.v0":
        raise ContractError("runtime lock schema drift")
    if identity.get("python") != "3.14.2" or identity.get("package_count") != 158:
        raise ContractError("runtime interpreter or package count drift")

    expected_files = {
        "runtime.requirements.in": identity.get("requirements_input_sha256"),
        "runtime.requirements.lock": identity.get("requirements_lock_sha256"),
    }
    for filename, expected_digest in expected_files.items():
        observed_digest = sha256_bytes((root / filename).read_bytes())
        if observed_digest != expected_digest:
            raise ContractError(f"runtime lock file drift: {filename}")

    if sha256_bytes(installed_freeze.read_bytes()) != identity.get("installed_freeze_sha256"):
        raise ContractError("installed package freeze drift")
    if len(installed_freeze.read_text().splitlines()) != identity["package_count"]:
        raise ContractError("installed package cardinality drift")

    for name in ("nltk_punkt", "nltk_punkt_tab"):
        resource = identity.get("auxiliary_resources", {}).get(name, {})
        resource_path = Path(resource.get("archive_path", ""))
        if not resource_path.is_file():
            raise ContractError(f"isolated auxiliary resource unavailable: {name}")
        if sha256_bytes(resource_path.read_bytes()) != resource.get("archive_sha256"):
            raise ContractError(f"isolated auxiliary resource drift: {name}")
    return identity


def validate_checkpoint_export(checkpoint: dict[str, Any]) -> dict[str, Any]:
    export_path = Path(checkpoint["export_path"])
    receipt_path = export_path.with_name(export_path.name.removesuffix("-export") + "-receipt.json")
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"checkpoint receipt unavailable: {checkpoint['id']}: {exc}") from exc

    receipt_checkpoint = receipt.get("checkpoint", {})
    receipt_export = receipt.get("export", {})
    if receipt.get("status") != "PASS" or receipt.get("lighteval_executed") is not False:
        raise ContractError(f"checkpoint preflight receipt drift: {checkpoint['id']}")
    if receipt_checkpoint.get("sha256") != checkpoint["checkpoint_sha256"]:
        raise ContractError(f"checkpoint SHA drift: {checkpoint['id']}")
    if receipt_checkpoint.get("observed_model_root") != checkpoint["model_root_sha256"]:
        raise ContractError(f"checkpoint model root drift: {checkpoint['id']}")
    if receipt_export.get("manifest_root") != checkpoint["export_manifest_root_sha256"]:
        raise ContractError(f"export manifest root drift: {checkpoint['id']}")

    files = receipt_export.get("files")
    if not isinstance(files, list) or not files:
        raise ContractError(f"export manifest empty: {checkpoint['id']}")
    observed_manifest_root = sha256_bytes(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    )
    if observed_manifest_root != checkpoint["export_manifest_root_sha256"]:
        raise ContractError(f"export manifest encoding drift: {checkpoint['id']}")

    observed_paths: set[str] = set()
    for item in files:
        relative = item.get("path")
        if not isinstance(relative, str) or relative in observed_paths:
            raise ContractError(f"export manifest path drift: {checkpoint['id']}")
        observed_paths.add(relative)
        path = export_path / relative
        if not path.is_file():
            raise ContractError(f"export file unavailable: {checkpoint['id']}: {relative}")
        if path.stat().st_size != item.get("size_bytes"):
            raise ContractError(f"export file size drift: {checkpoint['id']}: {relative}")
        if sha256_bytes(path.read_bytes()) != item.get("sha256"):
            raise ContractError(f"export file hash drift: {checkpoint['id']}: {relative}")
    return receipt


def validate_paired_receipts(receipts: list[dict[str, Any]], expected_contract_root: str) -> None:
    if len(receipts) != len(EXPECTED_CHECKPOINTS):
        raise ContractError("paired evaluation requires exactly two receipts")
    if tuple(receipt.get("checkpoint_id") for receipt in receipts) != EXPECTED_CHECKPOINTS:
        raise ContractError("paired receipt checkpoint order drift")
    for receipt in receipts:
        if receipt.get("contract_root") != expected_contract_root:
            raise ContractError("asymmetric evaluation contract")
        if receipt.get("max_samples") is not None:
            raise ContractError("partial evaluation cannot enter comparison")
        if tuple(receipt.get("tasks", ())) != EXPECTED_TASKS:
            raise ContractError("receipt task set drift")
        if receipt.get("status") != "complete":
            raise ContractError("incomplete evaluation cannot enter comparison")


def _is_git_commit(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 40:
        return False
    return all(character in "0123456789abcdef" for character in value)
