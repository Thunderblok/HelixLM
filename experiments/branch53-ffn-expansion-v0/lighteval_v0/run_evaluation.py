#!/usr/bin/env python3
"""Run one fail-closed Helix checkpoint evaluation under the frozen contract."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from contract import (
    ContractError,
    canonical_json,
    contract_root,
    load_contract,
    sha256_bytes,
    validate_checkpoint_export,
    validate_runtime_lock,
)
from dataset_projection import install_exact_parquet_loader


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[2]
FULL_TASKS = ",".join(
    [
        "helix_arc_easy|0",
        "helix_hellaswag|0",
        "helix_openbookqa|0",
        "helix_piqa|0",
        "helix_winogrande|0",
    ]
)
SMOKE_TASKS = "helix_arc_easy|0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--installed-freeze", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "suite-smoke", "full"), required=True)
    return parser.parse_args()


def package_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in (
            "datasets",
            "fsspec",
            "hf-xet",
            "huggingface-hub",
            "inspect-ai",
            "lighteval",
            "lxml",
            "pydantic",
            "tokenizers",
            "torch",
            "transformers",
            "triton",
        )
    }


def select_checkpoint(contract: dict[str, Any], checkpoint_id: str) -> dict[str, Any]:
    matches = [item for item in contract["checkpoints"] if item["id"] == checkpoint_id]
    if len(matches) != 1:
        raise ContractError(f"checkpoint is not admitted exactly once: {checkpoint_id}")
    return matches[0]


def write_json(path: Path, value: Any) -> str:
    encoded = canonical_json(value)
    path.write_bytes(encoded)
    return sha256_bytes(encoded)


def require_runtime() -> None:
    if package_versions()["lighteval"] != "0.13.0":
        raise ContractError("executing Lighteval version drift")
    if platform.python_version() != "3.14.2":
        raise ContractError("executing Python version drift")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ContractError("exactly one CUDA device is required")
    nltk_data = Path(os.environ.get("NLTK_DATA", ""))
    if nltk_data != Path("/home/mo/DEV/experiments/helix-lighteval-runtime-v0/cache/nltk"):
        raise ContractError("NLTK_DATA is not bound to the isolated runtime")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"REFUSED: nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    contract = load_contract(ROOT / "evaluation_contract.json")
    root = contract_root(contract)
    runtime_lock = validate_runtime_lock(ROOT, args.installed_freeze)
    checkpoint = select_checkpoint(contract, args.checkpoint_id)
    checkpoint_receipt = validate_checkpoint_export(checkpoint)
    require_runtime()

    if args.mode in {"suite-smoke", "full"}:
        tasks = FULL_TASKS
        receipt_tasks = [task["name"] for task in contract["tasks"]]
        max_samples = None if args.mode == "full" else 1
    else:
        tasks = SMOKE_TASKS
        receipt_tasks = ["helix_arc_easy"]
        max_samples = 1

    # Registration is explicit; remote-code loading remains forbidden. Python
    # starts with this script's directory on sys.path, so bind the repository
    # source root deliberately instead of relying on the operator's CWD.
    sys.path.insert(0, str(REPOSITORY_ROOT))
    import helix_lm.hf_model  # noqa: F401
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.models.transformers.transformers_model import TransformersModelConfig
    from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

    install_exact_parquet_loader(contract)

    cache_root = args.output_dir / "cache"
    tracker = EvaluationTracker(output_dir=str(args.output_dir / "lighteval"), save_details=True)
    model_config = TransformersModelConfig(
        model_name=checkpoint["export_path"],
        cache_dir=str(cache_root / "model"),
        batch_size=contract["model_adapter"]["batch_size"],
        max_length=contract["model_adapter"]["max_length"],
        dtype=contract["model_adapter"]["dtype"],
        device=contract["model_adapter"]["device"],
        trust_remote_code=False,
        add_special_tokens=True,
        skip_special_tokens=True,
        pairwise_tokenization=False,
        continuous_batching=False,
        compile=False,
    )
    parameters = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        dataset_loading_processes=contract["sampling"]["dataset_loading_processes"],
        custom_tasks_directory=str(ROOT / "custom_tasks.py"),
        num_fewshot_seeds=contract["sampling"]["few_shot_seeds"],
        max_samples=max_samples,
    )

    started_at = datetime.now(UTC).isoformat()
    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=parameters,
        evaluation_tracker=tracker,
        model_config=model_config,
    )
    pipeline.evaluate()
    pipeline.get_results()
    pipeline.save_and_push_results()
    completed_at = datetime.now(UTC).isoformat()

    saved_results = sorted((args.output_dir / "lighteval" / "results").rglob("results_*.json"))
    if len(saved_results) != 1:
        raise ContractError(f"expected one saved Lighteval result, observed {len(saved_results)}")
    results = json.loads(saved_results[0].read_text())
    results_sha256 = write_json(args.output_dir / "results.json", results)
    receipt = {
        "schema_version": "helix.lighteval.checkpoint-evaluation-receipt.v0",
        "status": "complete",
        "mode": args.mode,
        "checkpoint_id": checkpoint["id"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "model_root_sha256": checkpoint["model_root_sha256"],
        "export_manifest_root_sha256": checkpoint["export_manifest_root_sha256"],
        "preflight_receipt_root": checkpoint_receipt["receipt_root"],
        "contract_root": root,
        "tasks": receipt_tasks,
        "max_samples": max_samples,
        "few_shots": contract["sampling"]["few_shots"],
        "random_seed": contract["sampling"]["random_seed"],
        "scoring": contract["scoring"],
        "runtime_lock": runtime_lock,
        "runtime": {
            "python": platform.python_version(),
            "packages": package_versions(),
            "torch_cuda": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device": torch.cuda.get_device_name(0),
            "pid": os.getpid(),
        },
        "results_sha256": results_sha256,
        "started_at": started_at,
        "completed_at": completed_at,
        "production_effect": "none",
    }
    receipt_sha256 = write_json(args.output_dir / "execution_receipt.json", receipt)
    print(f"LIGHTEVAL_MODE={args.mode}")
    print(f"LIGHTEVAL_CHECKPOINT={checkpoint['id']}")
    print(f"LIGHTEVAL_CONTRACT_ROOT={root}")
    print(f"LIGHTEVAL_RESULTS_SHA256={results_sha256}")
    print(f"LIGHTEVAL_RECEIPT_SHA256={receipt_sha256}")
    print("LIGHTEVAL_STATUS=PASS")

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except ContractError as exc:
        print(f"LIGHTEVAL_STATUS=REFUSED reason={exc}", file=sys.stderr)
        raise SystemExit(2) from exc
