#!/usr/bin/env python3
"""Prepare the frozen Branch-50 Lighteval lane without running evaluation.

The final checkpoint is not known at source-authoring time, so this tool turns
an already-preflighted ``save_pretrained`` export into a version-pinned,
content-addressed Lighteval command packet.  It intentionally defaults to a
dry run: smoke runs must be launched explicitly by a human/operator after the
selected checkpoint preflight has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPECTED_LIGHTEVAL_VERSION = "0.13.0"
EXPECTED_TRANSFORMERS_VERSION = "5.8.1"
DEFAULT_TASKS = "arc:easy|0"
DEFAULT_SMOKE_MAX_SAMPLES = 16
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

DATASET_IDENTITY = {
    "repository": "david-thrower/HelixLM-medium-1500.0Mt-2988750pt-20260528",
    "revision": "bd85adc4fddfd33f5ccb8ce8e58cad2c0251185b",
    "train_manifest_sha256": "b67f33931c0e545c8701166dbf990a7af64cf1c3966c5500d20bd2381bc9b115",
    "val_manifest_sha256": "2c15971275e2834e378ea358fc2acf05f7251d2199ebfa8854e5974a29f7932b",
}

TOKENIZER_IDENTITY = {
    "name": "gpt2",
    "local_cache_ref": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    "vocab_size": 50257,
    "pad_token_id": 50256,
    "eos_token_id": 50256,
    "bos_token_id": 50256,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_root(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_manifest(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries


def installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def require_version(package: str, expected: str) -> str:
    observed = installed_version(package)
    if observed != expected:
        raise SystemExit(
            f"REFUSED: {package} version must be {expected}, found {observed!r}"
        )
    return observed


def git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def resolve_source_contract(git_fn: Any = git_value) -> dict[str, str]:
    head = git_fn("rev-parse", "HEAD")
    tree = git_fn("rev-parse", "HEAD^{tree}")
    dirty = git_fn("status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise SystemExit(
            "REFUSED: Lighteval packet requires clean committed Branch-50 source"
        )
    return {
        "branch50_source_head": head,
        "branch50_source_tree": tree,
    }


def load_json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"REFUSED: invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"REFUSED: JSON root must be an object: {path}")
    return value


def validate_preflight_receipt(path: Path, export_dir: Path) -> dict[str, Any]:
    receipt = load_json_file(path)
    if receipt.get("status") != "PASS":
        raise SystemExit("REFUSED: checkpoint preflight receipt is not PASS")
    if receipt.get("lighteval_executed") is not False:
        raise SystemExit("REFUSED: preflight receipt must not claim Lighteval execution")
    export = receipt.get("export")
    if not isinstance(export, dict):
        raise SystemExit("REFUSED: preflight receipt has no export block")
    receipt_export = Path(str(export.get("path", ""))).resolve()
    if receipt_export != export_dir.resolve():
        raise SystemExit(
            "REFUSED: preflight export path mismatch: "
            f"{receipt_export} != {export_dir.resolve()}"
        )
    tokenizer = receipt.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise SystemExit("REFUSED: preflight receipt has no tokenizer block")
    for key in ("pad_token_id", "eos_token_id", "bos_token_id"):
        if tokenizer.get(key) != TOKENIZER_IDENTITY[key]:
            raise SystemExit(
                "REFUSED: tokenizer identity mismatch for "
                f"{key}: {tokenizer.get(key)!r} != {TOKENIZER_IDENTITY[key]!r}"
            )
    config = receipt.get("resolved_config")
    if not isinstance(config, dict):
        raise SystemExit("REFUSED: preflight receipt has no resolved_config block")
    if config.get("parameter_count") != 53_592_340:
        raise SystemExit(
            "REFUSED: unexpected parameter count in preflight receipt: "
            f"{config.get('parameter_count')!r}"
        )
    if config.get("seq_len") != 512:
        raise SystemExit(
            "REFUSED: selected Branch-50 eval lane expects seq_len=512, "
            f"found {config.get('seq_len')!r}"
        )
    return receipt


def render_model_yaml(export_dir: Path, *, batch_size: int, dtype: str, device: str) -> str:
    # Lighteval 0.13.0 reads YAML by taking safe_load()["model_parameters"].
    # Keep this file boring and explicit so a hostile grep can court every
    # security-sensitive value without importing Lighteval.
    export = export_dir.resolve().as_posix()
    return "\n".join(
        [
            "model_parameters:",
            f"  model_name: {json.dumps(export)}",
            f"  tokenizer: {json.dumps(export)}",
            '  revision: "main"',
            f"  batch_size: {batch_size}",
            "  max_length: 512",
            f"  dtype: {json.dumps(dtype)}",
            f"  device: {json.dumps(device)}",
            "  trust_remote_code: false",
            "  compile: false",
            "  add_special_tokens: false",
            "  skip_special_tokens: false",
            "  model_parallel: false",
            "  pairwise_tokenization: false",
            "  continuous_batching: false",
            "  override_chat_template: false",
            "  model_loading_kwargs: {}",
            "",
        ]
    )


def shell_script(argv: list[str]) -> str:
    quoted = " ".join(shlex.quote(piece) for piece in argv)
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "export TOKENIZERS_PARALLELISM=false",
            "export HF_HUB_DISABLE_TELEMETRY=1",
            quoted,
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--smoke-max-samples", type=int, default=DEFAULT_SMOKE_MAX_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--lighteval-bin", default="lighteval")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_max_samples < 1:
        raise SystemExit("REFUSED: smoke max samples must be positive")
    if not args.tasks.strip():
        raise SystemExit("REFUSED: tasks must be explicit")
    for task in args.tasks.split(","):
        if "|" not in task:
            raise SystemExit(
                "REFUSED: every task must bind an explicit few-shot count "
                f"with '|', found {task!r}"
            )
    if args.batch_size < 1:
        raise SystemExit("REFUSED: batch size must be positive")
    if not args.export_dir.is_dir():
        raise SystemExit(f"REFUSED: export directory missing: {args.export_dir}")
    if not args.preflight_receipt.is_file():
        raise SystemExit(f"REFUSED: preflight receipt missing: {args.preflight_receipt}")

    lighteval_version = require_version("lighteval", EXPECTED_LIGHTEVAL_VERSION)
    transformers_version = require_version("transformers", EXPECTED_TRANSFORMERS_VERSION)
    receipt = validate_preflight_receipt(args.preflight_receipt, args.export_dir)
    source_contract = resolve_source_contract()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_yaml_path = args.output_dir / "branch50_lighteval_model.yaml"
    result_dir = args.output_dir / "results"
    details_dir = result_dir / "details"
    manifest_path = args.output_dir / "branch50_lighteval_manifest.json"
    command_path = args.output_dir / "branch50_lighteval_command.json"
    run_script_path = args.output_dir / "run_branch50_lighteval_smoke.sh"

    model_yaml_path.write_text(
        render_model_yaml(
            args.export_dir,
            batch_size=args.batch_size,
            dtype=args.dtype,
            device=args.device,
        )
    )

    argv = [
        args.lighteval_bin,
        "accelerate",
        str(model_yaml_path.resolve()),
        args.tasks,
        "--output-dir",
        str(result_dir.resolve()),
        "--save-details",
        "--max-samples",
        str(args.smoke_max_samples),
        "--dataset-loading-processes",
        "1",
        "--num-fewshot-seeds",
        "1",
        "--no-push-to-hub",
        "--no-push-to-tensorboard",
        "--no-public-run",
        "--no-wandb",
    ]
    run_script_path.write_text(shell_script(argv))
    run_script_path.chmod(0o755)

    command = {
        "argv": argv,
        "shell_script": str(run_script_path.resolve()),
        "dry_run_default": True,
        "execute_requested": args.execute,
        "smoke_only": True,
        "comparative_benchmark": False,
    }
    command_path.write_text(json.dumps(command, indent=2, sort_keys=True) + "\n")

    export_manifest = artifact_manifest(args.export_dir)
    manifest: dict[str, Any] = {
        "schema": "helix.branch50.lighteval_frozen_lane.v0",
        "status": "PREPARED",
        "publication_effect": "none",
        "lighteval_version": lighteval_version,
        "transformers_version": transformers_version,
        "trust_remote_code": False,
        "task_contract": {
            "tasks": args.tasks,
            "smoke_max_samples": args.smoke_max_samples,
            "smoke_only": True,
            "comparative_benchmark": False,
            "full_eval_requires_new_manifest_without_max_samples": True,
        },
        "source_contract": source_contract,
        "dataset_identity": DATASET_IDENTITY,
        "tokenizer_identity": TOKENIZER_IDENTITY,
        "checkpoint_preflight": {
            "receipt_path": str(args.preflight_receipt.resolve()),
            "receipt_sha256": sha256_file(args.preflight_receipt),
            "receipt_root": receipt.get("receipt_root"),
            "checkpoint": receipt.get("checkpoint"),
            "resolved_config": receipt.get("resolved_config"),
        },
        "model_export": {
            "path": str(args.export_dir.resolve()),
            "manifest_root": canonical_root(export_manifest),
            "files": export_manifest,
        },
        "lighteval_outputs": {
            "output_dir": str(result_dir.resolve()),
            "results_dir": str(result_dir.resolve()),
            "details_dir": str(details_dir.resolve()),
            "custody_required": True,
        },
        "generated_files": {
            "model_yaml": str(model_yaml_path.resolve()),
            "command_json": str(command_path.resolve()),
            "run_script": str(run_script_path.resolve()),
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
        },
    }
    manifest["manifest_root"] = canonical_root(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if args.execute:
        completed = subprocess.run(argv, text=True, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"REFUSED: Lighteval smoke exited {completed.returncode}")
        manifest["status"] = "SMOKE_EXECUTED"
        manifest["executed_command"] = argv
        manifest["manifest_root"] = canonical_root(manifest)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
