#!/usr/bin/env python3
"""Prepare and launch the matched GPT-2 versus LengthMAX model pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/home/mo/DEV/experiments/helix-lengthmax-matched-pilot-data-v0")
DEFAULT_ARTIFACTS_ROOT = Path(
    "/home/mo/DEV/experiments/helix-lengthmax-matched-pilot-artifacts-v0"
)
DEFAULT_LENGTHMAX_ARTIFACT = Path(
    "/home/mo/DEV/experiments/helix-lengthmax-david-v1/"
    "iterative-hybrid-dev/iterative-hybrid-tokenizer.json"
)
ARMS = ("gpt2", "lengthmax")
TARGET_CAUSAL_TARGETS = 4_721_640
MAX_OPTIMIZER_STEPS = 110


def canonical_root(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def source_identity() -> dict[str, str]:
    identity = {
        "head": run_git("rev-parse", "HEAD"),
        "tree": run_git("rev-parse", "HEAD^{tree}"),
        "dirty": run_git("status", "--porcelain"),
    }
    if identity["dirty"]:
        raise SystemExit(f"REFUSED: pilot source is dirty:\n{identity['dirty']}")
    return identity


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"REFUSED: required JSON missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_data(data_root: Path) -> dict[str, Any]:
    summary = load_json(data_root / "preparation-summary.json")
    if summary.get("raw_subject_matched") is not True:
        raise SystemExit("REFUSED: preparation summary lacks matched raw subject")
    manifests: dict[str, dict[str, dict[str, Any]]] = {}
    for split in ("train", "validation"):
        manifests[split] = {
            arm: load_json(data_root / split / arm / "manifest.json") for arm in ARMS
        }
        raw_keys = ("rows", "raw_utf8_bytes", "source_record_stream_sha256")
        subjects = [
            {key: manifests[split][arm][key] for key in raw_keys} for arm in ARMS
        ]
        if subjects[0] != subjects[1]:
            raise SystemExit(f"REFUSED: {split} raw subject differs between arms")
        for arm in ARMS:
            manifest = manifests[split][arm]
            if manifest.get("tokenizer") != arm or manifest.get("complete") is not True:
                raise SystemExit(f"REFUSED: invalid {split}/{arm} manifest")
    for arm in ARMS:
        if int(manifests["train"][arm]["tokens"]) < TARGET_CAUSAL_TARGETS:
            raise SystemExit(f"REFUSED: {arm} corpus is too small for pilot target")
    return {"summary": summary, "manifests": manifests}


def prepare_data(args: argparse.Namespace) -> None:
    summary = args.data_root / "preparation-summary.json"
    if summary.exists():
        return
    if args.data_root.exists():
        raise SystemExit(
            f"REFUSED: partial data root exists without summary: {args.data_root}"
        )
    command = [
        sys.executable,
        str(ROOT / "prepare_matched_tokenizer_corpora.py"),
        "--raw-root",
        str(args.raw_root),
        "--output-root",
        str(args.data_root),
        "--lengthmax-artifact",
        str(args.lengthmax_artifact),
        "--train-raw-bytes",
        str(args.train_raw_bytes),
        "--val-raw-bytes",
        str(args.val_raw_bytes),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def find_single_terminal(pair_root: Path, arm: str) -> tuple[Path, dict[str, Any]]:
    terminals = list((pair_root / arm).glob("*/terminal.json"))
    if len(terminals) != 1:
        raise SystemExit(f"REFUSED: expected one {arm} terminal, found {len(terminals)}")
    return terminals[0], load_json(terminals[0])


def run_arm(
    *, args: argparse.Namespace, identity: dict[str, str], pair_id: str, arm: str
) -> tuple[Path, dict[str, Any]]:
    pair_root = args.artifacts_root / pair_id
    log_path = pair_root / "logs" / f"{arm}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "run_matched_tokenizer_arm.py"),
        "--arm",
        arm,
        "--pilot-pair-id",
        pair_id,
        "--expected-source-head",
        identity["head"],
        "--expected-source-tree",
        identity["tree"],
        "--data-root",
        str(args.data_root),
        "--artifacts-root",
        str(args.artifacts_root),
        "--lengthmax-artifact",
        str(args.lengthmax_artifact),
        "--max-optimizer-steps",
        str(MAX_OPTIMIZER_STEPS),
        "--target-causal-targets",
        str(TARGET_CAUSAL_TARGETS),
        "--mlflow-uri",
        args.mlflow_uri,
    ]
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, text=True, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f"FAILED: {arm} exited {result.returncode}; see {log_path}")
    return find_single_terminal(pair_root, arm)


def verify_pair(
    *, pair_id: str, identity: dict[str, str], data: dict[str, Any], terminals: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    roots = {terminal["initial_model_state_root"] for terminal in terminals.values()}
    if len(roots) != 1:
        raise SystemExit(f"REFUSED: initial model state differs between arms: {roots}")
    config_roots = {terminal["resolved_config_root"] for terminal in terminals.values()}
    if len(config_roots) != 1:
        raise SystemExit(f"REFUSED: resolved model config differs between arms: {config_roots}")
    for arm, terminal in terminals.items():
        if terminal.get("status") != "PASS":
            raise SystemExit(f"REFUSED: {arm} terminal did not pass")
        if terminal.get("steps") != MAX_OPTIMIZER_STEPS:
            raise SystemExit(f"REFUSED: {arm} completed {terminal.get('steps')} steps")
        if terminal["data_offset"]["causal_targets_seen"] != TARGET_CAUSAL_TARGETS:
            raise SystemExit(f"REFUSED: {arm} causal-target budget drift")
        if not terminal.get("mlflow_run_id"):
            raise SystemExit(f"REFUSED: {arm} has no MLflow run id")

    result = {
        "schema": "helix.matched-tokenizer-pair-terminal.v0",
        "status": "PASS",
        "pilot_pair_id": pair_id,
        "source_head": identity["head"],
        "source_tree": identity["tree"],
        "initial_model_state_root": next(iter(roots)),
        "resolved_config_root": next(iter(config_roots)),
        "architecture": "branch49-d512-s512-k8-nl3-ffn2.5",
        "seed": 42,
        "optimizer": "AdamW(lr=1.5e-4,weight_decay=0.05,warmup_microbatches=2000)",
        "max_optimizer_steps_per_arm": MAX_OPTIMIZER_STEPS,
        "causal_targets_per_arm": TARGET_CAUSAL_TARGETS,
        "corpus_order": "same_raw_row_order_with_tokenizer_specific_window_boundaries",
        "raw_subject_matched": True,
        "preparation_summary_root": data["summary"]["summary_root"],
        "arms": {
            arm: {
                "mlflow_run_id": terminal["mlflow_run_id"],
                "estimated_raw_bytes_seen": terminal["estimated_raw_bytes_seen"],
                "last_validation": terminal["last_validation"],
                "run_root": terminal["run_root"],
            }
            for arm, terminal in terminals.items()
        },
        "quality_comparison": "estimated_bits_per_byte; perplexity is within-tokenizer only",
        "raw_byte_metric_posture": "estimated_from_full_materialized_split_tokens_per_raw_byte",
        "production_effect": "none",
    }
    result["pair_terminal_root"] = canonical_root(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root", type=Path, default=Path("/home/mo/DEV/Thunderline/data/david-helixlm")
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--lengthmax-artifact", type=Path, default=DEFAULT_LENGTHMAX_ARTIFACT)
    parser.add_argument("--train-raw-bytes", type=int, default=64_000_000)
    parser.add_argument("--val-raw-bytes", type=int, default=32_000_000)
    parser.add_argument(
        "--mlflow-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "https://mlflow.thunderline.net")
    )
    parser.add_argument("--arm-order", choices=("gpt2-first", "lengthmax-first"), default="gpt2-first")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity = source_identity()
    prepare_data(args)
    data = verify_data(args.data_root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pair_id = f"matched-tokenizer-b49-s110-{timestamp}"
    pair_root = args.artifacts_root / pair_id
    pair_root.mkdir(parents=True, exist_ok=False)
    preflight = {
        "schema": "helix.matched-tokenizer-preflight.v0",
        "pilot_pair_id": pair_id,
        "source": identity,
        "data_summary_root": data["summary"]["summary_root"],
        "arm_order": args.arm_order,
        "mlflow_uri": args.mlflow_uri,
        "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
        "causal_targets_per_arm": TARGET_CAUSAL_TARGETS,
        "production_effect": "none",
    }
    (pair_root / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    order = ARMS if args.arm_order == "gpt2-first" else tuple(reversed(ARMS))
    terminals: dict[str, dict[str, Any]] = {}
    for arm in order:
        _, terminals[arm] = run_arm(
            args=args, identity=identity, pair_id=pair_id, arm=arm
        )
    terminal = verify_pair(pair_id=pair_id, identity=identity, data=data, terminals=terminals)
    (pair_root / "pair-terminal.json").write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(terminal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
