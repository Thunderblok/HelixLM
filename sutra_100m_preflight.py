#!/usr/bin/env python3
"""Identity, sizing, and streaming courts for the Sutra 100M Helix baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from helix_lm.config import HelixConfig
from helix_lm.hf_model import HelixForCausalLM


DATASET = "codelion/sutra-10B"
DATASET_REVISION = "415549cff1a92b69df8b88c6108faa6097457068"
DATASET_SPLIT = "train"
TOKENIZER = "gpt2"
VOCAB_SIZE = 50_257
SEQ_LEN = 1_024
EXPECTED_PARAMETER_COUNT = 101_228_948
SCHEMA = "helix.sutra-100m-preflight.v0"


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_identity(root: Path) -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()

    return {
        "source_head": run("rev-parse", "HEAD"),
        "source_tree": run("rev-parse", "HEAD^{tree}"),
        "source_dirty": str(bool(run("status", "--porcelain"))).lower(),
    }


def build_config(*, batch_size: int = 1) -> HelixConfig:
    return HelixConfig.small_v2(
        vocab_size=VOCAB_SIZE,
        d_model=768,
        n_heads=12,
        n_loops=3,
        seq_len=SEQ_LEN,
        batch_size=batch_size,
        n_columns=3,
        nodes_per_column=(2, 3, 2),
        attention_mode="multi_scale_windowed",
        local_window=64,
        coarse_window=256,
        compressed_windows=8,
        compressed_views=8,
        use_titans_memory=False,
        use_ssm=False,
        strict_nan_check=True,
        dtype="float32",
        amp_dtype="bfloat16",
        dropout=0.05,
        attn_dropout=0.05,
        ffn_expansion=2.5,
        lr=1.5e-4,
        warmup_steps=2_000,
        weight_decay=0.05,
        grad_clip=1.0,
        tokenizer_name=TOKENIZER,
        pad_token_id=50_256,
        eos_token_id=50_256,
        bos_token_id=50_256,
        tie_word_embeddings=True,
        grad_buffer_ratio=0.0,
        architectures=["HelixForCausalLM"],
        seed=42,
    )


def model_court() -> dict[str, Any]:
    config = build_config()
    model = HelixForCausalLM(config)
    counts = model.count_parameters()
    if counts["total"] != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"parameter court failed: {counts['total']} != {EXPECTED_PARAMETER_COUNT}"
        )
    resolved = {
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "n_loops": config.n_loops,
        "n_columns": config.n_columns,
        "nodes_per_column": list(config.nodes_per_column),
        "seq_len": config.seq_len,
        "local_window": config.local_window,
        "coarse_window": config.coarse_window,
        "compressed_windows": config.compressed_windows,
        "compressed_views": config.compressed_views,
        "ffn_expansion": config.ffn_expansion,
        "parameter_count_total": counts["total"],
        "parameter_count_trainable": counts["trainable"],
    }
    return {**resolved, "resolved_config_root": sha256_bytes(canonical_json(resolved))}


def dataset_court(rows: Iterable[dict[str, Any]], *, limit: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    observed_fields: set[str] = set()
    row_count = 0
    raw_utf8_bytes = 0
    for index, row in enumerate(rows):
        if index >= limit:
            break
        if not isinstance(row, dict):
            raise RuntimeError(f"dataset row {index} is not a mapping")
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"dataset row {index} has no nonempty text")
        encoded = text.encode("utf-8")
        digest.update(index.to_bytes(8, "little"))
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        observed_fields.update(str(key) for key in row)
        row_count += 1
        raw_utf8_bytes += len(encoded)
    if row_count != limit:
        raise RuntimeError(f"dataset stream ended at {row_count}; expected {limit} rows")
    return {
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "dataset_split": DATASET_SPLIT,
        "rows_observed": row_count,
        "raw_utf8_bytes_observed": raw_utf8_bytes,
        "observed_fields": sorted(observed_fields),
        "ordered_row_text_root": digest.hexdigest(),
        "streaming": True,
        "materialized_full_dataset": False,
    }


def live_dataset_rows() -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        DATASET,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
        streaming=True,
    )
    return iter(dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-dataset", action="store_true")
    args = parser.parse_args()
    if args.rows <= 0:
        raise SystemExit("REFUSED: --rows must be positive")

    root = Path(__file__).resolve().parent
    terminal: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "production_effect": "none",
        "training_started": False,
        "state_probe_posture": "observer_only_no_model_feedback",
        "source": git_identity(root),
        "model": model_court(),
        "dataset": None,
    }
    if not args.skip_dataset:
        terminal["dataset"] = dataset_court(live_dataset_rows(), limit=args.rows)
    terminal["terminal_root"] = sha256_bytes(canonical_json(terminal))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n")
    print(json.dumps(terminal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
