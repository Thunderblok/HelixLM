#!/usr/bin/env python3
"""Compile a Hugging Face text split into deterministic Helix pretraining samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from helix_lm import HelixTokenizer, PretrainSampleCompiler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-shard", type=int, default=65_536)
    parser.add_argument("--max-source-rows", type=int)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = HelixTokenizer(args.tokenizer)
    dataset = load_dataset(
        args.dataset,
        revision=args.revision,
        split=args.split,
        streaming=True,
    )

    def texts():
        for row in dataset:
            value = row.get(args.text_column)
            if value is not None:
                yield value

    compiler = PretrainSampleCompiler(
        tokenizer,
        args.seq_len,
        args.output_dir,
        samples_per_shard=args.samples_per_shard,
        source={
            "dataset": args.dataset,
            "revision": args.revision,
            "split": args.split,
            "text_column": args.text_column,
            "tokenizer": args.tokenizer,
        },
    )
    manifest = compiler.compile(
        texts(),
        max_source_rows=args.max_source_rows,
        max_samples=args.max_samples,
    )
    print(json.dumps({
        "manifest": str(manifest.root / "manifest.json"),
        "manifest_sha256": manifest.manifest_sha256,
        "sample_count": manifest.sample_count,
        "causal_target_count": manifest.value["causal_target_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
