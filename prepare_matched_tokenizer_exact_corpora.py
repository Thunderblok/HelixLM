#!/usr/bin/env python3
"""Materialize GPT-2 and LengthMAX streams from identical raw HelixLM rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from helix_lm.tokenizer import HelixTokenizer  # noqa: E402

SCHEMA = "helix.matched-tokenizer-exact-corpus.v1"
SUMMARY_SCHEMA = "helix.matched-tokenizer-exact-preparation.v1"
DATASET = "david-thrower/HelixLM-medium-1500.0Mt-2988750pt-20260528"
DATASET_REVISION = "bd85adc4fddfd33f5ccb8ce8e58cad2c0251185b"
SPECIAL_ID = 50_256
VOCAB_SIZE = 50_257
DEFAULT_RAW_ROOT = Path("/home/mo/DEV/Thunderline/data/david-helixlm")
DEFAULT_LENGTHMAX_ARTIFACT = Path(
    "/home/mo/DEV/experiments/helix-lengthmax-david-v1/"
    "iterative-hybrid-dev/iterative-hybrid-tokenizer.json"
)
DEFAULT_OUTPUT_ROOT = Path("/home/mo/DEV/experiments/helix-lengthmax-matched-exact-data-v1")


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_root(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_manifest(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "rows": pq.ParquetFile(path).metadata.num_rows,
        }
        for path in paths
    ]


def iter_parquet_texts(paths: Iterable[Path]) -> Iterator[str]:
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["text"], batch_size=1024):
            for value in batch.column(0).to_pylist():
                if isinstance(value, str):
                    yield value


def validate_tokenizer(tokenizer: HelixTokenizer, *, name: str) -> dict[str, Any]:
    observed = {
        "name": name,
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "unk_token_id": tokenizer.unk_token_id,
    }
    expected = {
        "vocab_size": VOCAB_SIZE,
        "pad_token_id": SPECIAL_ID,
        "eos_token_id": SPECIAL_ID,
        "bos_token_id": SPECIAL_ID,
        "unk_token_id": SPECIAL_ID,
    }
    mismatch = {
        key: (observed[key], value)
        for key, value in expected.items()
        if observed[key] != value
    }
    if mismatch:
        raise SystemExit(f"REFUSED: {name} tokenizer identity mismatch: {mismatch}")
    return observed


def token_byte_lengths(tokenizer: HelixTokenizer, ids: list[int], *, arm: str) -> list[int]:
    if arm == "lengthmax":
        return [0 if token_id == SPECIAL_ID else len(tokenizer._backend.id_to_bytes[token_id]) for token_id in ids]
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1))
    visible += list(range(ord("®"), ord("ÿ") + 1))
    encoded = list(visible)
    offset = 0
    for value in range(256):
        if value not in visible:
            visible.append(value)
            encoded.append(256 + offset)
            offset += 1
    decoder = {chr(codepoint): value for value, codepoint in zip(visible, encoded, strict=True)}
    lengths = []
    for token_id in ids:
        if token_id == SPECIAL_ID:
            lengths.append(0)
            continue
        token = tokenizer._backend.convert_ids_to_tokens(token_id)
        lengths.append(len(bytes(decoder[value] for value in token)))
    return lengths


def write_shard(root: Path, shard_id: int, values: list[int], raw_byte_lengths: list[int]) -> dict[str, Any]:
    if len(values) != len(raw_byte_lengths):
        raise SystemExit("REFUSED: token/raw-byte attribution length mismatch")
    path = root / f"shard-{shard_id:05d}.u16"
    raw_path = root / f"shard-{shard_id:05d}.rawbytes.u32"
    array = np.asarray(values, dtype="<u2")
    raw_array = np.asarray(raw_byte_lengths, dtype="<u4")
    if array.size and int(array.max()) >= 65_536:
        raise SystemExit("REFUSED: token id exceeds uint16")
    array.tofile(path)
    raw_array.tofile(raw_path)
    return {
        "id": shard_id,
        "tokens": int(array.size),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "raw_bytes_file": raw_path.name,
        "raw_bytes_values": int(raw_array.size),
        "attributed_raw_bytes": int(raw_array.sum()),
        "raw_bytes_storage_bytes": raw_path.stat().st_size,
        "raw_bytes_sha256": sha256_file(raw_path),
    }


def materialize_matched_split(
    *,
    paths: list[Path],
    split: str,
    tokenizers: dict[str, HelixTokenizer],
    tokenizer_identities: dict[str, dict[str, Any]],
    output_root: Path,
    target_raw_bytes: int,
    shard_tokens: int,
    sources: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    split_root = output_root / split
    if split_root.exists():
        raise SystemExit(f"REFUSED: split output already exists: {split_root}")
    roots = {arm: split_root / arm for arm in tokenizers}
    for root in roots.values():
        root.mkdir(parents=True)

    buffers = {arm: [] for arm in tokenizers}
    raw_byte_buffers = {arm: [] for arm in tokenizers}
    shards: dict[str, list[dict[str, Any]]] = {arm: [] for arm in tokenizers}
    stream_digest = hashlib.sha256()
    raw_bytes = 0
    rows = 0
    started = time.time()

    for text in iter_parquet_texts(paths):
        encoded = text.encode("utf-8")
        stream_digest.update(len(encoded).to_bytes(8, "little"))
        stream_digest.update(encoded)
        raw_bytes += len(encoded)
        rows += 1
        for arm, tokenizer in tokenizers.items():
            ids = tokenizer.encode(text, add_special_tokens=False)
            if any(token < 0 or token >= VOCAB_SIZE for token in ids):
                raise SystemExit(f"REFUSED: {arm} produced token outside matched vocabulary")
            buffers[arm].extend(ids)
            lengths = token_byte_lengths(tokenizer, ids, arm=arm)
            if sum(lengths) != len(encoded):
                raise SystemExit(
                    f"REFUSED: {arm} token bytes do not reconstruct row {rows}: "
                    f"{sum(lengths)} != {len(encoded)}"
                )
            raw_byte_buffers[arm].extend(lengths)
            buffers[arm].append(tokenizer.eos_token_id)
            raw_byte_buffers[arm].append(0)
            while len(buffers[arm]) >= shard_tokens:
                shard = buffers[arm][:shard_tokens]
                raw_shard = raw_byte_buffers[arm][:shard_tokens]
                del buffers[arm][:shard_tokens]
                del raw_byte_buffers[arm][:shard_tokens]
                shards[arm].append(write_shard(roots[arm], len(shards[arm]), shard, raw_shard))
        if raw_bytes >= target_raw_bytes:
            break

    if raw_bytes < target_raw_bytes:
        raise SystemExit(
            f"REFUSED: {split} exhausted after {raw_bytes} raw bytes; "
            f"target was {target_raw_bytes}"
        )
    for arm in tokenizers:
        if buffers[arm]:
            shards[arm].append(
                write_shard(roots[arm], len(shards[arm]), buffers[arm], raw_byte_buffers[arm])
            )

    stream_sha = stream_digest.hexdigest()
    manifests: dict[str, dict[str, Any]] = {}
    for arm in tokenizers:
        total_tokens = sum(int(item["tokens"]) for item in shards[arm])
        attributed_raw_bytes = sum(int(item["attributed_raw_bytes"]) for item in shards[arm])
        if attributed_raw_bytes != raw_bytes:
            raise SystemExit(
                f"REFUSED: {arm} attributed raw bytes {attributed_raw_bytes} != {raw_bytes}"
            )
        manifest = {
            "schema": SCHEMA,
            "complete": True,
            "dataset": DATASET,
            "dataset_revision": DATASET_REVISION,
            "split": split,
            "column": "text",
            "source_files": sources,
            "source_record_stream_sha256": stream_sha,
            "rows": rows,
            "raw_utf8_bytes": raw_bytes,
            "tokens": total_tokens,
            "tokens_per_raw_byte": total_tokens / raw_bytes,
            "target_raw_bytes": target_raw_bytes,
            "target_reached": True,
            "tokenizer": arm,
            "tokenizer_identity": tokenizer_identities[arm],
            "vocab_size": VOCAB_SIZE,
            "unk_token_id": SPECIAL_ID,
            "bos_token_id": SPECIAL_ID,
            "eos_token_id": SPECIAL_ID,
            "pad_token_id": SPECIAL_ID,
            "eos_policy": "append_once_per_non_null_document",
            "dtype": "uint16_le",
            "raw_byte_attribution": "exact_per_token_source_utf8_bytes_v0",
            "raw_byte_attribution_dtype": "uint32_le",
            "attributed_raw_bytes": attributed_raw_bytes,
            "shard_tokens": shard_tokens,
            "shards": shards[arm],
            "elapsed_seconds": time.time() - started,
            "production_effect": "none",
        }
        manifest["manifest_root"] = canonical_root(
            {k: v for k, v in manifest.items() if k not in {"elapsed_seconds", "manifest_root"}}
        )
        (roots[arm] / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifests[arm] = manifest
    return manifests


def assert_matched_raw_subject(manifests: dict[str, dict[str, Any]]) -> None:
    keys = ("split", "rows", "raw_utf8_bytes", "source_record_stream_sha256")
    identities = {arm: {key: manifest[key] for key in keys} for arm, manifest in manifests.items()}
    if len({canonical_root(value) for value in identities.values()}) != 1:
        raise SystemExit(f"REFUSED: tokenizer arms consumed different rows: {identities}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--lengthmax-artifact", type=Path, default=DEFAULT_LENGTHMAX_ARTIFACT)
    parser.add_argument("--train-raw-bytes", type=int, default=128_000_000)
    parser.add_argument("--val-raw-bytes", type=int, default=32_000_000)
    parser.add_argument("--shard-tokens", type=int, default=8_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise SystemExit(f"REFUSED: output root already exists: {args.output_root}")
    if min(args.train_raw_bytes, args.val_raw_bytes) < 1 or args.shard_tokens < 512:
        raise SystemExit("REFUSED: invalid materialization bounds")
    artifact = args.lengthmax_artifact.resolve()
    if not artifact.is_file():
        raise SystemExit(f"REFUSED: LengthMAX artifact missing: {artifact}")
    train_paths = sorted(args.raw_root.glob("pretrain_train-*.parquet"))
    val_paths = sorted(args.raw_root.glob("pretrain_val-*.parquet"))
    if not train_paths or not val_paths:
        raise SystemExit(f"REFUSED: raw Parquet missing under {args.raw_root}")

    tokenizers = {
        "gpt2": HelixTokenizer("gpt2", local_files_only=True),
        "lengthmax": HelixTokenizer(f"lengthmax:{artifact}"),
    }
    identities = {
        arm: validate_tokenizer(tokenizer, name=arm) for arm, tokenizer in tokenizers.items()
    }
    identities["gpt2"].update({"backend": "transformers.AutoTokenizer", "model": "gpt2"})
    identities["lengthmax"].update(
        {
            "backend": "iterative-byte-bpe-vocab-leftmost-longest-v0",
            "artifact": str(artifact),
            "artifact_sha256": sha256_file(artifact),
        }
    )

    args.output_root.mkdir(parents=True)
    train = materialize_matched_split(
        paths=train_paths,
        split="train",
        tokenizers=tokenizers,
        tokenizer_identities=identities,
        output_root=args.output_root,
        target_raw_bytes=args.train_raw_bytes,
        shard_tokens=args.shard_tokens,
        sources=source_manifest(train_paths),
    )
    validation = materialize_matched_split(
        paths=val_paths,
        split="validation",
        tokenizers=tokenizers,
        tokenizer_identities=identities,
        output_root=args.output_root,
        target_raw_bytes=args.val_raw_bytes,
        shard_tokens=args.shard_tokens,
        sources=source_manifest(val_paths),
    )
    assert_matched_raw_subject(train)
    assert_matched_raw_subject(validation)

    summary = {
        "schema": SUMMARY_SCHEMA,
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "python_version": sys.version,
        "platform": platform.platform(),
        "arms": identities,
        "train": {
            arm: {key: manifest[key] for key in (
                "manifest_root", "rows", "raw_utf8_bytes", "tokens",
                "tokens_per_raw_byte", "source_record_stream_sha256"
            )} for arm, manifest in train.items()
        },
        "validation": {
            arm: {key: manifest[key] for key in (
                "manifest_root", "rows", "raw_utf8_bytes", "tokens",
                "tokens_per_raw_byte", "source_record_stream_sha256"
            )} for arm, manifest in validation.items()
        },
        "raw_subject_matched": True,
        "production_effect": "none",
    }
    summary["summary_root"] = canonical_root(summary)
    (args.output_root / "preparation-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
