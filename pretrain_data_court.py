#!/usr/bin/env python3
"""Independent equivalence and throughput courts for indexed pretraining data.

The implementation tests prove individual invariants.  This executable court
reconstructs a live continuous-token fixture independently and can replay a
complete persisted sample permutation while measuring whether storage can feed
the current Branch 60 workload with bounded headroom.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from helix_lm.dataset import ContinuousWindowDataset
from helix_lm.pretrain_data import (
    PERMUTATION_DTYPE,
    TOKEN_DTYPE,
    PretrainIndexedDataset,
    PretrainPermutation,
    PretrainSampleCompiler,
    create_pretrain_indexed_loader,
)


COURT_SCHEMA = "helix.pretrain.data-court.v0"
DEFAULT_MINIMUM_SAMPLES_PER_SECOND = 25.0


class CourtFailure(RuntimeError):
    """Raised when an observed data-path invariant disagrees with the contract."""


class IntegerTokenizer:
    """Deterministic fixture tokenizer with an observable EOS boundary."""

    eos_token_id = 99

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise CourtFailure("fixture tokenizer received implicit special tokens")
        return [int(value) for value in text.split()]


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def run_fixture_equivalence() -> dict[str, Any]:
    """Compare live and compiled semantics without sharing compiled samples."""

    texts = ["1 2", "3 4 5", "6 7 8 9 10", "11 12 13"]
    tokenizer = IntegerTokenizer()
    live_samples = list(ContinuousWindowDataset(texts, tokenizer, 4, shuffle=False))

    with tempfile.TemporaryDirectory(prefix="helix-pretrain-court-") as temporary:
        store = Path(temporary) / "samples"
        manifest = PretrainSampleCompiler(
            tokenizer,
            4,
            store,
            samples_per_shard=2,
            source={"fixture": "independent-integer-v0"},
        ).compile(texts)
        indexed = PretrainIndexedDataset(store, verify=True)
        permutation = PretrainPermutation.create(
            store / "permutations" / "epoch-0000-seed-17.u32",
            len(indexed),
            17,
            epoch=0,
        )
        loader = create_pretrain_indexed_loader(
            indexed,
            permutation,
            batch_size=2,
            num_workers=0,
            drop_last=False,
            pin_memory=False,
        )

        if len(live_samples) != len(indexed):
            raise CourtFailure("live and indexed fixture sample counts differ")
        for sample_id, live_sample in enumerate(live_samples):
            indexed_sample = indexed[sample_id]
            if not torch.equal(live_sample["input_ids"], indexed_sample["input_ids"]):
                raise CourtFailure(f"fixture token mismatch at sample {sample_id}")
            if not torch.equal(live_sample["labels"], indexed_sample["labels"]):
                raise CourtFailure(f"fixture label mismatch at sample {sample_id}")

        expected_ids = [int(value) for value in permutation.values()]
        observed_ids: list[int] = []
        observed_tokens: list[list[int]] = []
        for batch in loader:
            if not torch.equal(batch["input_ids"], batch["labels"]):
                raise CourtFailure("indexed fixture labels departed from input tokens")
            if not torch.all(batch["attention_mask"] == 1):
                raise CourtFailure("indexed fixture introduced masked or padded tokens")
            observed_ids.extend(int(value) for value in batch["sample_id"].tolist())
            observed_tokens.extend(batch["input_ids"].tolist())

        if observed_ids != expected_ids:
            raise CourtFailure("indexed fixture loader departed from persisted order")
        expected_tokens = [live_samples[sample_id]["input_ids"].tolist() for sample_id in expected_ids]
        if observed_tokens != expected_tokens:
            raise CourtFailure("indexed fixture loader changed live token windows")

        return {
            "sample_count": len(observed_ids),
            "causal_target_count": len(observed_ids) * 3,
            "manifest_sha256": manifest.manifest_sha256,
            "permutation_sha256": permutation.metadata["sha256"],
            "ordered_sample_ids_sha256": hashlib.sha256(
                np.asarray(observed_ids, dtype=PERMUTATION_DTYPE).tobytes(order="C")
            ).hexdigest(),
            "ordered_tokens_sha256": hashlib.sha256(
                np.asarray(observed_tokens, dtype=TOKEN_DTYPE).tobytes(order="C")
            ).hexdigest(),
        }


def replay_store(
    sample_store: Path,
    permutation_path: Path,
    *,
    batch_size: int,
    num_workers: int,
    maximum_samples: int | None,
    minimum_samples_per_second: float,
) -> dict[str, Any]:
    """Replay an indexed store and compare every observed ID with its file order."""

    dataset = PretrainIndexedDataset(sample_store, verify=True)
    permutation = PretrainPermutation.load(permutation_path)
    if permutation.sample_count != len(dataset):
        raise CourtFailure("permutation does not cover the complete sample store")
    if maximum_samples is not None and maximum_samples <= 0:
        raise CourtFailure("maximum_samples must be positive")

    expected_samples = len(dataset) if maximum_samples is None else min(maximum_samples, len(dataset))
    loader = create_pretrain_indexed_loader(
        dataset,
        permutation,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
    )
    permutation_values = permutation.values()
    id_digest = hashlib.sha256()
    token_digest = hashlib.sha256()
    observed_samples = 0
    observed_targets = 0
    seen = np.zeros(len(dataset), dtype=np.bool_) if expected_samples == len(dataset) else None
    started = time.perf_counter()

    for batch in loader:
        remaining = expected_samples - observed_samples
        if remaining <= 0:
            break
        take = min(remaining, len(batch["sample_id"]))
        ids = batch["sample_id"][:take].numpy().astype(PERMUTATION_DTYPE, copy=False)
        expected = np.asarray(
            permutation_values[observed_samples : observed_samples + take],
            dtype=PERMUTATION_DTYPE,
        )
        if not np.array_equal(ids, expected):
            raise CourtFailure(f"sample order mismatch at permutation position {observed_samples}")

        inputs = batch["input_ids"][:take]
        labels = batch["labels"][:take]
        masks = batch["attention_mask"][:take]
        if not torch.equal(inputs, labels):
            raise CourtFailure(f"labels departed from inputs at position {observed_samples}")
        if not torch.all(masks == 1):
            raise CourtFailure(f"padding or masked input appeared at position {observed_samples}")

        if seen is not None:
            if bool(seen[ids].any()):
                raise CourtFailure(f"duplicate sample ID observed at position {observed_samples}")
            seen[ids] = True
        id_digest.update(ids.tobytes(order="C"))
        token_digest.update(inputs.numpy().astype(TOKEN_DTYPE, copy=False).tobytes(order="C"))
        observed_samples += take
        observed_targets += take * (dataset.seq_len - 1)

    elapsed = time.perf_counter() - started
    if observed_samples != expected_samples:
        raise CourtFailure(f"observed {observed_samples} samples, expected {expected_samples}")
    if seen is not None and not bool(seen.all()):
        raise CourtFailure("complete replay omitted one or more sample IDs")

    samples_per_second = observed_samples / max(elapsed, 1e-9)
    targets_per_second = observed_targets / max(elapsed, 1e-9)
    if samples_per_second < minimum_samples_per_second:
        raise CourtFailure(
            f"indexed replay supplied {samples_per_second:.3f} samples/s, "
            f"below the declared {minimum_samples_per_second:.3f} samples/s floor"
        )
    if expected_samples == len(dataset) and id_digest.hexdigest() != permutation.metadata["sha256"]:
        raise CourtFailure("complete observed sample-ID order does not match permutation hash")

    return {
        "sample_store": str(sample_store.resolve()),
        "sample_manifest_sha256": dataset.manifest.manifest_sha256,
        "permutation": str(permutation_path.resolve()),
        "permutation_sha256": permutation.metadata["sha256"],
        "complete_replay": expected_samples == len(dataset),
        "sample_count": observed_samples,
        "causal_target_count": observed_targets,
        "ordered_sample_ids_sha256": id_digest.hexdigest(),
        "ordered_tokens_sha256": token_digest.hexdigest(),
        "elapsed_seconds": elapsed,
        "samples_per_second": samples_per_second,
        "causal_targets_per_second": targets_per_second,
        "minimum_samples_per_second": minimum_samples_per_second,
        "batch_size": batch_size,
        "num_workers": num_workers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-store", type=Path)
    parser.add_argument("--permutation", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--maximum-samples", type=int)
    parser.add_argument(
        "--minimum-samples-per-second",
        type=float,
        default=DEFAULT_MINIMUM_SAMPLES_PER_SECOND,
        help="Declared storage-only floor; the current Branch 60 GPU run consumes about 6.1 samples/s.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.sample_store is None) != (args.permutation is None):
        raise SystemExit("REFUSED: --sample-store and --permutation must be provided together")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise SystemExit("REFUSED: invalid DataLoader configuration")
    if args.minimum_samples_per_second < 0:
        raise SystemExit("REFUSED: performance floor cannot be negative")

    terminal: dict[str, Any] = {
        "schema": COURT_SCHEMA,
        "fixture_equivalence": run_fixture_equivalence(),
    }
    if args.sample_store is not None:
        terminal["store_replay"] = replay_store(
            args.sample_store,
            args.permutation,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            maximum_samples=args.maximum_samples,
            minimum_samples_per_second=args.minimum_samples_per_second,
        )
    else:
        terminal["store_replay"] = "NOT_RUN"
    terminal["terminal_root"] = _sha256_json(terminal)
    terminal["status"] = "PASS"
    payload = json.dumps(terminal, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
