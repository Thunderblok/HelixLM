#!/usr/bin/env python3
"""Run one arm of the matched GPT-2 versus LengthMAX Branch-49 pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT
DEFAULT_DATA_ROOT = Path("/home/mo/DEV/experiments/helix-lengthmax-matched-exact-data-v1")
DEFAULT_ARTIFACTS = Path("/home/mo/DEV/experiments/helix-lengthmax-matched-exact-artifacts-v1")
DEFAULT_LENGTHMAX_ARTIFACT = Path(
    "/home/mo/DEV/experiments/helix-lengthmax-david-v1/"
    "iterative-hybrid-dev/iterative-hybrid-tokenizer.json"
)

DATASET = "david-thrower/HelixLM-medium-1500.0Mt-2988750pt-20260528"
DATASET_REVISION = "bd85adc4fddfd33f5ccb8ce8e58cad2c0251185b"
GPT2_SPECIAL_ID = 50256
EXPECTED_PARAMETER_COUNT = 53_592_340

ORDERING_ALGORITHM = "sequential_u16_stream_preserving_raw_row_order_v0"
SHARD_CACHE_POSTURE = "u16_memmap_disk_backed_no_full_corpus_sample_list"
SAMPLE_DETERMINISM = "same_raw_row_order_with_tokenizer_specific_window_boundaries"

sys.path.insert(0, str(ROOT))
from realtime_mlflow import RealtimeMLflowLogger  # noqa: E402


@dataclass(frozen=True)
class DataOffset:
    epoch: int = 0
    shard_position: int = 0
    window_position: int = 0
    raw_tokens_seen: int = 0
    causal_targets_seen: int = 0
    samples_seen: int = 0
    raw_bytes_seen: int = 0
    causal_raw_bytes_seen: int = 0

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "DataOffset":
        if not value:
            return cls()
        return cls(
            epoch=int(value.get("epoch", 0)),
            shard_position=int(value.get("shard_position", 0)),
            window_position=int(value.get("window_position", 0)),
            raw_tokens_seen=int(value.get("raw_tokens_seen", 0)),
            causal_targets_seen=int(value.get("causal_targets_seen", 0)),
            samples_seen=int(value.get("samples_seen", 0)),
            raw_bytes_seen=int(value.get("raw_bytes_seen", 0)),
            causal_raw_bytes_seen=int(value.get("causal_raw_bytes_seen", 0)),
        )


@dataclass(frozen=True)
class ShardRef:
    shard_id: int
    path: Path
    tokens: int
    bytes: int
    sha256: str
    raw_bytes_path: Path
    raw_bytes_sha256: str


def run_git(args: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def verify_source_identity(*, expected_head: str, expected_tree: str) -> dict[str, str]:
    if not SOURCE.exists():
        raise SystemExit(f"REFUSED: source checkout missing: {SOURCE}")
    head = run_git(["rev-parse", "HEAD"], cwd=SOURCE)
    tree = run_git(["rev-parse", "HEAD^{tree}"], cwd=SOURCE)
    dirty = run_git(["status", "--porcelain"], cwd=SOURCE)
    if head != expected_head:
        raise SystemExit(f"REFUSED: source HEAD drift {head} != {expected_head}")
    if tree != expected_tree:
        raise SystemExit(f"REFUSED: source TREE drift {tree} != {expected_tree}")
    if dirty:
        raise SystemExit(f"REFUSED: source checkout dirty:\n{dirty}")
    return {"source_head": head, "source_tree": tree, "source_dirty": "false"}


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_manifest(
    root: Path, *, tokenizer: str, verify_hashes: bool
) -> tuple[dict[str, Any], list[ShardRef]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"REFUSED: manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "complete": True,
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "dtype": "uint16_le",
        "tokenizer": tokenizer,
        "eos_token_id": GPT2_SPECIAL_ID,
        "raw_byte_attribution": "exact_per_token_source_utf8_bytes_v0",
        "raw_byte_attribution_dtype": "uint32_le",
    }
    mismatches = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
    if mismatches:
        raise SystemExit(f"REFUSED: manifest mismatch for {root}: {mismatches!r}")

    refs: list[ShardRef] = []
    for item in manifest.get("shards", []):
        shard_id = int(item["id"])
        path = root / f"shard-{shard_id:05d}.u16"
        if not path.exists():
            raise SystemExit(f"REFUSED: shard missing: {path}")
        actual_bytes = path.stat().st_size
        expected_bytes = int(item["bytes"])
        if actual_bytes != expected_bytes:
            raise SystemExit(f"REFUSED: shard byte mismatch {path}: {actual_bytes} != {expected_bytes}")
        if verify_hashes:
            actual_sha = sha256_file(path)
            if actual_sha != item["sha256"]:
                raise SystemExit(f"REFUSED: shard hash mismatch {path}: {actual_sha} != {item['sha256']}")
        raw_path = root / str(item["raw_bytes_file"])
        if not raw_path.exists() or raw_path.stat().st_size != int(item["raw_bytes_storage_bytes"]):
            raise SystemExit(f"REFUSED: raw-byte attribution shard invalid: {raw_path}")
        if verify_hashes and sha256_file(raw_path) != item["raw_bytes_sha256"]:
            raise SystemExit(f"REFUSED: raw-byte attribution hash mismatch: {raw_path}")
        refs.append(ShardRef(shard_id, path, int(item["tokens"]), expected_bytes, str(item["sha256"]), raw_path, str(item["raw_bytes_sha256"])))
    if not refs:
        raise SystemExit(f"REFUSED: no shards declared by manifest: {manifest_path}")
    return manifest, refs


def stable_permutation(length: int, *, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(length)


def shard_order(num_shards: int, *, seed: int, epoch: int) -> np.ndarray:
    del seed, epoch
    return np.arange(num_shards)


def window_order(num_windows: int, *, seed: int, epoch: int, shard_id: int) -> np.ndarray:
    del seed, epoch, shard_id
    return np.arange(num_windows)


def iter_u16_windows(
    shards: list[ShardRef],
    *,
    seq_len: int,
    seed: int,
    start: DataOffset,
    target_causal_targets: int | None,
) -> Iterator[tuple[torch.Tensor, DataOffset, int, int]]:
    offset = start
    epoch = offset.epoch
    shard_pos = offset.shard_position
    window_pos = offset.window_position
    while True:
        order = shard_order(len(shards), seed=seed, epoch=epoch)
        while shard_pos < len(order):
            shard = shards[int(order[shard_pos])]
            values = np.memmap(shard.path, dtype="<u2", mode="r")
            raw_byte_values = np.memmap(shard.raw_bytes_path, dtype="<u4", mode="r")
            if len(raw_byte_values) != len(values):
                raise RuntimeError(f"raw-byte attribution count mismatch for shard {shard.shard_id}")
            num_windows = max(0, (len(values) - seq_len) // seq_len + 1)
            windows = window_order(num_windows, seed=seed, epoch=epoch, shard_id=shard.shard_id)
            while window_pos < len(windows):
                if target_causal_targets is not None and offset.causal_targets_seen + (seq_len - 1) > target_causal_targets:
                    return
                start_index = int(windows[window_pos]) * seq_len
                token_ids = torch.from_numpy(np.asarray(values[start_index : start_index + seq_len], dtype=np.int64).copy())
                window_raw_bytes = int(np.asarray(raw_byte_values[start_index : start_index + seq_len], dtype=np.uint64).sum())
                causal_raw_bytes = int(np.asarray(raw_byte_values[start_index + 1 : start_index + seq_len], dtype=np.uint64).sum())
                offset = DataOffset(
                    epoch=epoch,
                    shard_position=shard_pos,
                    window_position=window_pos + 1,
                    raw_tokens_seen=offset.raw_tokens_seen + seq_len,
                    causal_targets_seen=offset.causal_targets_seen + (seq_len - 1),
                    samples_seen=offset.samples_seen + 1,
                    raw_bytes_seen=offset.raw_bytes_seen + window_raw_bytes,
                    causal_raw_bytes_seen=offset.causal_raw_bytes_seen + causal_raw_bytes,
                )
                yield token_ids, offset, window_raw_bytes, causal_raw_bytes
                window_pos += 1
            shard_pos += 1
            window_pos = 0
        epoch += 1
        shard_pos = 0
        window_pos = 0


def iter_batches(
    sample_iter: Iterator[tuple[torch.Tensor, DataOffset, int, int]],
    *,
    batch_size: int,
) -> Iterator[tuple[dict[str, torch.Tensor], DataOffset, int, int]]:
    while True:
        rows: list[torch.Tensor] = []
        last_offset: DataOffset | None = None
        batch_raw_bytes = 0
        batch_causal_raw_bytes = 0
        try:
            for _ in range(batch_size):
                row, last_offset, row_raw_bytes, row_causal_raw_bytes = next(sample_iter)
                rows.append(row)
                batch_raw_bytes += row_raw_bytes
                batch_causal_raw_bytes += row_causal_raw_bytes
        except StopIteration:
            return
        if len(rows) != batch_size or last_offset is None:
            return
        input_ids = torch.stack(rows)
        yield {"input_ids": input_ids, "labels": input_ids.clone(), "attention_mask": torch.ones_like(input_ids)}, last_offset, batch_raw_bytes, batch_causal_raw_bytes


def count_causal_targets(labels: torch.Tensor) -> int:
    return int((labels[:, 1:] != -100).sum().item())


def to_device(batch: dict[str, torch.Tensor], *, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def perplexity(loss: float) -> float:
    return float(math.exp(min(loss, 20.0)))


def mlflow_param_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True, default=str)


def get_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": bool(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def set_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_state["keys"].cpu().numpy().astype(np.uint32, copy=False),
            int(numpy_state["position"]),
            bool(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def set_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    optimizer_step_number: int,
    warmup_optimizer_steps: int,
) -> float:
    lr = base_lr if warmup_optimizer_steps <= 0 else base_lr * min(1.0, max(1, optimizer_step_number) / warmup_optimizer_steps)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def scheduler_state(*, base_lr: float, warmup_microbatches: int, grad_accum: int) -> dict[str, Any]:
    return {
        "type": "linear_warmup_then_constant",
        "base_lr": base_lr,
        "warmup_microbatches": warmup_microbatches,
        "grad_accum": grad_accum,
        "warmup_optimizer_steps": max(1, warmup_microbatches // grad_accum),
        "minimum_lr_ratio_after_warmup": 1.0,
    }


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    data_offset: DataOffset,
    scheduler: dict[str, Any],
    manifest_roots: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "data_offset": asdict(data_offset),
            "rng_state": get_rng_state(),
            "scheduler": scheduler,
            "manifest_roots": manifest_roots,
        },
        path,
    )


def manifest_root(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def model_state_root(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("gpt2", "lengthmax"), required=True)
    ap.add_argument("--pilot-pair-id", required=True)
    ap.add_argument("--expected-source-head", required=True)
    ap.add_argument("--expected-source-tree", required=True)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    ap.add_argument("--lengthmax-artifact", type=Path, default=DEFAULT_LENGTHMAX_ARTIFACT)
    ap.add_argument("--max-optimizer-steps", type=int, default=400)
    ap.add_argument("--target-causal-targets", type=int, default=17_169_600)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--grad-accum", type=int, default=7)
    ap.add_argument("--compressed-windows", type=int, default=8)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--validation-batches", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--learning-rate", type=float, default=1.5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-microbatches", type=int, default=2000)
    ap.add_argument("--mlflow-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "https://mlflow.thunderline.net"))
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--skip-shard-sha256", action="store_true", help="Debug-only speed escape; launch runs should not use it.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    source_identity = verify_source_identity(
        expected_head=args.expected_source_head, expected_tree=args.expected_source_tree
    )
    tokenizer_name = (
        "gpt2"
        if args.arm == "gpt2"
        else f"lengthmax:{args.lengthmax_artifact.resolve()}"
    )
    data = args.data_root / "train" / args.arm
    val_data = args.data_root / "validation" / args.arm
    if args.compressed_windows != 8:
        raise SystemExit("REFUSED: this bound runner is fixed to compressed_windows=8")
    if args.batch_size != 12 or args.grad_accum != 7:
        raise SystemExit("REFUSED: this bound runner is fixed to batch_size=12 and grad_accum=7")
    if args.learning_rate != 1.5e-4 or args.weight_decay != 0.05:
        raise SystemExit("REFUSED: this bound runner is fixed to lr=1.5e-4 and weight_decay=0.05")
    causal_targets_per_optimizer_step = args.batch_size * args.grad_accum * (512 - 1)
    aligned_target_causal_targets = (
        math.ceil(args.target_causal_targets / causal_targets_per_optimizer_step)
        * causal_targets_per_optimizer_step
    )
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise SystemExit("REFUSED: RTX5080 CUDA device is not available")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("REFUSED: BF16 unsupported")

    train_manifest, train_shards = load_and_validate_manifest(
        data, tokenizer=args.arm, verify_hashes=not args.skip_shard_sha256
    )
    val_manifest, val_shards = load_and_validate_manifest(
        val_data, tokenizer=args.arm, verify_hashes=not args.skip_shard_sha256
    )
    if int(train_manifest.get("tokens", 0)) < args.target_causal_targets:
        raise SystemExit("REFUSED: train manifest contains fewer tokens than requested")

    sys.path.insert(0, str(SOURCE))
    from helix_lm.config import HelixConfig  # noqa: WPS433
    from helix_lm.hf_model import HelixForCausalLM  # noqa: WPS433
    from helix_lm.tokenizer import HelixTokenizer  # noqa: WPS433

    tokenizer = HelixTokenizer(
        tokenizer_name, **({"local_files_only": True} if args.arm == "gpt2" else {})
    )
    tokenizer_court = {
        "tokenizer_len": len(tokenizer),
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "tokenizer_bos_token_id": tokenizer.bos_token_id,
    }
    if tokenizer_court != {
        "tokenizer_len": 50_257,
        "tokenizer_pad_token_id": GPT2_SPECIAL_ID,
        "tokenizer_eos_token_id": GPT2_SPECIAL_ID,
        "tokenizer_bos_token_id": GPT2_SPECIAL_ID,
    }:
        raise SystemExit(f"REFUSED: tokenizer court mismatch: {tokenizer_court!r}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = HelixConfig.small_v2(
        vocab_size=50_257,
        d_model=512,
        n_heads=8,
        n_loops=3,
        seq_len=512,
        batch_size=args.batch_size,
        n_columns=3,
        nodes_per_column=(2, 3, 2),
        attention_mode="multi_scale_windowed",
        local_window=64,
        coarse_window=128,
        compressed_windows=args.compressed_windows,
        compressed_views=8,
        use_titans_memory=False,
        use_ssm=False,
        strict_nan_check=True,
        dtype="float32",
        amp_dtype="bfloat16",
        dropout=0.05,
        attn_dropout=0.05,
        ffn_expansion=2.5,
        lr=args.learning_rate,
        warmup_steps=args.warmup_microbatches,
        weight_decay=args.weight_decay,
        grad_clip=1.0,
        tokenizer_name=tokenizer_name,
        pad_token_id=GPT2_SPECIAL_ID,
        eos_token_id=GPT2_SPECIAL_ID,
        bos_token_id=GPT2_SPECIAL_ID,
        tie_word_embeddings=True,
        grad_buffer_ratio=0.0,
        architectures=["HelixForCausalLM"],
    )
    resolved = {
        "vocab_size": cfg.vocab_size,
        "pad_token_id": cfg.pad_token_id,
        "eos_token_id": cfg.eos_token_id,
        "bos_token_id": cfg.bos_token_id,
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "grad_buffer_ratio": cfg.grad_buffer_ratio,
        "d_model": cfg.d_model,
        "seq_len": cfg.seq_len,
        "n_loops": cfg.n_loops,
        "batch_size": cfg.batch_size,
        "compressed_windows": cfg.compressed_windows,
        "compressed_views": cfg.compressed_views,
        "dropout": cfg.dropout,
        "attn_dropout": cfg.attn_dropout,
        "lr": cfg.lr,
        "warmup_steps": cfg.warmup_steps,
        "weight_decay": cfg.weight_decay,
        "dtype": str(cfg.dtype),
        "amp_dtype": cfg.amp_dtype,
        "architectures": getattr(cfg, "architectures", None),
    }
    expected = {
        "vocab_size": 50_257,
        "pad_token_id": GPT2_SPECIAL_ID,
        "eos_token_id": GPT2_SPECIAL_ID,
        "bos_token_id": GPT2_SPECIAL_ID,
        "tie_word_embeddings": True,
        "grad_buffer_ratio": 0.0,
        "d_model": 512,
        "seq_len": 512,
        "n_loops": 3,
        "batch_size": 12,
        "compressed_windows": 8,
        "compressed_views": 8,
        "dropout": 0.05,
        "attn_dropout": 0.05,
        "lr": 1.5e-4,
        "warmup_steps": 2000,
        "weight_decay": 0.05,
        "dtype": "torch.float32",
        "amp_dtype": "bfloat16",
        "architectures": ["HelixForCausalLM"],
    }
    if resolved != expected:
        raise SystemExit(f"REFUSED: resolved model config mismatch: {resolved!r} != {expected!r}")
    resolved_config_root = manifest_root(resolved)

    device = torch.device("cuda")
    model = HelixForCausalLM(cfg).to(device)
    params = model.count_parameters()
    if int(params["total"]) != EXPECTED_PARAMETER_COUNT or int(params["trainable"]) != EXPECTED_PARAMETER_COUNT:
        raise SystemExit(f"REFUSED: parameter count mismatch: {params!r}")
    initial_model_state_root = model_state_root(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    schedule = scheduler_state(base_lr=args.learning_rate, warmup_microbatches=args.warmup_microbatches, grad_accum=args.grad_accum)
    warmup_optimizer_steps = int(schedule["warmup_optimizer_steps"])
    step = 0
    data_offset = DataOffset()
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        data_offset = DataOffset.from_mapping(state.get("data_offset"))
        set_rng_state(state.get("rng_state"))
        if state.get("scheduler") != schedule:
            raise SystemExit(f"REFUSED: scheduler drift in resume checkpoint: {state.get('scheduler')!r} != {schedule!r}")
        set_optimizer_lr(optimizer, base_lr=args.learning_rate, optimizer_step_number=step + 1, warmup_optimizer_steps=warmup_optimizer_steps)

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"matched-b49-{args.arm}-d512-s512-k8-nl3-b12-a7-s400-{run_stamp}"
    run_root = args.artifacts_root / args.pilot_pair_id / args.arm / run_name
    run_root.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(run_root / "tokenizer")
    manifest_roots = {"train_manifest_sha256": manifest_root(train_manifest), "val_manifest_sha256": manifest_root(val_manifest)}

    config_params = {f"cfg_{key}": mlflow_param_value(value) for key, value in cfg.to_dict().items()}
    logger = RealtimeMLflowLogger(
        tracking_uri=args.mlflow_uri,
        experiment="helix-lengthmax-matched-model-promotion-v1",
        run_name=run_name,
        spool_path=run_root / "mlflow_spool.jsonl",
        params={
            **config_params,
            **source_identity,
            **manifest_roots,
            **tokenizer_court,
            "dataset": DATASET,
            "dataset_revision": DATASET_REVISION,
            "pilot_pair_id": args.pilot_pair_id,
            "pilot_arm": args.arm,
            "tokenizer": tokenizer_name,
            "tokenizer_artifact_sha256": (
                sha256_file(args.lengthmax_artifact.resolve())
                if args.arm == "lengthmax"
                else "not_applicable"
            ),
            "initial_model_state_root": initial_model_state_root,
            "resolved_config_root": resolved_config_root,
            "source_path": str(SOURCE),
            "runner": Path(__file__).name,
            "runner_mode": "matched_tokenizer_branch49_exact_bytes_v1",
            "ordering_algorithm": ORDERING_ALGORITHM,
            "shard_cache_posture": SHARD_CACHE_POSTURE,
            "sample_sequence_determinism": SAMPLE_DETERMINISM,
            "data_root": str(data),
            "validation_data_root": str(val_data),
            "train_raw_utf8_bytes": train_manifest["raw_utf8_bytes"],
            "train_tokens_per_raw_byte": train_manifest["tokens_per_raw_byte"],
            "validation_raw_utf8_bytes": val_manifest["raw_utf8_bytes"],
            "validation_tokens_per_raw_byte": val_manifest["tokens_per_raw_byte"],
            "target_causal_targets": args.target_causal_targets,
            "aligned_target_causal_targets": aligned_target_causal_targets,
            "train_manifest_tokens": train_manifest["tokens"],
            "train_manifest_shards": len(train_shards),
            "val_manifest_tokens": val_manifest["tokens"],
            "val_manifest_shards": len(val_shards),
            "parameter_count_total": EXPECTED_PARAMETER_COUNT,
            "parameter_count_trainable": EXPECTED_PARAMETER_COUNT,
            "effective_sequences": args.batch_size * args.grad_accum,
            "effective_causal_targets_per_optimizer_step": causal_targets_per_optimizer_step,
            "learning_rate_scheduler": schedule["type"],
            "scheduler_warmup_microbatches": args.warmup_microbatches,
            "scheduler_warmup_optimizer_steps": warmup_optimizer_steps,
            "scheduler_min_lr_ratio": 1.0,
            "master_dtype": "float32",
            "amp_dtype": "bfloat16",
            "strict_nan_check": True,
            "grad_buffer_ratio": 0.0,
            "max_optimizer_steps": args.max_optimizer_steps,
        },
        tags={
            "run_kind": "matched_tokenizer_branch49_promotion_v1",
            "pilot_pair_id": args.pilot_pair_id,
            "pilot_arm": args.arm,
            "mlflow_logging": "realtime_step_and_target_weighted_validation",
            "production_effect": "none",
        },
    )
    if logger.start() is None:
        raise RuntimeError("MLFLOW_START_FAILED: refusing training without a live run")
    (run_root / "resolved_config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True))
    (run_root / "scheduler.json").write_text(json.dumps(schedule, indent=2, sort_keys=True))
    (run_root / "manifest_roots.json").write_text(json.dumps(manifest_roots, indent=2, sort_keys=True))
    logger.log_metrics(
        {
            "system/parameter_count": float(EXPECTED_PARAMETER_COUNT),
            "system/peak_vram_bytes": 0.0,
            "train/causal_targets_seen": float(data_offset.causal_targets_seen),
            "train/raw_tokens_seen": float(data_offset.raw_tokens_seen),
            "train/raw_bytes_seen": float(data_offset.raw_bytes_seen),
            "train/causal_raw_bytes_seen": float(data_offset.causal_raw_bytes_seen),
        },
        step=step,
        phase="preflight",
    )

    sample_iter = iter_u16_windows(
        train_shards,
        seq_len=cfg.seq_len,
        seed=args.seed,
        start=data_offset,
        target_causal_targets=aligned_target_causal_targets,
    )
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    run_status = "FINISHED"
    micro_losses: list[tuple[float, int]] = []
    last_validation: dict[str, float] | None = None
    set_optimizer_lr(optimizer, base_lr=args.learning_rate, optimizer_step_number=step + 1, warmup_optimizer_steps=warmup_optimizer_steps)
    try:
        for batch, batch_offset, _batch_raw_bytes, _batch_causal_raw_bytes in iter_batches(sample_iter, batch_size=args.batch_size):
            if args.max_optimizer_steps and step >= args.max_optimizer_steps:
                break
            device_batch = to_device(batch, device=device)
            with autocast:
                out = model(**device_batch, return_dict=True)
                loss = out.loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"NONFINITE_LOSS optimizer_step={step}")
            causal_targets = count_causal_targets(device_batch["labels"])
            (loss / args.grad_accum).backward()
            immediate_loss = float(loss.detach().cpu())
            micro_losses.append((immediate_loss, causal_targets))
            data_offset = batch_offset
            if len(micro_losses) < args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            current_lr = set_optimizer_lr(optimizer, base_lr=args.learning_rate, optimizer_step_number=step + 1, warmup_optimizer_steps=warmup_optimizer_steps)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step += 1

            accum_targets = sum(targets for _, targets in micro_losses)
            accum_loss = sum(loss_value * targets for loss_value, targets in micro_losses) / max(accum_targets, 1)
            elapsed = max(time.time() - start_time, 1e-6)
            logger.log_metrics(
                {
                    "train/loss": immediate_loss,
                    "train_ppl": perplexity(immediate_loss),
                    "train/accum_loss": accum_loss,
                    "train/accum_ppl": perplexity(accum_loss),
                    "train/causal_targets_seen": float(data_offset.causal_targets_seen),
                    "train/raw_tokens_seen": float(data_offset.raw_tokens_seen),
                    "train/causal_targets_per_second": data_offset.causal_targets_seen / elapsed,
                    "train/raw_tokens_per_second": data_offset.raw_tokens_seen / elapsed,
                    "train/raw_bytes_seen": float(data_offset.raw_bytes_seen),
                    "train/causal_raw_bytes_seen": float(data_offset.causal_raw_bytes_seen),
                    "train/raw_bytes_per_second": data_offset.raw_bytes_seen / elapsed,
                    "train/lr": current_lr,
                    "system/peak_vram_bytes": float(torch.cuda.max_memory_allocated()),
                },
                step=step,
            )
            micro_losses.clear()

            if step % args.eval_every == 0:
                model.eval()
                val_total = 0.0
                val_targets = 0
                val_raw_bytes = 0
                val_batch = None
                val_device_batch = None
                val_out = None
                val_loss_tensor = None
                val_iter = iter_u16_windows(
                    val_shards,
                    seq_len=cfg.seq_len,
                    seed=args.seed,
                    start=DataOffset(),
                    target_causal_targets=args.validation_batches * args.batch_size * (cfg.seq_len - 1),
                )
                with torch.no_grad():
                    for index, (val_batch, _, val_batch_raw_bytes, val_batch_causal_raw_bytes) in enumerate(iter_batches(val_iter, batch_size=args.batch_size)):
                        if index >= args.validation_batches:
                            break
                        val_device_batch = to_device(val_batch, device=device)
                        with autocast:
                            val_out = model(**val_device_batch, return_dict=True)
                            val_loss_tensor = val_out.loss
                        if val_loss_tensor is None or not torch.isfinite(val_loss_tensor):
                            raise RuntimeError(f"NONFINITE_VAL_LOSS optimizer_step={step}")
                        targets = count_causal_targets(val_device_batch["labels"])
                        val_total += float(val_loss_tensor.detach().cpu()) * targets
                        val_targets += targets
                        val_raw_bytes += val_batch_causal_raw_bytes
                val_loss = val_total / max(val_targets, 1)
                val_exact_bits_per_byte = (
                    val_loss * val_targets / math.log(2) / max(val_raw_bytes, 1)
                )
                last_validation = {
                    "loss": val_loss,
                    "ppl": perplexity(val_loss),
                    "causal_targets": float(val_targets),
                    "raw_bytes": float(val_raw_bytes),
                    "exact_bits_per_byte": val_exact_bits_per_byte,
                }
                logger.log_metrics(
                    {
                        "val/loss": val_loss,
                        "val_loss": val_loss,
                        "val/ppl": perplexity(val_loss),
                        "val_ppl": perplexity(val_loss),
                        "val/causal_targets": float(val_targets),
                        "val/raw_bytes": float(val_raw_bytes),
                        "val/exact_bits_per_byte": val_exact_bits_per_byte,
                    },
                    step=step,
                    phase="validation",
                )
                del val_batch, val_device_batch, val_out, val_loss_tensor
                torch.cuda.empty_cache()
                model.train()

            if step % args.checkpoint_every == 0:
                ckpt = run_root / "checkpoints" / f"step-{step:08d}.pt"
                save_checkpoint(ckpt, model=model, optimizer=optimizer, step=step, data_offset=data_offset, scheduler=schedule, manifest_roots=manifest_roots)
                logger._append({"event": "checkpoint", "step": step, "path": str(ckpt), "data_offset": asdict(data_offset), "ts": time.time()})
            set_optimizer_lr(optimizer, base_lr=args.learning_rate, optimizer_step_number=step + 1, warmup_optimizer_steps=warmup_optimizer_steps)

        if step > 0:
            terminal_checkpoint = run_root / "checkpoints" / "terminal.pt"
            save_checkpoint(
                terminal_checkpoint,
                model=model,
                optimizer=optimizer,
                step=step,
                data_offset=data_offset,
                scheduler=schedule,
                manifest_roots=manifest_roots,
            )
            logger._append(
                {
                    "event": "terminal_checkpoint",
                    "step": step,
                    "path": str(terminal_checkpoint),
                    "data_offset": asdict(data_offset),
                    "ts": time.time(),
                }
            )
    except BaseException:
        run_status = "FAILED"
        raise
    finally:
        logger.finish(status=run_status)

    terminal = {
        "status": "PASS",
        "pilot_pair_id": args.pilot_pair_id,
        "pilot_arm": args.arm,
        "steps": step,
        "data_offset": asdict(data_offset),
        "params": params,
        "initial_model_state_root": initial_model_state_root,
        "resolved_config_root": resolved_config_root,
        "train_manifest_root": manifest_roots["train_manifest_sha256"],
        "validation_manifest_root": manifest_roots["val_manifest_sha256"],
        "raw_bytes_seen": data_offset.raw_bytes_seen,
        "causal_raw_bytes_seen": data_offset.causal_raw_bytes_seen,
        "raw_byte_metric_posture": "exact_per_selected_token_window",
        "last_validation": last_validation,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "run_root": str(run_root),
        "mlflow_run_id": logger.run_id,
        "mlflow_errors": logger.mlflow_errors,
        "scheduler": schedule,
    }
    (run_root / "terminal.json").write_text(json.dumps(terminal, indent=2, sort_keys=True, default=str))
    print(json.dumps(terminal, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
