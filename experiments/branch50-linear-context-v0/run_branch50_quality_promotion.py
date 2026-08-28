#!/usr/bin/env python3
"""Run a matched 512-vs-1024 Branch-50 quality-promotion trial.

Both variants consume the same ordered 1024-token corpus blocks and predict
the same token positions.  The 512 control sees each block as two independent
windows; the 1024 candidate masks the boundary target at index 512 so it does
not receive an extra supervised token.  Batch geometry keeps 42 base blocks
and 42,924 causal targets per optimizer update in both variants.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
SOURCE = Path(__file__).resolve().parents[2]
RUN_ROOT = Path(
    os.environ.get(
        "HELIX_BRANCH50_RUN_ROOT",
        "/home/mo/DEV/experiments/helix-branch50-linear-context-v0",
    )
)
BASELINE_ROOT = Path("/home/mo/DEV/experiments/helix-branch49-5080-scaling-v0")
COMMON_PATH = BASELINE_ROOT / "run_512d_streaming_k32_loops3_1500m.py"
MODEL_BASE_HEAD = "03d0698dd3365c81695d9ed8d4568d35d6044fbb"
MODEL_BASE_TREE = "745c042db9860bca4cdfa180543f8a60a769c936"
EXPECTED_PARAMETER_COUNT = 53_592_340
GPT2_SPECIAL_ID = 50_256
BASE_BLOCK_TOKENS = 1024
TARGETS_PER_BASE_BLOCK = 1022


@dataclass(frozen=True)
class PairOffset:
    epoch: int = 0
    shard_position: int = 0
    block_position: int = 0
    subwindow_position: int = 0
    raw_tokens_seen: int = 0
    causal_targets_seen: int = 0
    samples_seen: int = 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_common():
    receipt = ROOT / "executed" / "baseline-runner.sha256"
    if not receipt.exists():
        raise SystemExit(f"REFUSED: common-runner receipt missing: {receipt}")
    expected = receipt.read_text().strip().split()[0]
    actual = sha256(COMMON_PATH)
    if actual != expected:
        raise SystemExit(
            f"REFUSED: common runner drift: actual={actual} expected={expected}"
        )
    spec = importlib.util.spec_from_file_location("branch50_u16_common", COMMON_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit("REFUSED: cannot load common U16 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, actual


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(SOURCE), *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def verify_source_identity() -> dict[str, str]:
    head = git("rev-parse", "HEAD").stdout.strip()
    tree = git("rev-parse", "HEAD^{tree}").stdout.strip()
    dirty = git("status", "--porcelain").stdout.strip()
    model_diff = git(
        "diff",
        "--quiet",
        MODEL_BASE_HEAD,
        "--",
        "helix_lm",
        "requirements.txt",
        check=False,
    )
    if dirty:
        raise SystemExit(f"REFUSED: Branch-50 source checkout dirty:\n{dirty}")
    if model_diff.returncode != 0:
        raise SystemExit(
            "REFUSED: model-source bytes drift from admitted Branch-50 base "
            f"{MODEL_BASE_HEAD}"
        )
    return {
        "source_head": head,
        "source_tree": tree,
        "source_dirty": "false",
        "model_base_head": MODEL_BASE_HEAD,
        "model_base_tree": MODEL_BASE_TREE,
        "model_source_diff": "false",
    }


def initialize_cuda() -> None:
    device_count = int(torch._C._cuda_getDeviceCount())
    if device_count < 1:
        raise SystemExit("REFUSED: CUDA driver reports zero devices")
    try:
        torch.cuda.init()
    except RuntimeError as error:
        raise SystemExit(f"REFUSED: CUDA initialization failed: {error}") from error
    if torch.cuda.get_device_capability() != (12, 0):
        raise SystemExit(
            "REFUSED: expected RTX5080 sm_120, found "
            f"{torch.cuda.get_device_capability()}"
        )
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("REFUSED: BF16 unavailable")


def model_state_root(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(json.dumps(list(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def paired_windows(
    common: Any,
    shards: list[Any],
    *,
    seq_len: int,
    seed: int,
    target_causal_targets: int,
) -> Iterator[tuple[torch.Tensor, PairOffset]]:
    offset = PairOffset()
    epoch = 0
    while offset.causal_targets_seen < target_causal_targets:
        shard_order = common.shard_order(len(shards), seed=seed, epoch=epoch)
        for shard_position, ordered_index in enumerate(shard_order):
            shard = shards[int(ordered_index)]
            values = np.memmap(shard.path, dtype="<u2", mode="r")
            num_blocks = max(0, len(values) // BASE_BLOCK_TOKENS)
            block_order = common.window_order(
                num_blocks,
                seed=seed,
                epoch=epoch,
                shard_id=shard.shard_id,
            )
            for block_position, block_index in enumerate(block_order):
                start = int(block_index) * BASE_BLOCK_TOKENS
                block = np.asarray(
                    values[start : start + BASE_BLOCK_TOKENS],
                    dtype=np.int64,
                ).copy()
                pieces = [block] if seq_len == 1024 else [block[:512], block[512:]]
                for subwindow_position, piece in enumerate(pieces):
                    targets = TARGETS_PER_BASE_BLOCK if seq_len == 1024 else seq_len - 1
                    if offset.causal_targets_seen + targets > target_causal_targets:
                        return
                    offset = PairOffset(
                        epoch=epoch,
                        shard_position=shard_position,
                        block_position=block_position,
                        subwindow_position=subwindow_position + 1,
                        raw_tokens_seen=offset.raw_tokens_seen + seq_len,
                        causal_targets_seen=offset.causal_targets_seen + targets,
                        samples_seen=offset.samples_seen + 1,
                    )
                    yield torch.from_numpy(piece), offset
        epoch += 1


def paired_batches(
    sample_iter: Iterator[tuple[torch.Tensor, PairOffset]],
    *,
    batch_size: int,
    seq_len: int,
) -> Iterator[tuple[dict[str, torch.Tensor], PairOffset]]:
    while True:
        rows: list[torch.Tensor] = []
        offset: PairOffset | None = None
        try:
            for _ in range(batch_size):
                row, offset = next(sample_iter)
                rows.append(row)
        except StopIteration:
            return
        if len(rows) != batch_size or offset is None:
            return
        input_ids = torch.stack(rows)
        labels = input_ids.clone()
        if seq_len == 1024:
            labels[:, 512] = -100
        yield {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
        }, offset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, required=True, choices=[512, 1024])
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--grad-accum", type=int, default=7)
    parser.add_argument("--target-causal-targets", type=int, default=100_000_000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--validation-base-blocks", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlflow-uri", default="https://mlflow.thunderline.net")
    parser.add_argument("--skip-shard-sha256", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common, common_sha = load_common()
    source_identity = verify_source_identity()
    initialize_cuda()
    expected_batch = {512: 12, 1024: 6}[args.seq_len]
    if args.batch_size != expected_batch or args.grad_accum != 7:
        raise SystemExit(
            "REFUSED: matched-token geometry requires "
            f"seq{args.seq_len}=batch{expected_batch}xaccum7"
        )
    if (
        args.target_causal_targets < 1
        or args.eval_every < 1
        or args.checkpoint_every < 1
        or args.validation_base_blocks < 1
    ):
        raise SystemExit("REFUSED: target/evaluation/checkpoint settings must be positive")

    train_manifest, train_shards = common.load_and_validate_manifest(
        common.DATA, verify_hashes=not args.skip_shard_sha256
    )
    val_manifest, val_shards = common.load_and_validate_manifest(
        common.VAL_DATA, verify_hashes=not args.skip_shard_sha256
    )
    sys.path.insert(0, str(BASELINE_ROOT))
    sys.path.insert(0, str(SOURCE))
    from helix_lm.config import HelixConfig
    from helix_lm.hf_model import HelixForCausalLM
    from realtime_mlflow import RealtimeMLflowLogger

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = HelixConfig.small_v2(
        vocab_size=50_257,
        d_model=512,
        n_heads=8,
        n_loops=3,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        n_columns=3,
        nodes_per_column=(2, 3, 2),
        attention_mode="multi_scale_windowed",
        local_window=64,
        coarse_window=128,
        compressed_windows=8,
        compressed_views=8,
        consensus_type="cosine",
        corrector_type="ffn",
        use_titans_memory=False,
        use_ssm=False,
        use_cca=False,
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
        tokenizer_name="gpt2",
        pad_token_id=GPT2_SPECIAL_ID,
        eos_token_id=GPT2_SPECIAL_ID,
        bos_token_id=GPT2_SPECIAL_ID,
        tie_word_embeddings=True,
        grad_buffer_ratio=0.0,
        architectures=["HelixForCausalLM"],
        seed=args.seed,
    )
    device = torch.device("cuda")
    model = HelixForCausalLM(cfg).to(device)
    params = model.count_parameters()
    if int(params["total"]) != EXPECTED_PARAMETER_COUNT or int(params["trainable"]) != EXPECTED_PARAMETER_COUNT:
        raise SystemExit(f"REFUSED: parameter drift: {params!r}")
    initial_model_root = model_state_root(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup_optimizer_steps = max(1, cfg.warmup_steps // args.grad_accum)
    base_blocks_per_step = (
        args.batch_size * args.grad_accum * args.seq_len // BASE_BLOCK_TOKENS
    )
    if base_blocks_per_step != 42:
        raise SystemExit(
            f"REFUSED: paired geometry drift: {base_blocks_per_step} != 42 base blocks/update"
        )
    causal_targets_per_step = base_blocks_per_step * TARGETS_PER_BASE_BLOCK
    raw_tokens_per_step = args.batch_size * args.grad_accum * args.seq_len
    baseline_raw_tokens_per_step = 12 * 7 * 512
    if raw_tokens_per_step != baseline_raw_tokens_per_step:
        raise SystemExit(
            "REFUSED: raw tokens per optimizer step drift: "
            f"{raw_tokens_per_step} != {baseline_raw_tokens_per_step}"
        )
    steps = math.ceil(args.target_causal_targets / causal_targets_per_step)
    aligned_target_causal_targets = steps * causal_targets_per_step

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = (
        f"branch50-quality-s{args.seq_len}-b{args.batch_size}-a{args.grad_accum}"
        f"-t{args.target_causal_targets}-{stamp}"
    )
    run_root = RUN_ROOT / "artifacts" / "quality-promotion-v0" / run_name
    run_root.mkdir(parents=True, exist_ok=False)
    manifest_roots = {
        "train_manifest_sha256": common.manifest_root(train_manifest),
        "val_manifest_sha256": common.manifest_root(val_manifest),
    }
    harness_sha = sha256(Path(__file__))
    logger = RealtimeMLflowLogger(
        tracking_uri=args.mlflow_uri,
        experiment="helix-branch50-linear-context-v0",
        run_name=run_name,
        spool_path=run_root / "mlflow_spool.jsonl",
        params={
            **source_identity,
            **manifest_roots,
            "source_path": str(SOURCE),
            "harness_sha256": harness_sha,
            "common_runner_sha256": common_sha,
            "dataset": common.DATASET,
            "data_root": str(common.DATA),
            "validation_data_root": str(common.VAL_DATA),
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accum,
            "effective_sequences": args.batch_size * args.grad_accum,
            "raw_tokens_per_optimizer_step": raw_tokens_per_step,
            "causal_targets_per_optimizer_step": causal_targets_per_step,
            "base_blocks_per_optimizer_step": base_blocks_per_step,
            "target_causal_targets": args.target_causal_targets,
            "aligned_target_causal_targets": aligned_target_causal_targets,
            "steps": steps,
            "initial_model_root": initial_model_root,
            "parameter_count_total": EXPECTED_PARAMETER_COUNT,
            "parameter_count_trainable": EXPECTED_PARAMETER_COUNT,
            "d_model": 512,
            "n_heads": 8,
            "n_loops": 3,
            "local_window": 64,
            "coarse_window": 128,
            "compressed_windows": 8,
            "compressed_views": 8,
            "learning_rate": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "dropout": cfg.dropout,
            "attention_dropout": cfg.attn_dropout,
            "master_dtype": "float32",
            "amp_dtype": "bfloat16",
            "strict_nan_check": True,
            "grad_buffer_ratio": 0.0,
            "ordering_algorithm": "paired_1024_block_permutation_split_for_512_v1",
            "target_alignment": "mask_1024_boundary_index_512_v1",
            "validation_base_blocks": args.validation_base_blocks,
            "matched_512_baseline_raw_tokens_per_step": True,
            "matched_causal_targets_per_optimizer_step": True,
        },
        tags={
            "run_kind": "branch50_1024_quality_promotion_v0",
            "production_effect": "none",
        },
    )
    if logger.start() is None:
        raise RuntimeError("MLFLOW_START_FAILED")
    (run_root / "resolved_config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, sort_keys=True, default=str) + "\n"
    )

    sample_iter = paired_windows(
        common,
        train_shards,
        seq_len=args.seq_len,
        seed=args.seed,
        target_causal_targets=aligned_target_causal_targets,
    )
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    offset = PairOffset()
    losses: list[tuple[float, int]] = []
    start_time = time.time()
    run_status = "FINISHED"
    try:
        for batch, batch_offset in paired_batches(
            sample_iter,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        ):
            device_batch = common.to_device(batch, device=device)
            with autocast:
                output = model(**device_batch, return_dict=True)
                loss = output.loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"NONFINITE_LOSS step={step}")
            targets = common.count_causal_targets(device_batch["labels"])
            (loss / args.grad_accum).backward()
            losses.append((float(loss.detach().cpu()), targets))
            offset = batch_offset
            if len(losses) < args.grad_accum:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"NONFINITE_GRAD_NORM step={step}")
            lr = common.set_optimizer_lr(
                optimizer,
                base_lr=cfg.lr,
                optimizer_step_number=step + 1,
                warmup_optimizer_steps=warmup_optimizer_steps,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step += 1
            accum_targets = sum(targets for _, targets in losses)
            accum_loss = sum(value * targets for value, targets in losses) / accum_targets
            elapsed = max(time.time() - start_time, 1e-9)
            logger.log_metrics(
                {
                    "train/loss": losses[-1][0],
                    "train_loss": losses[-1][0],
                    "train/ppl": common.perplexity(losses[-1][0]),
                    "train_ppl": common.perplexity(losses[-1][0]),
                    "train/accum_loss": accum_loss,
                    "train/accum_ppl": common.perplexity(accum_loss),
                    "train/raw_tokens_seen": float(offset.raw_tokens_seen),
                    "train/causal_targets_seen": float(offset.causal_targets_seen),
                    "train/raw_tokens_per_second": offset.raw_tokens_seen / elapsed,
                    "train/causal_targets_per_second": offset.causal_targets_seen / elapsed,
                    "train/lr": lr,
                    "train/gradient_norm_pre_clip": float(grad_norm.detach().cpu()),
                    "system/peak_vram_bytes": float(torch.cuda.max_memory_allocated()),
                },
                step=step,
            )
            losses.clear()

            if step % args.checkpoint_every == 0:
                checkpoint = run_root / "checkpoints" / f"step-{step:08d}.pt"
                common.save_checkpoint(
                    checkpoint,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    data_offset=offset,
                    scheduler=common.scheduler_state(
                        base_lr=cfg.lr,
                        warmup_microbatches=cfg.warmup_steps,
                        grad_accum=args.grad_accum,
                    ),
                    manifest_roots=manifest_roots,
                )
                logger._append(
                    {
                        "event": "checkpoint",
                        "step": step,
                        "path": str(checkpoint),
                        "sha256": sha256(checkpoint),
                        "data_offset": asdict(offset),
                        "ts": time.time(),
                    }
                )

            if step % args.eval_every == 0 or step == steps:
                model.eval()
                val_sum = 0.0
                val_targets = 0
                val_iter = paired_windows(
                    common,
                    val_shards,
                    seq_len=args.seq_len,
                    seed=args.seed,
                    target_causal_targets=(
                        args.validation_base_blocks * TARGETS_PER_BASE_BLOCK
                    ),
                )
                with torch.no_grad():
                    for val_batch, _ in paired_batches(
                        val_iter,
                        batch_size=args.batch_size,
                        seq_len=args.seq_len,
                    ):
                        val_device = common.to_device(val_batch, device=device)
                        with autocast:
                            val_output = model(**val_device, return_dict=True)
                        if val_output.loss is None or not torch.isfinite(val_output.loss):
                            raise RuntimeError(f"NONFINITE_VAL_LOSS step={step}")
                        count = common.count_causal_targets(val_device["labels"])
                        val_sum += float(val_output.loss.detach().cpu()) * count
                        val_targets += count
                val_loss = val_sum / max(val_targets, 1)
                logger.log_metrics(
                    {
                        "val/loss": val_loss,
                        "val_loss": val_loss,
                        "val/ppl": common.perplexity(val_loss),
                        "val_ppl": common.perplexity(val_loss),
                        "val/causal_targets": float(val_targets),
                    },
                    step=step,
                    phase="validation",
                )
                model.train()
                torch.cuda.empty_cache()
            if step >= steps:
                break

        checkpoint = run_root / "checkpoint-terminal.pt"
        common.save_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            step=step,
            data_offset=offset,
            scheduler=common.scheduler_state(
                base_lr=cfg.lr,
                warmup_microbatches=cfg.warmup_steps,
                grad_accum=args.grad_accum,
            ),
            manifest_roots=manifest_roots,
        )
        logger._append(
            {
                "event": "terminal_checkpoint",
                "step": step,
                "path": str(checkpoint),
                "data_offset": asdict(offset),
                "ts": time.time(),
            }
        )
        checkpoint_sha256 = sha256(checkpoint)
        restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
        restore_ok = (
            int(restored["step"]) == step
            and restored["manifest_roots"] == manifest_roots
            and restored["scheduler"]
            == common.scheduler_state(
                base_lr=cfg.lr,
                warmup_microbatches=cfg.warmup_steps,
                grad_accum=args.grad_accum,
            )
            and restored["data_offset"] == asdict(offset)
        )
        if not restore_ok:
            raise RuntimeError("CHECKPOINT_READBACK_MISMATCH")
    except BaseException:
        run_status = "FAILED"
        raise
    finally:
        logger.finish(status=run_status)

    terminal = {
        "status": "PASS",
        "steps": step,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "raw_tokens_per_optimizer_step": raw_tokens_per_step,
        "causal_targets_per_optimizer_step": causal_targets_per_step,
        "base_blocks_per_optimizer_step": base_blocks_per_step,
        "target_causal_targets": args.target_causal_targets,
        "aligned_target_causal_targets": aligned_target_causal_targets,
        "initial_model_root": initial_model_root,
        "data_offset": asdict(offset),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "mlflow_run_id": logger.run_id,
        "mlflow_errors": logger.mlflow_errors,
        "run_root": str(run_root),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_readback": "PASS",
    }
    (run_root / "terminal.json").write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(terminal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
