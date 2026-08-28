#!/usr/bin/env python3
"""Run a bounded real-corpus Branch-50 larger-context training trial."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
BASELINE_ROOT = Path("/home/mo/DEV/experiments/helix-branch49-5080-scaling-v0")
COMMON_PATH = BASELINE_ROOT / "run_512d_streaming_k32_loops3_1500m.py"
EXPECTED_COMMON_SHA256 = ""  # Filled and enforced at runtime from the adjacent receipt.
EXPECTED_PARAMETER_COUNT = 53_592_340
GPT2_SPECIAL_ID = 50_256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_common():
    receipt = ROOT / "baseline-runner.sha256"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, required=True, choices=[1024, 2048])
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--grad-accum", type=int, default=7)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlflow-uri", default="https://mlflow.thunderline.net")
    parser.add_argument("--skip-shard-sha256", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common, common_sha = load_common()
    source_identity = common.verify_source_identity()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise SystemExit("REFUSED: RTX5080 sm_120 device unavailable")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("REFUSED: BF16 unavailable")
    expected_batch = {1024: 6, 2048: 3}[args.seq_len]
    if args.batch_size != expected_batch or args.grad_accum != 7:
        raise SystemExit(
            "REFUSED: matched-token geometry requires "
            f"seq{args.seq_len}=batch{expected_batch}xaccum7"
        )
    if args.steps < 1 or args.eval_every < 1 or args.validation_batches < 1:
        raise SystemExit("REFUSED: steps/evaluation settings must be positive")

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup_optimizer_steps = max(1, cfg.warmup_steps // args.grad_accum)
    causal_targets_per_step = args.batch_size * args.grad_accum * (args.seq_len - 1)
    raw_tokens_per_step = args.batch_size * args.grad_accum * args.seq_len
    baseline_raw_tokens_per_step = 12 * 7 * 512
    if raw_tokens_per_step != baseline_raw_tokens_per_step:
        raise SystemExit(
            "REFUSED: raw tokens per optimizer step drift: "
            f"{raw_tokens_per_step} != {baseline_raw_tokens_per_step}"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = (
        f"branch50-linear-context-s{args.seq_len}-b{args.batch_size}-a{args.grad_accum}"
        f"-steps{args.steps}-{stamp}"
    )
    run_root = ROOT / "artifacts" / run_name
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
            "steps": args.steps,
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
            "ordering_algorithm": common.ORDERING_ALGORITHM,
            "matched_512_baseline_raw_tokens_per_step": True,
        },
        tags={
            "run_kind": "branch50_linear_context_real_corpus_trial",
            "production_effect": "none",
        },
    )
    if logger.start() is None:
        raise RuntimeError("MLFLOW_START_FAILED")
    (run_root / "resolved_config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, sort_keys=True, default=str) + "\n"
    )

    target_causal = args.steps * causal_targets_per_step
    sample_iter = common.iter_u16_windows(
        train_shards,
        seq_len=args.seq_len,
        seed=args.seed,
        start=common.DataOffset(),
        target_causal_targets=target_causal,
    )
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    offset = common.DataOffset()
    losses: list[tuple[float, int]] = []
    start_time = time.time()
    run_status = "FINISHED"
    try:
        for batch, batch_offset in common.iter_batches(sample_iter, batch_size=args.batch_size):
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

            if step % args.eval_every == 0 or step == args.steps:
                model.eval()
                val_sum = 0.0
                val_targets = 0
                val_iter = common.iter_u16_windows(
                    val_shards,
                    seq_len=args.seq_len,
                    seed=args.seed,
                    start=common.DataOffset(),
                    target_causal_targets=args.validation_batches * args.batch_size * (args.seq_len - 1),
                )
                with torch.no_grad():
                    for index, (val_batch, _) in enumerate(
                        common.iter_batches(val_iter, batch_size=args.batch_size)
                    ):
                        if index >= args.validation_batches:
                            break
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
            if step >= args.steps:
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
        "data_offset": asdict(offset),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "mlflow_run_id": logger.run_id,
        "mlflow_errors": logger.mlflow_errors,
        "run_root": str(run_root),
    }
    (run_root / "terminal.json").write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(terminal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
