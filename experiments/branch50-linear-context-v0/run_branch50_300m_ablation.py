#!/usr/bin/env python3
"""Run one resumable, single-knob Branch-50 300M-token ablation."""

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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
SEQ_LEN = 512
CAUSAL_TARGETS_PER_SAMPLE = SEQ_LEN - 1


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
    wake = subprocess.run(
        ["nvidia-smi", "-L"],
        check=False,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if wake.returncode != 0:
        raise SystemExit("REFUSED: nvidia-smi could not wake the CUDA driver")
    device_count = int(torch._C._cuda_getDeviceCount())
    if device_count < 1:
        raise SystemExit("REFUSED: CUDA driver reports zero devices after nvidia-smi wake-up")
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


def state_dict_root(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in state_dict.items():
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(json.dumps(list(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def model_state_root(model: torch.nn.Module) -> str:
    return state_dict_root(model.state_dict())


def canonical_root(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def save_ablation_checkpoint(
    common: Any,
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    data_offset: Any,
    scheduler: dict[str, Any],
    manifest_roots: dict[str, str],
    ablation_contract: dict[str, Any],
    best_val_loss: float,
    best_val_step: int,
    last_val_loss: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    model_root = model_state_root(model)
    optimizer_state = optimizer.state_dict()
    torch.save(
        {
            "model": model.state_dict(),
            "model_root": model_root,
            "optimizer": optimizer_state,
            "optimizer_state_entries": len(optimizer_state["state"]),
            "step": int(step),
            "data_offset": asdict(data_offset),
            "rng_state": common.get_rng_state(),
            "scheduler": scheduler,
            "manifest_roots": manifest_roots,
            "ablation_contract": ablation_contract,
            "ablation_contract_root": canonical_root(ablation_contract),
            "best_val_loss": best_val_loss,
            "best_val_step": best_val_step,
            "last_val_loss": last_val_loss,
        },
        temporary,
    )
    os.replace(temporary, path)


def save_best_model(
    path: Path,
    *,
    model: torch.nn.Module,
    step: int,
    val_loss: float,
    manifest_roots: dict[str, str],
    ablation_contract: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    model_root = model_state_root(model)
    torch.save(
        {
            "model": model.state_dict(),
            "model_root": model_root,
            "step": step,
            "val_loss": val_loss,
            "manifest_roots": manifest_roots,
            "ablation_contract": ablation_contract,
        },
        temporary,
    )
    os.replace(temporary, path)


def save_rotating_checkpoint(common: Any, checkpoint_dir: Path, **kwargs: Any) -> Path:
    latest = checkpoint_dir / "latest.pt"
    previous = checkpoint_dir / "previous.pt"
    staged = checkpoint_dir / "staged.pt"
    save_ablation_checkpoint(common, staged, **kwargs)
    if latest.exists():
        os.replace(latest, previous)
    os.replace(staged, latest)
    return latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-id", required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--grad-accum", type=int, default=7)
    parser.add_argument("--target-causal-targets", type=int, default=300_000_000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--warmup-microbatches", type=int, default=2_000)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--attention-dropout", type=float, default=0.05)
    parser.add_argument("--ffn-expansion", type=float, default=2.5)
    parser.add_argument("--mlflow-uri", default="https://mlflow.thunderline.net")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--skip-shard-sha256", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed_id_chars = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    )
    if not args.ablation_id or any(char not in allowed_id_chars for char in args.ablation_id):
        raise SystemExit("REFUSED: ablation_id contains unsafe characters")
    common, common_sha = load_common()
    source_identity = verify_source_identity()
    initialize_cuda()
    if args.batch_size != 12 or args.grad_accum != 7:
        raise SystemExit("REFUSED: 300M ablations require batch12xaccum7")
    if args.skip_shard_sha256 and not args.max_optimizer_steps:
        raise SystemExit("REFUSED: full ablations require shard SHA-256 verification")
    if (
        args.target_causal_targets < 1
        or args.eval_every < 1
        or args.checkpoint_every < 1
        or args.validation_batches < 1
        or args.learning_rate <= 0
        or args.warmup_microbatches < 0
        or args.weight_decay < 0
        or args.grad_clip <= 0
        or not 0 <= args.dropout < 1
        or not 0 <= args.attention_dropout < 1
        or args.ffn_expansion <= 0
    ):
        raise SystemExit("REFUSED: invalid ablation settings")
    baseline_knobs = {
        "learning_rate": 1.5e-4,
        "warmup_microbatches": 2_000,
        "weight_decay": 0.05,
        "grad_clip": 1.0,
        "dropout": 0.05,
        "attention_dropout": 0.05,
        "ffn_expansion": 2.5,
    }
    resolved_knobs = {
        "learning_rate": args.learning_rate,
        "warmup_microbatches": args.warmup_microbatches,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "dropout": args.dropout,
        "attention_dropout": args.attention_dropout,
        "ffn_expansion": args.ffn_expansion,
    }
    changed_knobs = [
        key for key, baseline in baseline_knobs.items() if resolved_knobs[key] != baseline
    ]
    if args.ablation_id == "control":
        if changed_knobs:
            raise SystemExit(f"REFUSED: control changes knobs: {changed_knobs}")
    elif len(changed_knobs) != 1:
        raise SystemExit(
            "REFUSED: an ablation must change exactly one knob; "
            f"changed={changed_knobs}"
        )

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
        seq_len=SEQ_LEN,
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
        dropout=args.dropout,
        attn_dropout=args.attention_dropout,
        ffn_expansion=args.ffn_expansion,
        lr=args.learning_rate,
        warmup_steps=args.warmup_microbatches,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
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
    if args.ffn_expansion == 2.5 and (
        int(params["total"]) != EXPECTED_PARAMETER_COUNT
        or int(params["trainable"]) != EXPECTED_PARAMETER_COUNT
    ):
        raise SystemExit(f"REFUSED: baseline parameter drift: {params!r}")
    initial_model_root = model_state_root(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup_optimizer_steps = max(1, cfg.warmup_steps // args.grad_accum)
    causal_targets_per_step = args.batch_size * args.grad_accum * CAUSAL_TARGETS_PER_SAMPLE
    raw_tokens_per_step = args.batch_size * args.grad_accum * SEQ_LEN
    steps = math.ceil(args.target_causal_targets / causal_targets_per_step)
    aligned_target_causal_targets = steps * causal_targets_per_step
    manifest_roots = {
        "train_manifest_sha256": common.manifest_root(train_manifest),
        "val_manifest_sha256": common.manifest_root(val_manifest),
    }
    ablation_contract = {
        "schema": "helix.branch50.300m-ablation.v0",
        "ablation_id": args.ablation_id,
        "changed_knobs": changed_knobs,
        "knobs": resolved_knobs,
        "seq_len": SEQ_LEN,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "target_causal_targets": args.target_causal_targets,
        "eval_every": args.eval_every,
        "checkpoint_every": args.checkpoint_every,
        "validation_batches": args.validation_batches,
        "seed": args.seed,
        "manifest_roots": manifest_roots,
        "source_identity": source_identity,
    }

    schedule = common.scheduler_state(
        base_lr=cfg.lr,
        warmup_microbatches=cfg.warmup_steps,
        grad_accum=args.grad_accum,
    )
    step = 0
    offset = common.DataOffset()
    best_val_loss = math.inf
    best_val_step = 0
    last_val_loss: float | None = None
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        if state.get("manifest_roots") != manifest_roots:
            raise SystemExit("REFUSED: resume manifest roots do not match ablation")
        if state.get("scheduler") != schedule:
            raise SystemExit("REFUSED: resume scheduler does not match ablation")
        if state.get("ablation_contract") != ablation_contract:
            raise SystemExit("REFUSED: resume ablation contract does not match")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        offset = common.DataOffset.from_mapping(state.get("data_offset"))
        best_val_loss = float(state.get("best_val_loss", math.inf))
        best_val_step = int(state.get("best_val_step", 0))
        last_val_loss = state.get("last_val_loss")
        common.set_rng_state(state.get("rng_state"))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = (
        f"branch50-ablation-{args.ablation_id}-s512-b{args.batch_size}"
        f"-a{args.grad_accum}-t{args.target_causal_targets}-{stamp}"
    )
    run_root = RUN_ROOT / "artifacts" / "ablation-300m-v0" / run_name
    run_root.mkdir(parents=True, exist_ok=False)
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
            "ablation_id": args.ablation_id,
            "changed_knobs": json.dumps(changed_knobs),
            "ablation_contract_root": canonical_root(ablation_contract),
            "seq_len": SEQ_LEN,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accum,
            "effective_sequences": args.batch_size * args.grad_accum,
            "raw_tokens_per_optimizer_step": raw_tokens_per_step,
            "causal_targets_per_optimizer_step": causal_targets_per_step,
            "target_causal_targets": args.target_causal_targets,
            "aligned_target_causal_targets": aligned_target_causal_targets,
            "steps": steps,
            "resume_checkpoint": str(args.resume) if args.resume else "none",
            "resume_step": step,
            "initial_model_root": initial_model_root,
            "parameter_count_total": int(params["total"]),
            "parameter_count_trainable": int(params["trainable"]),
            "d_model": 512,
            "n_heads": 8,
            "n_loops": 3,
            "local_window": 64,
            "coarse_window": 128,
            "compressed_windows": 8,
            "compressed_views": 8,
            "ffn_expansion": cfg.ffn_expansion,
            "learning_rate": cfg.lr,
            "warmup_microbatches": cfg.warmup_steps,
            "warmup_optimizer_steps": warmup_optimizer_steps,
            "scheduler_policy": schedule["type"],
            "scheduler_min_lr_ratio": schedule["minimum_lr_ratio_after_warmup"],
            "weight_decay": cfg.weight_decay,
            "grad_clip": cfg.grad_clip,
            "dropout": cfg.dropout,
            "attention_dropout": cfg.attn_dropout,
            "master_dtype": "float32",
            "amp_dtype": "bfloat16",
            "strict_nan_check": True,
            "grad_buffer_ratio": 0.0,
            "ordering_algorithm": common.ORDERING_ALGORITHM,
            "validation_batches": args.validation_batches,
            "single_knob_contract": True,
            "shard_sha256_verified": not args.skip_shard_sha256,
        },
        tags={
            "run_kind": "branch50_300m_single_knob_ablation_v0",
            "production_effect": "none",
        },
    )
    if logger.start() is None:
        raise RuntimeError("MLFLOW_START_FAILED")
    (run_root / "resolved_config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, sort_keys=True, default=str) + "\n"
    )
    (run_root / "ablation_contract.json").write_text(
        json.dumps(ablation_contract, indent=2, sort_keys=True) + "\n"
    )

    sample_iter = common.iter_u16_windows(
        train_shards,
        seq_len=SEQ_LEN,
        seed=args.seed,
        start=offset,
        target_causal_targets=aligned_target_causal_targets,
    )
    autocast = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[tuple[float, int]] = []
    start_time = time.time()
    session_start_raw_tokens = offset.raw_tokens_seen
    session_start_causal_targets = offset.causal_targets_seen
    run_status = "FINISHED"
    common.set_optimizer_lr(
        optimizer,
        base_lr=cfg.lr,
        optimizer_step_number=step + 1,
        warmup_optimizer_steps=warmup_optimizer_steps,
    )
    try:
        for batch, batch_offset in common.iter_batches(sample_iter, batch_size=args.batch_size):
            if args.max_optimizer_steps and step >= args.max_optimizer_steps:
                break
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
                    "train/raw_tokens_per_second": (
                        offset.raw_tokens_seen - session_start_raw_tokens
                    )
                    / elapsed,
                    "train/causal_targets_per_second": (
                        offset.causal_targets_seen - session_start_causal_targets
                    )
                    / elapsed,
                    "train/lr": lr,
                    "train/gradient_norm_pre_clip": float(grad_norm.detach().cpu()),
                    "system/peak_vram_bytes": float(torch.cuda.max_memory_allocated()),
                    "train/skipped_batches": 0.0,
                    "train/nonfinite_events": 0.0,
                },
                step=step,
            )
            losses.clear()

            if step % args.checkpoint_every == 0:
                checkpoint = save_rotating_checkpoint(
                    common,
                    run_root / "checkpoints",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    data_offset=offset,
                    scheduler=schedule,
                    manifest_roots=manifest_roots,
                    ablation_contract=ablation_contract,
                    best_val_loss=best_val_loss,
                    best_val_step=best_val_step,
                    last_val_loss=last_val_loss,
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
                val_iter = common.iter_u16_windows(
                    val_shards,
                    seq_len=SEQ_LEN,
                    seed=args.seed,
                    start=common.DataOffset(),
                    target_causal_targets=args.validation_batches
                    * args.batch_size
                    * CAUSAL_TARGETS_PER_SAMPLE,
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
                last_val_loss = val_loss
                improved = val_loss < best_val_loss
                if improved:
                    best_val_loss = val_loss
                    best_val_step = step
                    save_best_model(
                        run_root / "checkpoints" / "best-model.pt",
                        model=model,
                        step=step,
                        val_loss=val_loss,
                        manifest_roots=manifest_roots,
                        ablation_contract=ablation_contract,
                    )
                logger.log_metrics(
                    {
                        "val/loss": val_loss,
                        "val_loss": val_loss,
                        "val/ppl": common.perplexity(val_loss),
                        "val_ppl": common.perplexity(val_loss),
                        "val/causal_targets": float(val_targets),
                        "val/best_loss": best_val_loss,
                        "val/best_ppl": common.perplexity(best_val_loss),
                    },
                    step=step,
                    phase="validation",
                )
                model.train()
                torch.cuda.empty_cache()
            if step >= steps or (args.max_optimizer_steps and step >= args.max_optimizer_steps):
                break

        checkpoint = run_root / "checkpoints" / "terminal.pt"
        save_ablation_checkpoint(
            common,
            checkpoint,
            model=model,
            optimizer=optimizer,
            step=step,
            data_offset=offset,
            scheduler=schedule,
            manifest_roots=manifest_roots,
            ablation_contract=ablation_contract,
            best_val_loss=best_val_loss,
            best_val_step=best_val_step,
            last_val_loss=last_val_loss,
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
        restored_model_root = state_dict_root(restored["model"])
        best_checkpoint = run_root / "checkpoints" / "best-model.pt"
        best_checkpoint_ok = best_val_step == 0
        if best_val_step:
            best_state = torch.load(best_checkpoint, map_location="cpu", weights_only=True)
            best_checkpoint_ok = (
                int(best_state["step"]) == best_val_step
                and float(best_state["val_loss"]) == best_val_loss
                and state_dict_root(best_state["model"]) == best_state["model_root"]
                and best_state["manifest_roots"] == manifest_roots
                and best_state["ablation_contract"] == ablation_contract
            )
        restore_ok = (
            int(restored["step"]) == step
            and restored_model_root == restored["model_root"]
            and int(restored["optimizer_state_entries"]) > 0
            and len(restored["optimizer"]["state"]) == restored["optimizer_state_entries"]
            and set(restored["rng_state"]) == {"python", "numpy", "torch", "cuda"}
            and restored["manifest_roots"] == manifest_roots
            and restored["scheduler"] == schedule
            and restored["ablation_contract"] == ablation_contract
            and restored["data_offset"] == asdict(offset)
            and float(restored["best_val_loss"]) == best_val_loss
            and int(restored["best_val_step"]) == best_val_step
            and restored["last_val_loss"] == last_val_loss
            and best_checkpoint_ok
        )
        if not restore_ok:
            raise RuntimeError("CHECKPOINT_READBACK_MISMATCH")
    except BaseException as error:
        run_status = "FAILED"
        (run_root / "failure.json").write_text(
            json.dumps(
                {
                    "status": "FAILED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "step": step,
                    "data_offset": asdict(offset),
                    "ablation_contract_root": canonical_root(ablation_contract),
                    "mlflow_run_id": logger.run_id,
                    "mlflow_errors": logger.mlflow_errors,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        raise
    finally:
        logger.finish(status=run_status)

    terminal_status = "SMOKE_PASS" if args.max_optimizer_steps else "PASS"
    if logger.mlflow_errors:
        terminal_status = "HOLD_MLFLOW_ERRORS"
    terminal = {
        "status": terminal_status,
        "steps": step,
        "ablation_id": args.ablation_id,
        "seq_len": SEQ_LEN,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "raw_tokens_per_optimizer_step": raw_tokens_per_step,
        "causal_targets_per_optimizer_step": causal_targets_per_step,
        "target_causal_targets": args.target_causal_targets,
        "aligned_target_causal_targets": aligned_target_causal_targets,
        "initial_model_root": initial_model_root,
        "final_model_root": restored_model_root,
        "parameter_count": params,
        "learning_rate": cfg.lr,
        "warmup_microbatches": cfg.warmup_steps,
        "weight_decay": cfg.weight_decay,
        "grad_clip": cfg.grad_clip,
        "dropout": cfg.dropout,
        "attention_dropout": cfg.attn_dropout,
        "ffn_expansion": cfg.ffn_expansion,
        "last_val_loss": last_val_loss,
        "last_val_ppl": common.perplexity(last_val_loss)
        if last_val_loss is not None
        else None,
        "best_val_loss": best_val_loss if math.isfinite(best_val_loss) else None,
        "best_val_ppl": common.perplexity(best_val_loss)
        if math.isfinite(best_val_loss)
        else None,
        "best_val_step": best_val_step,
        "skipped_batches": 0,
        "nonfinite_events": 0,
        "data_offset": asdict(offset),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "mlflow_run_id": logger.run_id,
        "mlflow_errors": logger.mlflow_errors,
        "run_root": str(run_root),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_readback": "PASS",
        "checkpoint_model_root": restored_model_root,
        "checkpoint_optimizer_state_entries": restored["optimizer_state_entries"],
        "best_checkpoint": str(best_checkpoint) if best_val_step else None,
        "best_checkpoint_readback": "PASS" if best_checkpoint_ok else "FAIL",
        "mlflow_spool": str(run_root / "mlflow_spool.jsonl"),
    }
    (run_root / "terminal.json").write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(terminal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
