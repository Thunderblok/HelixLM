#!/usr/bin/env python3
"""Run one resumable Branch52 activation-checkpointing ablation."""

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
from typing import Any, Iterator

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
SOURCE = Path(__file__).resolve().parents[2]
RUN_ROOT = Path(
    os.environ.get(
        "HELIX_BRANCH52_RUN_ROOT",
        "/home/mo/DEV/experiments/helix-branch52-activation-checkpointing-v0",
    )
)
BASELINE_ROOT = Path("/home/mo/DEV/experiments/helix-branch49-5080-scaling-v0")
COMMON_PATH = BASELINE_ROOT / "run_512d_streaming_k32_loops3_1500m.py"
COMMON_RECEIPT = (
    SOURCE
    / "experiments"
    / "branch50-linear-context-v0"
    / "executed"
    / "baseline-runner.sha256"
)
MODEL_BASE_HEAD = "03d0698dd3365c81695d9ed8d4568d35d6044fbb"
MODEL_BASE_TREE = "745c042db9860bca4cdfa180543f8a60a769c936"
BRANCH51_BASE_HEAD = "d297a3c633f04751bc9e0a0f7af28e2751c47853"
EXPECTED_PARAMETER_COUNT = 53_592_340
GPT2_SPECIAL_ID = 50_256
SEQ_LEN = 512
CAUSAL_TARGETS_PER_SAMPLE = SEQ_LEN - 1
BASELINE_BATCH_SIZE = 12
BASELINE_GRAD_ACCUM = 7
BASELINE_N_LOOPS = 3
GEOMETRY_WARMUP_MICROBATCHES = {
    (12, 7): 2_000,
    (10, 6): 1_710,
    (8, 8): 2_280,
    (7, 13): 3_705,
    (12, 9): 2_565,
}
ALLOWED_OPTIMIZER_GEOMETRIES = frozenset(GEOMETRY_WARMUP_MICROBATCHES)


def build_scheduler_state(
    *,
    policy: str,
    base_lr: float,
    warmup_microbatches: int,
    grad_accum: int,
    total_optimizer_steps: int,
    min_lr_ratio: float,
) -> dict[str, Any]:
    warmup_optimizer_steps = max(1, warmup_microbatches // grad_accum)
    if policy == "linear_warmup_then_constant" and min_lr_ratio != 1.0:
        raise SystemExit(
            "REFUSED: scheduler_min_lr_ratio is only active with cosine_decay"
        )
    if policy == "cosine_decay" and total_optimizer_steps <= warmup_optimizer_steps:
        raise SystemExit(
            "REFUSED: cosine_decay requires total optimizer steps beyond warmup"
        )
    return {
        "type": policy,
        "base_lr": base_lr,
        "warmup_microbatches": warmup_microbatches,
        "grad_accum": grad_accum,
        "warmup_optimizer_steps": warmup_optimizer_steps,
        "total_optimizer_steps": total_optimizer_steps,
        "minimum_lr_ratio_after_warmup": min_lr_ratio,
    }


def optimizer_lr_for_step(
    *,
    base_lr: float,
    optimizer_step_number: int,
    warmup_optimizer_steps: int,
    total_optimizer_steps: int,
    min_lr_ratio: float,
    policy: str,
) -> float:
    if warmup_optimizer_steps <= 0:
        return base_lr
    bounded_step = max(1, optimizer_step_number)
    if bounded_step <= warmup_optimizer_steps:
        return base_lr * min(1.0, bounded_step / warmup_optimizer_steps)
    if policy == "linear_warmup_then_constant":
        return base_lr
    if policy == "cosine_decay":
        decay_steps = max(1, total_optimizer_steps - warmup_optimizer_steps)
        progress = min(1.0, (bounded_step - warmup_optimizer_steps) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)
    raise SystemExit(f"REFUSED: unsupported scheduler policy: {policy}")


def set_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    optimizer_step_number: int,
    warmup_optimizer_steps: int,
    total_optimizer_steps: int,
    min_lr_ratio: float,
    policy: str,
) -> float:
    lr = optimizer_lr_for_step(
        base_lr=base_lr,
        optimizer_step_number=optimizer_step_number,
        warmup_optimizer_steps=warmup_optimizer_steps,
        total_optimizer_steps=total_optimizer_steps,
        min_lr_ratio=min_lr_ratio,
        policy=policy,
    )
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def corpus_pass_plan(
    train_manifest: dict[str, Any],
    *,
    seq_len: int,
    batch_size: int,
    grad_accum: int,
) -> dict[str, int]:
    raw_tokens = int(train_manifest.get("tokens", 0))
    if raw_tokens < seq_len or raw_tokens % seq_len != 0:
        raise SystemExit(
            "REFUSED: full-corpus pass requires manifest tokens to be an exact "
            f"multiple of seq_len={seq_len}; tokens={raw_tokens}"
        )
    total_samples = raw_tokens // seq_len
    causal_targets = total_samples * (seq_len - 1)
    samples_per_full_step = batch_size * grad_accum
    full_optimizer_steps = total_samples // samples_per_full_step
    remaining_samples = total_samples % samples_per_full_step
    optimizer_steps = full_optimizer_steps + (1 if remaining_samples else 0)
    return {
        "raw_tokens": raw_tokens,
        "total_samples": total_samples,
        "causal_targets": causal_targets,
        "samples_per_full_optimizer_step": samples_per_full_step,
        "full_optimizer_steps": full_optimizer_steps,
        "remaining_samples": remaining_samples,
        "optimizer_steps": optimizer_steps,
    }


def diminishing_return_decision(
    validation_history: list[dict[str, Any]],
    *,
    window_evals: int,
    min_improvement: float,
    patience_windows: int,
    min_optimizer_steps: int,
) -> dict[str, Any]:
    if window_evals <= 0 or patience_windows <= 0:
        return {"enabled": False, "should_stop": False, "bad_windows": 0}
    if len(validation_history) < window_evals + 1:
        return {"enabled": True, "should_stop": False, "bad_windows": 0}
    current_step = int(validation_history[-1]["step"])
    if current_step < min_optimizer_steps:
        return {"enabled": True, "should_stop": False, "bad_windows": 0}

    bad_windows = 0
    last_improvement: float | None = None
    for end in range(len(validation_history), window_evals, -1):
        start = end - window_evals
        previous = validation_history[:start]
        current = validation_history[start:end]
        if not previous:
            break
        previous_best = min(float(item["val_loss"]) for item in previous)
        current_best = min(float(item["val_loss"]) for item in current)
        improvement = previous_best - current_best
        last_improvement = improvement
        if improvement < min_improvement:
            bad_windows += 1
            continue
        break

    return {
        "enabled": True,
        "should_stop": bad_windows >= patience_windows,
        "bad_windows": bad_windows,
        "last_window_improvement": last_improvement,
        "window_evals": window_evals,
        "min_improvement": min_improvement,
        "patience_windows": patience_windows,
        "min_optimizer_steps": min_optimizer_steps,
    }


def terminal_status_record(
    *,
    max_optimizer_steps: int,
    stop_state: dict[str, Any],
    mlflow_errors: list[Any],
) -> dict[str, Any]:
    status = "SMOKE_PASS" if max_optimizer_steps else "PASS"
    promotion_eligible = not bool(max_optimizer_steps)
    if stop_state.get("stop_reason") == "diminishing_return":
        status = "STOPPED_DIMINISHING_RETURN"
        promotion_eligible = False
    if mlflow_errors:
        status = "HOLD_MLFLOW_ERRORS"
        promotion_eligible = False
    return {
        "status": status,
        "promotion_eligible": promotion_eligible,
        "numerical_health": "PASS",
        "checkpoint_health": "PASS",
        "mlflow_health": "PASS" if not mlflow_errors else "HOLD",
    }


def checkpoint_payload(
    common: Any,
    *,
    model_state: dict[str, torch.Tensor],
    model_root: str,
    optimizer_state: dict[str, Any],
    step: int,
    data_offset: Any,
    scheduler: dict[str, Any],
    manifest_roots: dict[str, str],
    ablation_contract: dict[str, Any],
    best_val_loss: float,
    best_val_step: int,
    last_val_loss: float | None,
    validation_history: list[dict[str, Any]],
    stop_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model_state,
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
        "validation_history": validation_history,
        "stop_state": stop_state,
    }


def iter_batches_with_policy(
    sample_iter: Iterator[tuple[torch.Tensor, Any]],
    *,
    batch_size: int,
    allow_partial_batch: bool,
) -> Iterator[tuple[dict[str, torch.Tensor], Any]]:
    while True:
        rows: list[torch.Tensor] = []
        last_offset: Any | None = None
        try:
            for _ in range(batch_size):
                row, last_offset = next(sample_iter)
                rows.append(row)
        except StopIteration:
            if not allow_partial_batch or not rows or last_offset is None:
                return
        if not rows or last_offset is None:
            return
        if len(rows) != batch_size and not allow_partial_batch:
            return
        input_ids = torch.stack(rows)
        yield {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
        }, last_offset
        if len(rows) != batch_size:
            return


def iter_accumulation_groups(
    sample_iter: Iterator[tuple[torch.Tensor, Any]],
    *,
    batch_size: int,
    grad_accum: int,
    allow_partial_batch: bool,
    allow_partial_accumulation: bool,
) -> Iterator[list[tuple[dict[str, torch.Tensor], Any]]]:
    batch_iter = iter_batches_with_policy(
        sample_iter,
        batch_size=batch_size,
        allow_partial_batch=allow_partial_batch,
    )
    while True:
        group: list[tuple[dict[str, torch.Tensor], Any]] = []
        try:
            for _ in range(grad_accum):
                group.append(next(batch_iter))
        except StopIteration:
            if allow_partial_accumulation and group:
                yield group
            return
        yield group


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_common():
    receipt = COMMON_RECEIPT
    if not receipt.exists():
        raise SystemExit(f"REFUSED: common-runner receipt missing: {receipt}")
    expected = receipt.read_text().strip().split()[0]
    actual = sha256(COMMON_PATH)
    if actual != expected:
        raise SystemExit(
            f"REFUSED: common runner drift: actual={actual} expected={expected}"
        )
    spec = importlib.util.spec_from_file_location("branch52_u16_common", COMMON_PATH)
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
    if dirty:
        raise SystemExit(f"REFUSED: Branch52 source checkout dirty:\n{dirty}")
    changed_paths = {
        path
        for path in git(
            "diff", "--name-only", f"{BRANCH51_BASE_HEAD}...HEAD"
        ).stdout.splitlines()
        if path
    }
    allowed_paths = {
        "helix_lm/hf_model.py",
        "experiments/branch51-quality-vram-v0/run_branch51_quality_vram_ablation.py",
        "experiments/branch52-activation-checkpointing-v0/README.md",
        "experiments/branch52-activation-checkpointing-v0/BRANCH52_EXPERIMENT_LEDGER.md",
        "experiments/branch52-activation-checkpointing-v0/test_branch52_activation_checkpointing.py",
    }
    unexpected_paths = changed_paths - allowed_paths
    required_paths = {
        "helix_lm/hf_model.py",
        "experiments/branch51-quality-vram-v0/run_branch51_quality_vram_ablation.py",
    }
    if unexpected_paths or not required_paths.issubset(changed_paths):
        raise SystemExit(
            "REFUSED: Branch52 source boundary mismatch: "
            f"changed={sorted(changed_paths)!r} unexpected={sorted(unexpected_paths)!r}"
        )
    return {
        "source_head": head,
        "source_tree": tree,
        "source_dirty": "false",
        "branch51_base_head": BRANCH51_BASE_HEAD,
        "model_base_head": MODEL_BASE_HEAD,
        "model_base_tree": MODEL_BASE_TREE,
        "model_source_diff": "activation_checkpointing_only",
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


def changed_knobs_from(
    baseline_knobs: dict[str, Any],
    resolved_knobs: dict[str, Any],
) -> list[str]:
    changed_knobs: list[str] = []
    geometry_changed = (
        resolved_knobs["batch_size"] != baseline_knobs["batch_size"]
        or resolved_knobs["grad_accum"] != baseline_knobs["grad_accum"]
    )
    if geometry_changed:
        changed_knobs.append("optimizer_geometry")
    geometry = (resolved_knobs["batch_size"], resolved_knobs["grad_accum"])
    geometry_warmup = GEOMETRY_WARMUP_MICROBATCHES.get(geometry)
    changed_knobs.extend(
        key
        for key, baseline in baseline_knobs.items()
        if key not in {"batch_size", "grad_accum"}
        and not key.startswith("scheduler_")
        and not (
            key == "warmup_microbatches"
            and geometry_changed
            and resolved_knobs[key] == geometry_warmup
        )
        and resolved_knobs[key] != baseline
    )
    if (
        resolved_knobs["scheduler_policy"] != baseline_knobs["scheduler_policy"]
        or resolved_knobs["scheduler_min_lr_ratio"]
        != baseline_knobs["scheduler_min_lr_ratio"]
    ):
        changed_knobs.append("scheduler")
    return changed_knobs


def validate_promotion_manifest(
    manifest: dict[str, Any],
    *,
    resolved_knobs: dict[str, Any],
    changed_knobs: list[str],
) -> dict[str, Any]:
    if manifest.get("schema") != "helix.branch51.promotion-decision.v0":
        raise SystemExit("REFUSED: unsupported promotion manifest schema")
    if manifest.get("status") != "PROMOTED":
        raise SystemExit("REFUSED: promotion manifest is not PROMOTED")
    if manifest.get("selected_knobs") != resolved_knobs:
        raise SystemExit("REFUSED: promotion manifest selected knobs do not match run")
    if manifest.get("changed_knobs") != changed_knobs:
        raise SystemExit("REFUSED: promotion manifest changed knobs do not match run")
    evidence_run_ids = manifest.get("evidence_run_ids")
    if (
        not isinstance(evidence_run_ids, list)
        or not evidence_run_ids
        or any(not isinstance(run_id, str) or not run_id for run_id in evidence_run_ids)
        or len(set(evidence_run_ids)) != len(evidence_run_ids)
    ):
        raise SystemExit("REFUSED: promotion manifest evidence run IDs are invalid")
    decision = manifest.get("decision")
    if not isinstance(decision, str) or not decision.strip():
        raise SystemExit("REFUSED: promotion manifest decision is missing")
    return manifest


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
    validation_history: list[dict[str, Any]],
    stop_state: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    model_root = model_state_root(model)
    optimizer_state = optimizer.state_dict()
    torch.save(
        checkpoint_payload(
            common,
            model_state=model.state_dict(),
            model_root=model_root,
            optimizer_state=optimizer_state,
            step=step,
            data_offset=data_offset,
            scheduler=scheduler,
            manifest_roots=manifest_roots,
            ablation_contract=ablation_contract,
            best_val_loss=best_val_loss,
            best_val_step=best_val_step,
            last_val_loss=last_val_loss,
            validation_history=validation_history,
            stop_state=stop_state,
        ),
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
    parser.add_argument(
        "--scheduler-policy",
        choices=("linear_warmup_then_constant", "cosine_decay"),
        default="linear_warmup_then_constant",
    )
    parser.add_argument("--scheduler-min-lr-ratio", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--attention-dropout", type=float, default=0.05)
    parser.add_argument("--ffn-expansion", type=float, default=2.5)
    parser.add_argument("--n-loops", type=int, default=BASELINE_N_LOOPS)
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--mlflow-uri", default="https://mlflow.thunderline.net")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--skip-shard-sha256", action="store_true")
    parser.add_argument("--full-corpus-pass", action="store_true")
    parser.add_argument("--promotion-manifest", type=Path)
    parser.add_argument("--expected-full-corpus-raw-tokens", type=int, default=1_504_000_000)
    parser.add_argument("--diminishing-window-evals", type=int, default=0)
    parser.add_argument("--diminishing-min-improvement", type=float, default=0.0)
    parser.add_argument("--diminishing-patience-windows", type=int, default=0)
    parser.add_argument("--diminishing-min-optimizer-steps", type=int, default=0)
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
    if (args.batch_size, args.grad_accum) not in ALLOWED_OPTIMIZER_GEOMETRIES:
        allowed = ", ".join(
            f"batch{batch}xaccum{accum}"
            for batch, accum in sorted(ALLOWED_OPTIMIZER_GEOMETRIES)
        )
        raise SystemExit(
            "REFUSED: Branch52 supported optimizer geometries are "
            f"{allowed}"
        )
    expected_warmup_microbatches = GEOMETRY_WARMUP_MICROBATCHES[
        (args.batch_size, args.grad_accum)
    ]
    if args.warmup_microbatches != expected_warmup_microbatches:
        raise SystemExit(
            "REFUSED: optimizer geometry must preserve the Branch50 "
            "285 optimizer-step warmup; "
            f"batch{args.batch_size}xaccum{args.grad_accum} requires "
            f"warmup_microbatches={expected_warmup_microbatches}"
        )
    if args.skip_shard_sha256 and not args.max_optimizer_steps:
        raise SystemExit("REFUSED: full ablations require shard SHA-256 verification")
    if (
        args.target_causal_targets < 1
        or args.eval_every < 1
        or args.checkpoint_every < 1
        or args.validation_batches < 1
        or args.learning_rate <= 0
        or args.warmup_microbatches < 0
        or not 0 < args.scheduler_min_lr_ratio <= 1
        or args.weight_decay < 0
        or args.grad_clip <= 0
        or not 0 <= args.dropout < 1
        or not 0 <= args.attention_dropout < 1
        or args.ffn_expansion <= 0
        or args.n_loops < 1
        or args.diminishing_window_evals < 0
        or args.diminishing_min_improvement < 0
        or args.diminishing_patience_windows < 0
        or args.diminishing_min_optimizer_steps < 0
    ):
        raise SystemExit("REFUSED: invalid ablation settings")
    diminishing_enabled = (
        args.diminishing_window_evals > 0
        or args.diminishing_min_improvement > 0
        or args.diminishing_patience_windows > 0
        or args.diminishing_min_optimizer_steps > 0
    )
    if diminishing_enabled and (
        args.diminishing_window_evals < 1
        or args.diminishing_patience_windows < 1
        or args.diminishing_min_optimizer_steps < 1
    ):
        raise SystemExit(
            "REFUSED: diminishing-return stop requires positive window, patience, "
            "and minimum optimizer steps"
        )
    if args.full_corpus_pass and args.max_optimizer_steps:
        raise SystemExit("REFUSED: full-corpus pass cannot be combined with max-optimizer-steps")
    baseline_knobs = {
        "learning_rate": 1.5e-4,
        "warmup_microbatches": 2_000,
        "scheduler_policy": "linear_warmup_then_constant",
        "scheduler_min_lr_ratio": 1.0,
        "checkpoint_every": 500,
        "weight_decay": 0.05,
        "grad_clip": 1.0,
        "dropout": 0.05,
        "attention_dropout": 0.05,
        "ffn_expansion": 2.5,
        "batch_size": BASELINE_BATCH_SIZE,
        "grad_accum": BASELINE_GRAD_ACCUM,
        "n_loops": BASELINE_N_LOOPS,
        "activation_checkpointing": False,
    }
    resolved_knobs = {
        "learning_rate": args.learning_rate,
        "warmup_microbatches": args.warmup_microbatches,
        "scheduler_policy": args.scheduler_policy,
        "scheduler_min_lr_ratio": args.scheduler_min_lr_ratio,
        "checkpoint_every": args.checkpoint_every,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "dropout": args.dropout,
        "attention_dropout": args.attention_dropout,
        "ffn_expansion": args.ffn_expansion,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "n_loops": args.n_loops,
        "activation_checkpointing": args.activation_checkpointing,
    }
    changed_knobs = changed_knobs_from(baseline_knobs, resolved_knobs)
    promotion_manifest: dict[str, Any] | None = None
    promotion_manifest_root: str | None = None
    if args.promotion_manifest:
        if args.ablation_id == "control":
            raise SystemExit("REFUSED: control cannot use a promotion manifest")
        if not args.promotion_manifest.is_file():
            raise SystemExit(
                f"REFUSED: promotion manifest missing: {args.promotion_manifest}"
            )
        promotion_manifest = validate_promotion_manifest(
            json.loads(args.promotion_manifest.read_text()),
            resolved_knobs=resolved_knobs,
            changed_knobs=changed_knobs,
        )
        promotion_manifest_root = canonical_root(promotion_manifest)
    elif args.ablation_id == "control":
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
        n_loops=args.n_loops,
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
    if args.activation_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if bool(model.gradient_checkpointing) != args.activation_checkpointing:
        raise SystemExit("REFUSED: activation checkpointing instantiation mismatch")
    params = model.count_parameters()
    if args.ffn_expansion == 2.5 and args.n_loops == BASELINE_N_LOOPS and (
        int(params["total"]) != EXPECTED_PARAMETER_COUNT
        or int(params["trainable"]) != EXPECTED_PARAMETER_COUNT
    ):
        raise SystemExit(f"REFUSED: baseline parameter drift: {params!r}")
    initial_model_root = model_state_root(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup_optimizer_steps = max(1, cfg.warmup_steps // args.grad_accum)
    causal_targets_per_step = args.batch_size * args.grad_accum * CAUSAL_TARGETS_PER_SAMPLE
    raw_tokens_per_step = args.batch_size * args.grad_accum * SEQ_LEN
    full_corpus_plan = None
    if args.full_corpus_pass:
        full_corpus_plan = corpus_pass_plan(
            train_manifest,
            seq_len=SEQ_LEN,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
        )
        if full_corpus_plan["raw_tokens"] != args.expected_full_corpus_raw_tokens:
            raise SystemExit(
                "REFUSED: full-corpus manifest token count mismatch: "
                f"{full_corpus_plan['raw_tokens']} != {args.expected_full_corpus_raw_tokens}"
            )
        steps = full_corpus_plan["optimizer_steps"]
        target_causal_targets = full_corpus_plan["causal_targets"]
        aligned_target_causal_targets = full_corpus_plan["causal_targets"]
        target_raw_tokens = full_corpus_plan["raw_tokens"]
    else:
        steps = math.ceil(args.target_causal_targets / causal_targets_per_step)
        target_causal_targets = args.target_causal_targets
        aligned_target_causal_targets = steps * causal_targets_per_step
        target_raw_tokens = steps * raw_tokens_per_step
    manifest_roots = {
        "train_manifest_sha256": common.manifest_root(train_manifest),
        "val_manifest_sha256": common.manifest_root(val_manifest),
    }
    ablation_contract = {
        "schema": "helix.branch52.activation-checkpointing-ablation.v0",
        "branch52_profile": "activation_checkpointing_v0",
        "ablation_id": args.ablation_id,
        "changed_knobs": changed_knobs,
        "knobs": resolved_knobs,
        "seq_len": SEQ_LEN,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "target_causal_targets": target_causal_targets,
        "target_raw_tokens": target_raw_tokens,
        "eval_every": args.eval_every,
        "checkpoint_every": args.checkpoint_every,
        "validation_batches": args.validation_batches,
        "seed": args.seed,
        "manifest_roots": manifest_roots,
        "source_identity": source_identity,
        "full_corpus_pass": args.full_corpus_pass,
        "expected_full_corpus_raw_tokens": args.expected_full_corpus_raw_tokens,
        "full_corpus_plan": full_corpus_plan,
        "diminishing_return": {
            "enabled": diminishing_enabled,
            "window_evals": args.diminishing_window_evals,
            "min_improvement": args.diminishing_min_improvement,
            "patience_windows": args.diminishing_patience_windows,
            "min_optimizer_steps": args.diminishing_min_optimizer_steps,
        },
        "promotion_manifest": promotion_manifest,
        "promotion_manifest_root": promotion_manifest_root,
    }

    schedule = build_scheduler_state(
        policy=args.scheduler_policy,
        base_lr=cfg.lr,
        warmup_microbatches=cfg.warmup_steps,
        grad_accum=args.grad_accum,
        total_optimizer_steps=steps,
        min_lr_ratio=args.scheduler_min_lr_ratio,
    )
    step = 0
    offset = common.DataOffset()
    best_val_loss = math.inf
    best_val_step = 0
    last_val_loss: float | None = None
    validation_history: list[dict[str, Any]] = []
    stop_state: dict[str, Any] = {
        "stop_reason": "target_reached",
        "diminishing_decision": {"enabled": diminishing_enabled, "should_stop": False},
    }
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
        validation_history = list(state.get("validation_history", []))
        stop_state = dict(state.get("stop_state", stop_state))
        common.set_rng_state(state.get("rng_state"))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = (
        f"branch52-ablation-{args.ablation_id}-s512-b{args.batch_size}"
        f"-a{args.grad_accum}-t{target_causal_targets}-{stamp}"
    )
    run_root = RUN_ROOT / "artifacts" / "quality-vram-ablation-v0" / run_name
    run_root.mkdir(parents=True, exist_ok=False)
    harness_sha = sha256(Path(__file__))
    logger = RealtimeMLflowLogger(
        tracking_uri=args.mlflow_uri,
        experiment="helix-branch52-activation-checkpointing-v0",
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
            "target_causal_targets": target_causal_targets,
            "aligned_target_causal_targets": aligned_target_causal_targets,
            "target_raw_tokens": target_raw_tokens,
            "full_corpus_pass": args.full_corpus_pass,
            "expected_full_corpus_raw_tokens": args.expected_full_corpus_raw_tokens,
            "full_corpus_raw_tokens": full_corpus_plan["raw_tokens"] if full_corpus_plan else "none",
            "full_corpus_samples": full_corpus_plan["total_samples"] if full_corpus_plan else "none",
            "full_corpus_remaining_samples": full_corpus_plan["remaining_samples"] if full_corpus_plan else "none",
            "steps": steps,
            "resume_checkpoint": str(args.resume) if args.resume else "none",
            "resume_step": step,
            "initial_model_root": initial_model_root,
            "parameter_count_total": int(params["total"]),
            "parameter_count_trainable": int(params["trainable"]),
            "baseline_parameter_count_total": EXPECTED_PARAMETER_COUNT,
            "parameter_count_delta_from_branch50": (
                int(params["total"]) - EXPECTED_PARAMETER_COUNT
            ),
            "d_model": 512,
            "n_heads": 8,
            "n_loops": cfg.n_loops,
            "local_window": 64,
            "coarse_window": 128,
            "compressed_windows": 8,
            "compressed_views": 8,
            "ffn_expansion": cfg.ffn_expansion,
            "learning_rate": cfg.lr,
            "warmup_microbatches": cfg.warmup_steps,
            "warmup_optimizer_steps": warmup_optimizer_steps,
            "scheduler_policy": schedule["type"],
            "scheduler_total_optimizer_steps": schedule["total_optimizer_steps"],
            "scheduler_min_lr_ratio": schedule["minimum_lr_ratio_after_warmup"],
            "weight_decay": cfg.weight_decay,
            "grad_clip": cfg.grad_clip,
            "dropout": cfg.dropout,
            "attention_dropout": cfg.attn_dropout,
            "master_dtype": "float32",
            "amp_dtype": "bfloat16",
            "strict_nan_check": True,
            "grad_buffer_ratio": 0.0,
            "activation_checkpointing_requested": args.activation_checkpointing,
            "activation_checkpointing_instantiated": bool(
                model.gradient_checkpointing
            ),
            "ordering_algorithm": common.ORDERING_ALGORITHM,
            "validation_batches": args.validation_batches,
            "checkpoint_every": args.checkpoint_every,
            "shard_sha256_verified": not args.skip_shard_sha256,
            "diminishing_return_enabled": diminishing_enabled,
            "diminishing_window_evals": args.diminishing_window_evals,
            "diminishing_min_improvement": args.diminishing_min_improvement,
            "diminishing_patience_windows": args.diminishing_patience_windows,
            "diminishing_min_optimizer_steps": args.diminishing_min_optimizer_steps,
            "promotion_manifest_path": (
                str(args.promotion_manifest) if args.promotion_manifest else "none"
            ),
            "promotion_manifest_root": promotion_manifest_root or "none",
            "single_factor_contract": promotion_manifest is None,
            "allowed_optimizer_geometries": ",".join(
                f"{batch}x{accum}"
                for batch, accum in sorted(ALLOWED_OPTIMIZER_GEOMETRIES)
            ),
            "thunderline_projection_schema": "thunderline.training.mission.projection.v0",
        },
        tags={
            "run_kind": (
                "branch52_promoted_full_corpus_v0"
                if promotion_manifest is not None and args.full_corpus_pass
                else (
                    "branch52_promoted_combined_pilot_v0"
                    if promotion_manifest is not None
                    else "branch52_single_factor_ablation_v0"
                )
            ),
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
    (run_root / "thunderline_training_projection.json").write_text(
        json.dumps(
            {
                "schema": "thunderline.training.mission.projection.v0",
                "profile": "HELIX_BRANCH52_ACTIVATION_CHECKPOINTING_V0",
                "mission": {
                    "workload": "helix_model_training",
                    "production_effect": "none",
                    "source_head": source_identity["source_head"],
                    "source_tree": source_identity["source_tree"],
                    "input_binding": manifest_roots,
                    "expected_outputs": [
                        "terminal_checkpoint",
                        "best_checkpoint",
                        "mlflow_spool",
                        "terminal_summary",
                    ],
                },
                "pass": {
                    "name": "train_ablation",
                    "ablation_contract_root": canonical_root(ablation_contract),
                    "batch_size": args.batch_size,
                    "gradient_accumulation": args.grad_accum,
                    "target_causal_targets": target_causal_targets,
                    "runner": Path(__file__).name,
                },
                "artifact_custody": {
                    "run_root": str(run_root),
                    "heavy_bytes_live_outside_git": True,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
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
    set_optimizer_lr(
        optimizer,
        base_lr=cfg.lr,
        optimizer_step_number=step + 1,
        warmup_optimizer_steps=warmup_optimizer_steps,
        total_optimizer_steps=steps,
        min_lr_ratio=args.scheduler_min_lr_ratio,
        policy=args.scheduler_policy,
    )
    try:
        group_iter = iter_accumulation_groups(
            sample_iter,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            allow_partial_batch=args.full_corpus_pass,
            allow_partial_accumulation=args.full_corpus_pass,
        )
        for group in group_iter:
            should_stop_now = False
            if args.max_optimizer_steps and step >= args.max_optimizer_steps:
                break
            group_targets = sum(
                common.count_causal_targets(batch["labels"]) for batch, _ in group
            )
            if group_targets <= 0:
                raise RuntimeError(f"EMPTY_ACCUMULATION_GROUP step={step}")
            for batch, batch_offset in group:
                device_batch = common.to_device(batch, device=device)
                with autocast:
                    output = model(**device_batch, return_dict=True)
                    loss = output.loss
                if loss is None or not torch.isfinite(loss):
                    raise RuntimeError(f"NONFINITE_LOSS step={step}")
                targets = common.count_causal_targets(device_batch["labels"])
                (loss * (targets / group_targets)).backward()
                if args.activation_checkpointing and (
                    model._gradient_checkpoint_function_calls
                    <= model._gradient_checkpoint_forward_calls
                ):
                    raise RuntimeError(
                        "ACTIVATION_CHECKPOINTING_RECOMPUTE_NOT_OBSERVED"
                    )
                losses.append((float(loss.detach().cpu()), targets))
                offset = batch_offset
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"NONFINITE_GRAD_NORM step={step}")
            lr = set_optimizer_lr(
                optimizer,
                base_lr=cfg.lr,
                optimizer_step_number=step + 1,
                warmup_optimizer_steps=warmup_optimizer_steps,
                total_optimizer_steps=steps,
                min_lr_ratio=args.scheduler_min_lr_ratio,
                policy=args.scheduler_policy,
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
                    "train/gradient_checkpoint_forward_calls": float(
                        model._gradient_checkpoint_forward_calls
                    ),
                    "train/gradient_checkpoint_function_calls": float(
                        model._gradient_checkpoint_function_calls
                    ),
                    "train/gradient_checkpoint_recompute_calls": float(
                        model._gradient_checkpoint_function_calls
                        - model._gradient_checkpoint_forward_calls
                    ),
                    "train/gradient_norm_pre_clip": float(grad_norm.detach().cpu()),
                    "system/peak_vram_bytes": float(torch.cuda.max_memory_allocated()),
                    "train/skipped_batches": 0.0,
                    "train/nonfinite_events": 0.0,
                    "train/accum_microbatches": float(len(losses)),
                    "train/accum_samples": float(group[-1][1].samples_seen - (group[0][1].samples_seen - group[0][0]["input_ids"].shape[0])),
                },
                step=step,
            )
            losses.clear()

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
                validation_history.append(
                    {
                        "step": step,
                        "val_loss": val_loss,
                        "val_ppl": common.perplexity(val_loss),
                        "val_targets": val_targets,
                    }
                )
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
                decision = diminishing_return_decision(
                    validation_history,
                    window_evals=args.diminishing_window_evals,
                    min_improvement=args.diminishing_min_improvement,
                    patience_windows=args.diminishing_patience_windows,
                    min_optimizer_steps=args.diminishing_min_optimizer_steps,
                )
                stop_state = {
                    "stop_reason": "diminishing_return"
                    if decision.get("should_stop")
                    else "target_reached",
                    "diminishing_decision": decision,
                }
                if decision.get("enabled"):
                    logger.log_metrics(
                        {
                            "diminishing/bad_windows": float(decision.get("bad_windows", 0)),
                            "diminishing/last_window_improvement": float(
                                decision.get("last_window_improvement", 0.0) or 0.0
                            ),
                            "diminishing/should_stop": 1.0
                            if decision.get("should_stop")
                            else 0.0,
                        },
                        step=step,
                        phase="validation",
                    )
                model.train()
                torch.cuda.empty_cache()
                if decision.get("should_stop"):
                    should_stop_now = True
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
                    validation_history=validation_history,
                    stop_state=stop_state,
                )
                logger._append(
                    {
                        "event": "checkpoint",
                        "step": step,
                        "path": str(checkpoint),
                        "sha256": sha256(checkpoint),
                        "data_offset": asdict(offset),
                        "validation_history_len": len(validation_history),
                        "stop_state": stop_state,
                        "ts": time.time(),
                    }
                )
            if should_stop_now:
                break
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
            validation_history=validation_history,
            stop_state=stop_state,
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
            and restored["validation_history"] == validation_history
            and restored["stop_state"] == stop_state
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

    terminal_record = terminal_status_record(
        max_optimizer_steps=args.max_optimizer_steps,
        stop_state=stop_state,
        mlflow_errors=logger.mlflow_errors,
    )
    terminal = {
        **terminal_record,
        "steps": step,
        "ablation_id": args.ablation_id,
        "seq_len": SEQ_LEN,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "raw_tokens_per_optimizer_step": raw_tokens_per_step,
        "causal_targets_per_optimizer_step": causal_targets_per_step,
        "target_causal_targets": target_causal_targets,
        "aligned_target_causal_targets": aligned_target_causal_targets,
        "target_raw_tokens": target_raw_tokens,
        "full_corpus_pass": args.full_corpus_pass,
        "expected_full_corpus_raw_tokens": args.expected_full_corpus_raw_tokens,
        "full_corpus_plan": full_corpus_plan,
        "initial_model_root": initial_model_root,
        "final_model_root": restored_model_root,
        "parameter_count": params,
        "baseline_parameter_count": EXPECTED_PARAMETER_COUNT,
        "parameter_count_delta_from_branch50": (
            int(params["total"]) - EXPECTED_PARAMETER_COUNT
        ),
        "learning_rate": cfg.lr,
        "warmup_microbatches": cfg.warmup_steps,
        "weight_decay": cfg.weight_decay,
        "grad_clip": cfg.grad_clip,
        "dropout": cfg.dropout,
        "attention_dropout": cfg.attn_dropout,
        "ffn_expansion": cfg.ffn_expansion,
        "n_loops": cfg.n_loops,
        "activation_checkpointing_requested": args.activation_checkpointing,
        "activation_checkpointing_instantiated": bool(model.gradient_checkpointing),
        "activation_checkpointing_forward_calls": (
            model._gradient_checkpoint_forward_calls
        ),
        "activation_checkpointing_function_calls": (
            model._gradient_checkpoint_function_calls
        ),
        "activation_checkpointing_recompute_calls": (
            model._gradient_checkpoint_function_calls
            - model._gradient_checkpoint_forward_calls
        ),
        "activation_checkpointing_executed": bool(
            args.activation_checkpointing
            and model._gradient_checkpoint_function_calls
            > model._gradient_checkpoint_forward_calls
        ),
        "scheduler_policy": args.scheduler_policy,
        "scheduler_min_lr_ratio": args.scheduler_min_lr_ratio,
        "scheduler": schedule,
        "checkpoint_every": args.checkpoint_every,
        "eval_every": args.eval_every,
        "validation_history": validation_history,
        "stop_state": stop_state,
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
