#!/usr/bin/env python3
"""Train the frozen 101M/T1024 Sutra baseline with passive state probes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from automata_state_probe import observe_hidden_sequence
from helix_lm.hf_model import HelixForCausalLM
from helix_lm.tokenizer import HelixTokenizer
from realtime_mlflow import RealtimeMLflowLogger
from sutra_100m_preflight import (
    DATASET,
    DATASET_REVISION,
    EXPECTED_PARAMETER_COUNT,
    SEQ_LEN,
    build_config,
)
from sutra_stream import SutraStreamOffset, iter_packed_sequences, resume_rows


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "helix-sutra100m-automata-capacity-v0"
ESTIMATED_CHECKPOINT_BYTES = EXPECTED_PARAMETER_COUNT * 12


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def verify_source(expected_head: str, expected_tree: str) -> dict[str, str]:
    observed = {
        "source_head": git_value("rev-parse", "HEAD"),
        "source_tree": git_value("rev-parse", "HEAD^{tree}"),
        "source_dirty": str(bool(git_value("status", "--porcelain"))).lower(),
    }
    if observed["source_head"] != expected_head:
        raise SystemExit(f"REFUSED: source HEAD drift: {observed}")
    if observed["source_tree"] != expected_tree:
        raise SystemExit(f"REFUSED: source tree drift: {observed}")
    if observed["source_dirty"] != "false":
        raise SystemExit(f"REFUSED: source checkout is dirty: {observed}")
    return observed


def canonical_root(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def storage_court(
    artifacts_root: Path,
    *,
    max_optimizer_steps: int,
    checkpoint_every: int,
    estimated_checkpoint_bytes: int = ESTIMATED_CHECKPOINT_BYTES,
) -> dict[str, int | str]:
    existing = artifacts_root.resolve()
    while not existing.exists():
        if existing.parent == existing:
            raise RuntimeError(f"no existing parent for artifacts root {artifacts_root}")
        existing = existing.parent
    periodic_checkpoints = math.ceil(max_optimizer_steps / checkpoint_every)
    required = estimated_checkpoint_bytes * (periodic_checkpoints + 3)
    free = shutil.disk_usage(existing).free
    if free < required:
        raise SystemExit(
            "UNAVAILABLE: checkpoint filesystem lacks bounded free space: "
            f"free={free} required={required} root={existing}"
        )
    return {
        "filesystem_root": str(existing),
        "free_bytes": free,
        "estimated_checkpoint_bytes": estimated_checkpoint_bytes,
        "planned_periodic_checkpoints": periodic_checkpoints,
        "required_free_bytes": required,
    }


def iter_batches(
    sequences: Iterator[tuple[torch.Tensor, SutraStreamOffset]], *, batch_size: int
) -> Iterator[tuple[dict[str, torch.Tensor], SutraStreamOffset]]:
    while True:
        rows: list[torch.Tensor] = []
        last_offset: SutraStreamOffset | None = None
        try:
            for _ in range(batch_size):
                row, last_offset = next(sequences)
                rows.append(row)
        except StopIteration:
            return
        if len(rows) != batch_size or last_offset is None:
            return
        input_ids = torch.stack(rows)
        yield {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
        }, last_offset


def count_causal_targets(labels: torch.Tensor) -> int:
    return int((labels[:, 1:] != -100).sum().item())


def aligned_training_budget(
    *, target_causal_targets: int, batch_size: int, grad_accum: int
) -> dict[str, int]:
    if min(target_causal_targets, batch_size, grad_accum) <= 0:
        raise ValueError("training budget values must be positive")
    causal_targets_per_optimizer_step = batch_size * grad_accum * (SEQ_LEN - 1)
    optimizer_steps = math.ceil(target_causal_targets / causal_targets_per_optimizer_step)
    return {
        "requested_causal_targets": target_causal_targets,
        "aligned_causal_targets": optimizer_steps * causal_targets_per_optimizer_step,
        "causal_targets_per_optimizer_step": causal_targets_per_optimizer_step,
        "optimizer_steps": optimizer_steps,
    }


def scheduler_state(
    *, base_lr: float, warmup_microbatches: int, grad_accum: int
) -> dict[str, Any]:
    if base_lr <= 0 or warmup_microbatches <= 0 or grad_accum <= 0:
        raise ValueError("scheduler values must be positive")
    return {
        "type": "linear_warmup_then_constant",
        "base_lr": base_lr,
        "warmup_microbatches": warmup_microbatches,
        "grad_accum": grad_accum,
        "warmup_optimizer_steps": max(1, warmup_microbatches // grad_accum),
        "minimum_lr_ratio_after_warmup": 1.0,
    }


def set_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    optimizer_step_number: int,
    warmup_optimizer_steps: int,
) -> float:
    lr = base_lr * min(1.0, max(1, optimizer_step_number) / warmup_optimizer_steps)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    offset: SutraStreamOffset,
    run_contract: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "offset": offset.to_dict(),
            "rng_state": rng_state(),
            "run_contract": run_contract,
            "run_contract_root": canonical_root(run_contract),
        },
        temporary,
    )
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_contract: dict[str, Any],
) -> tuple[int, SutraStreamOffset]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("run_contract_root") != canonical_root(expected_contract):
        raise SystemExit("REFUSED: checkpoint run-contract mismatch")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    restore_rng_state(state["rng_state"])
    return int(state["step"]), SutraStreamOffset(**state["offset"])


def evaluate(
    model: HelixForCausalLM,
    tokenizer: HelixTokenizer,
    dataset: Any,
    *,
    validation_rows: int,
    validation_batches: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    rows = dataset.take(validation_rows)
    sequences = iter_packed_sequences(rows, tokenizer, seq_len=SEQ_LEN)
    total_loss = 0.0
    total_targets = 0
    model.eval()
    with torch.no_grad():
        for index, (batch, _) in enumerate(iter_batches(sequences, batch_size=batch_size)):
            if index >= validation_batches:
                break
            device_batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**device_batch, return_dict=True)
            if output.loss is None or not torch.isfinite(output.loss):
                raise RuntimeError("non-finite validation loss")
            targets = count_causal_targets(device_batch["labels"])
            total_loss += float(output.loss.detach().cpu()) * targets
            total_targets += targets
    if total_targets == 0:
        raise RuntimeError("validation produced zero causal targets")
    loss = total_loss / total_targets
    return {"loss": loss, "ppl": math.exp(min(loss, 20)), "causal_targets": total_targets}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-source-head", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--target-causal-targets", type=int, default=1_500_000_000)
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--ffn-expansion", type=float, default=2.5)
    parser.add_argument("--expected-parameter-count", type=int, default=EXPECTED_PARAMETER_COUNT)
    parser.add_argument("--tokenizer", choices=("gpt2", "lengthmax"), default="gpt2")
    parser.add_argument("--tokenizer-artifact", type=Path)
    parser.add_argument("--tokenizer-artifact-sha256")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-microbatches", type=int, default=2_000)
    parser.add_argument("--validation-rows", type=int, default=1_024)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--probe-every", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--mlflow-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "https://mlflow.thunderline.net")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.target_causal_targets,
        args.batch_size,
        args.grad_accum,
        args.warmup_microbatches,
        args.validation_rows,
        args.validation_batches,
        args.eval_every,
        args.checkpoint_every,
        args.probe_every,
    ) <= 0:
        raise SystemExit("REFUSED: counts and intervals must be positive")
    if args.max_optimizer_steps < 0:
        raise SystemExit("REFUSED: --max-optimizer-steps cannot be negative")
    if args.ffn_expansion <= 0 or args.expected_parameter_count <= 0:
        raise SystemExit("REFUSED: FFN expansion and expected parameter count must be positive")
    if args.tokenizer == "lengthmax":
        if args.tokenizer_artifact is None or args.tokenizer_artifact_sha256 is None:
            raise SystemExit("REFUSED: LengthMAX requires an artifact path and expected SHA-256")
        tokenizer_artifact = args.tokenizer_artifact.resolve()
        if not tokenizer_artifact.is_file():
            raise SystemExit(f"REFUSED: LengthMAX artifact is absent: {tokenizer_artifact}")
        observed_tokenizer_sha256 = file_sha256(tokenizer_artifact)
        if observed_tokenizer_sha256 != args.tokenizer_artifact_sha256:
            raise SystemExit(
                "REFUSED: LengthMAX artifact hash mismatch: "
                f"{observed_tokenizer_sha256} != {args.tokenizer_artifact_sha256}"
            )
        tokenizer_spec = f"lengthmax:{tokenizer_artifact}"
    else:
        if args.tokenizer_artifact is not None or args.tokenizer_artifact_sha256 is not None:
            raise SystemExit("REFUSED: GPT-2 does not accept a LengthMAX artifact")
        tokenizer_artifact = None
        observed_tokenizer_sha256 = None
        tokenizer_spec = "gpt2"
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise SystemExit("UNAVAILABLE: exact RTX 5080 CUDA runtime is absent")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("UNAVAILABLE: BF16 is unsupported")

    source = verify_source(args.expected_source_head, args.expected_source_tree)
    budget = aligned_training_budget(
        target_causal_targets=args.target_causal_targets,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
    )
    bounded_optimizer_steps = args.max_optimizer_steps or budget["optimizer_steps"]
    bounded_optimizer_steps = min(bounded_optimizer_steps, budget["optimizer_steps"])
    storage = storage_court(
        args.artifacts_root,
        max_optimizer_steps=bounded_optimizer_steps,
        checkpoint_every=args.checkpoint_every,
        estimated_checkpoint_bytes=args.expected_parameter_count * 12,
    )
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tokenizer = HelixTokenizer(tokenizer_spec, local_files_only=True)
    if len(tokenizer) != 50_257 or tokenizer.eos_token_id != 50_256:
        raise SystemExit("REFUSED: tokenizer identity mismatch")
    config = build_config(
        batch_size=args.batch_size,
        ffn_expansion=args.ffn_expansion,
        tokenizer_name=tokenizer_spec,
    )
    config.lr = args.learning_rate
    config.weight_decay = args.weight_decay
    config.memory_efficient_forward = True
    model = HelixForCausalLM(config).to("cuda")
    observed_parameter_count = model.count_parameters()["total"]
    if observed_parameter_count != args.expected_parameter_count:
        raise SystemExit(
            "REFUSED: model parameter-count drift: "
            f"{observed_parameter_count} != {args.expected_parameter_count}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    schedule = scheduler_state(
        base_lr=args.learning_rate,
        warmup_microbatches=args.warmup_microbatches,
        grad_accum=args.grad_accum,
    )

    run_contract = {
        **source,
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "validation_posture": "first_ordered_rows_held_out_from_training",
        "validation_rows": args.validation_rows,
        "tokenizer": args.tokenizer,
        "tokenizer_artifact_sha256": observed_tokenizer_sha256,
        "tokenizer_vocab_size": len(tokenizer),
        "parameter_count": observed_parameter_count,
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "n_columns": config.n_columns,
        "nodes_per_column": list(config.nodes_per_column),
        "n_loops": config.n_loops,
        "ffn_expansion": config.ffn_expansion,
        "local_window": config.local_window,
        "coarse_window": config.coarse_window,
        "compressed_windows": config.compressed_windows,
        "compressed_views": config.compressed_views,
        "sequence_length": SEQ_LEN,
        "lateral_p": config.lateral_p,
        "vertical_p": config.vertical_p,
        "vertical_depth": config.vertical_depth,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "training_budget": budget,
        "scheduler": schedule,
        "seed": seed,
        "state_probe_posture": "detached_observer_only_no_model_feedback",
    }
    contract_root = canonical_root(run_contract)
    step = 0
    offset = SutraStreamOffset()
    if args.resume:
        step, offset = load_checkpoint(
            args.resume, model=model, optimizer=optimizer, expected_contract=run_contract
        )
    set_optimizer_lr(
        optimizer,
        base_lr=args.learning_rate,
        optimizer_step_number=step + 1,
        warmup_optimizer_steps=int(schedule["warmup_optimizer_steps"]),
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    tokenizer_code = "g2" if args.tokenizer == "gpt2" else "lm"
    ffn_code = str(args.ffn_expansion).replace(".", "")
    lr_code = f"{args.learning_rate:.0e}".replace("-", "m")
    run_name = f"hlx-b49-sutra100m-{tokenizer_code}-t1024-l3-f{ffn_code}-lr{lr_code}-s42-{stamp}"
    run_root = args.artifacts_root / run_name
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "run_contract.json").write_text(
        json.dumps(run_contract, indent=2, sort_keys=True) + "\n"
    )
    (run_root / "storage_court.json").write_text(
        json.dumps(storage, indent=2, sort_keys=True) + "\n"
    )
    if tokenizer_artifact is not None:
        copied_artifact = run_root / "lengthmax-tokenizer.json"
        shutil.copy2(tokenizer_artifact, copied_artifact)
        if file_sha256(copied_artifact) != observed_tokenizer_sha256:
            raise RuntimeError("LengthMAX artifact changed while entering run custody")

    logger = RealtimeMLflowLogger(
        tracking_uri=args.mlflow_uri,
        experiment=EXPERIMENT,
        run_name=run_name,
        spool_path=run_root / "mlflow-events.jsonl",
        params={
            **run_contract,
            "run_contract_root": contract_root,
            **{f"storage_{key}": value for key, value in storage.items()},
        },
        tags={
            "run_kind": "sutra100m_compound_candidate_with_passive_state_probe_v0",
            "comparison_posture": "exploratory_compound_not_single_factor",
            "production_effect": "none",
            "automata_feedback": "disabled",
        },
    )
    if logger.start() is None:
        raise RuntimeError("MLFLOW_START_FAILED: refusing untracked training")

    from datasets import load_dataset

    stream = load_dataset(
        DATASET, split="train", revision=DATASET_REVISION, streaming=True
    )
    train_rows = resume_rows(stream.skip(args.validation_rows), offset)
    sequence_iter = iter_packed_sequences(
        train_rows, tokenizer, seq_len=SEQ_LEN, start=offset
    )

    device = torch.device("cuda")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro_count = 0
    accumulated_targets = 0
    accumulated_weighted_loss = 0.0
    started = time.time()
    run_status = "FINISHED"
    last_validation: dict[str, float] | None = None
    state_probe_path = run_root / "state-probes.jsonl"
    try:
        for batch, batch_offset in iter_batches(sequence_iter, batch_size=args.batch_size):
            if step >= bounded_optimizer_steps:
                break
            if offset.causal_targets_emitted >= budget["aligned_causal_targets"]:
                break
            device_batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            should_probe = (step + 1) % args.probe_every == 0 and micro_count == args.grad_accum - 1
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    **device_batch,
                    output_hidden_states=should_probe,
                    return_dict=True,
                )
            if output.loss is None or not torch.isfinite(output.loss):
                raise RuntimeError(f"non-finite training loss at optimizer step {step}")
            targets = count_causal_targets(device_batch["labels"])
            (output.loss / args.grad_accum).backward()
            accumulated_targets += targets
            accumulated_weighted_loss += float(output.loss.detach().cpu()) * targets
            micro_count += 1
            offset = batch_offset

            if should_probe:
                if not isinstance(output.hidden_states, torch.Tensor):
                    raise RuntimeError("requested hidden-state probe was unavailable")
                probes = observe_hidden_sequence(
                    output.hidden_states, device_batch["input_ids"], segment_tokens=64
                )
                with state_probe_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "optimizer_step_before_update": step,
                                "offset": offset.to_dict(),
                                "records": [record.to_dict() for record in probes],
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )

            if micro_count < args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            current_lr = set_optimizer_lr(
                optimizer,
                base_lr=args.learning_rate,
                optimizer_step_number=step + 1,
                warmup_optimizer_steps=int(schedule["warmup_optimizer_steps"]),
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step += 1
            train_loss = accumulated_weighted_loss / accumulated_targets
            elapsed = max(time.time() - started, 1e-6)
            logger.log_metrics(
                {
                    "train/loss": train_loss,
                    "train/ppl": math.exp(min(train_loss, 20)),
                    "train/causal_targets_seen": float(offset.causal_targets_emitted),
                    "train/source_raw_utf8_bytes_read": float(offset.raw_utf8_bytes_read),
                    "train/causal_targets_per_second": offset.causal_targets_emitted / elapsed,
                    "train/lr": current_lr,
                    "system/peak_vram_bytes": float(torch.cuda.max_memory_allocated()),
                },
                step=step,
            )
            micro_count = 0
            accumulated_targets = 0
            accumulated_weighted_loss = 0.0

            if step % args.eval_every == 0:
                last_validation = evaluate(
                    model,
                    tokenizer,
                    stream,
                    validation_rows=args.validation_rows,
                    validation_batches=args.validation_batches,
                    batch_size=args.batch_size,
                    device=device,
                )
                logger.log_metrics(
                    {f"val/{key}": value for key, value in last_validation.items()},
                    step=step,
                    phase="validation",
                )
                model.train()

            if step % args.checkpoint_every == 0:
                save_checkpoint(
                    run_root / "checkpoints" / f"step-{step:08d}.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    offset=offset,
                    run_contract=run_contract,
                )
            set_optimizer_lr(
                optimizer,
                base_lr=args.learning_rate,
                optimizer_step_number=step + 1,
                warmup_optimizer_steps=int(schedule["warmup_optimizer_steps"]),
            )

        periodic_terminal = run_root / "checkpoints" / f"step-{step:08d}.pt"
        if step > 0 and periodic_terminal.exists():
            terminal_checkpoint = periodic_terminal
        else:
            terminal_checkpoint = run_root / "checkpoints" / "terminal.pt"
            save_checkpoint(
                terminal_checkpoint,
                model=model,
                optimizer=optimizer,
                step=step,
                offset=offset,
                run_contract=run_contract,
            )
    except BaseException:
        run_status = "FAILED"
        raise
    finally:
        logger.finish(status=run_status)

    terminal = {
        "schema": "helix.sutra-100m-baseline-terminal.v0",
        "status": "PASS",
        "run_name": run_name,
        "run_contract_root": contract_root,
        "optimizer_steps": step,
        "training_budget": budget,
        "scheduler": schedule,
        "offset": offset.to_dict(),
        "last_validation": last_validation,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "mlflow_run_id": logger.run_id,
        "mlflow_errors": logger.mlflow_errors,
        "state_probe_path": str(state_probe_path),
        "terminal_checkpoint": str(terminal_checkpoint),
        "production_effect": "none",
    }
    terminal["terminal_root"] = canonical_root(terminal)
    (run_root / "terminal.json").write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(terminal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
