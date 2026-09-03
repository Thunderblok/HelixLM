#!/usr/bin/env python3
"""Evaluate one frozen Helix checkpoint across local, retrieval, and state arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from helix_lm.hf_model import HelixForCausalLM
from helix_lm.tokenizer import HelixTokenizer
from long_memory_capacity import benchmark_contract, build_cases, canonical_root, render_arm
from realtime_mlflow import RealtimeMLflowLogger
from sutra_100m_preflight import EXPECTED_PARAMETER_COUNT, SEQ_LEN, build_config


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "helix-sutra100m-automata-capacity-v0"
FLOP_METHOD = "estimated_2_x_active_parameters_x_input_tokens_x_recurrent_loops"


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def verify_evaluator_source(expected_head: str, expected_tree: str) -> dict[str, str]:
    result = {
        "evaluator_head": git_value("rev-parse", "HEAD"),
        "evaluator_tree": git_value("rev-parse", "HEAD^{tree}"),
        "evaluator_dirty": str(bool(git_value("status", "--porcelain"))).lower(),
    }
    if result != {
        "evaluator_head": expected_head,
        "evaluator_tree": expected_tree,
        "evaluator_dirty": "false",
    }:
        raise SystemExit(f"REFUSED: evaluator source drift: {result}")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_checkpoint(path: Path, *, device: torch.device) -> tuple[HelixForCausalLM, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "step", "offset", "run_contract", "run_contract_root"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise SystemExit(f"REFUSED: checkpoint missing fields: {missing}")
    if canonical_root(checkpoint["run_contract"]) != checkpoint["run_contract_root"]:
        raise SystemExit("REFUSED: checkpoint run-contract root mismatch")
    config = build_config(batch_size=1)
    config.memory_efficient_forward = False
    model = HelixForCausalLM(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    if model.count_parameters()["total"] != EXPECTED_PARAMETER_COUNT:
        raise SystemExit("REFUSED: evaluator parameter-count drift")
    identity = {
        "checkpoint_path": str(path.resolve()),
        "checkpoint_sha256": sha256_file(path),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_offset": checkpoint["offset"],
        "checkpoint_run_contract_root": checkpoint["run_contract_root"],
        "checkpoint_source_head": checkpoint["run_contract"].get("source_head"),
        "checkpoint_source_tree": checkpoint["run_contract"].get("source_tree"),
    }
    return model, identity


def score_rendered(
    model: HelixForCausalLM,
    rendered: dict[str, Any],
    *,
    device: torch.device,
    active_parameters: int,
    loops: int,
    repetitions: int,
) -> dict[str, Any]:
    prompt_ids = rendered["prompt_ids"]
    answer_ids = rendered["answer_ids"]
    full_ids = prompt_ids + answer_ids
    if len(full_ids) > SEQ_LEN:
        raise ValueError("rendered prompt and answer exceed model context")
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    labels[:, len(prompt_ids) :] = input_ids[:, len(prompt_ids) :]
    latencies = []
    output = None
    with torch.no_grad():
        for _ in range(repetitions):
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=True,
                    return_dict=True,
                )
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1_000)
    if output is None or output.logits is None or output.loss is None:
        raise RuntimeError("model did not return scoreable logits and loss")
    answer_logits = output.logits[:, len(prompt_ids) - 1 : -1, :].float()
    targets = input_ids[:, len(prompt_ids) :]
    if answer_logits.shape[1] != targets.shape[1]:
        raise AssertionError("answer logit/target alignment drift")
    token_nll = F.cross_entropy(
        answer_logits.reshape(-1, answer_logits.shape[-1]), targets.reshape(-1), reduction="mean"
    )
    predicted = answer_logits.argmax(dim=-1)
    token_correct = int((predicted == targets).sum().item())
    answer_tokens = int(targets.numel())
    past_key_values = output.past_key_values
    if past_key_values is not None:
        raise RuntimeError("Helix unexpectedly returned a KV cache")
    input_tokens = len(full_ids)
    return {
        "answer_mean_nll": float(token_nll.cpu()),
        "answer_perplexity": math.exp(min(float(token_nll.cpu()), 20)),
        "answer_token_accuracy": token_correct / answer_tokens,
        "answer_exact_match": token_correct == answer_tokens,
        "answer_tokens": answer_tokens,
        "latency_ms_median": statistics.median(latencies),
        "latency_ms_samples": latencies,
        "input_tokens": input_tokens,
        "model_input_bytes": input_tokens * 8,
        "kv_cache_bytes": 0,
        "inference_flops_estimate": 2 * active_parameters * input_tokens * loops,
        "inference_flops_method": FLOP_METHOD,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-evaluator-head", required=True)
    parser.add_argument("--expected-evaluator-tree", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--distances", type=int, nargs="+", default=[10_000, 50_000, 250_000, 750_000]
    )
    parser.add_argument(
        "--mlflow-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "https://mlflow.thunderline.net")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repetitions <= 0 or any(distance <= 0 for distance in args.distances):
        raise SystemExit("REFUSED: repetitions and distances must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise SystemExit("UNAVAILABLE: exact RTX 5080 runtime absent")
    source = verify_evaluator_source(args.expected_evaluator_head, args.expected_evaluator_tree)
    device = torch.device("cuda")
    tokenizer = HelixTokenizer("gpt2", local_files_only=True)
    cases = build_cases(tokenizer, distances=args.distances, seed=42)
    contract = benchmark_contract(cases)
    model, checkpoint = load_frozen_checkpoint(args.checkpoint, device=device)
    active_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if active_parameters != EXPECTED_PARAMETER_COUNT:
        raise SystemExit("REFUSED: active parameter count drift")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    run_name = f"hlx-long-memory-abc-{checkpoint['checkpoint_step']}-{stamp}"
    run_root = args.output_root / run_name
    run_root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "helix.long-memory-abc-run.v0",
        "source": source,
        "checkpoint": checkpoint,
        "benchmark": contract,
        "active_parameters": active_parameters,
        "sequence_length": SEQ_LEN,
        "repetitions": args.repetitions,
        "distances": args.distances,
        "production_effect": "none",
    }
    manifest["manifest_root"] = canonical_root(manifest)
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger = RealtimeMLflowLogger(
        tracking_uri=args.mlflow_uri,
        experiment=EXPERIMENT,
        run_name=run_name,
        spool_path=run_root / "mlflow-events.jsonl",
        params={
            "evaluator_head": source["evaluator_head"],
            "evaluator_tree": source["evaluator_tree"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "checkpoint_step": checkpoint["checkpoint_step"],
            "benchmark_contract_root": contract["contract_root"],
            "active_parameters": active_parameters,
            "distances": args.distances,
        },
        tags={
            "run_kind": "frozen_checkpoint_long_memory_abc_v0",
            "production_effect": "none",
        },
    )
    if logger.start() is None:
        raise RuntimeError("MLFLOW_START_FAILED: refusing untracked evaluation")

    rows = []
    status = "FINISHED"
    try:
        for case_index, case in enumerate(cases):
            for arm_index, arm in enumerate(("A", "B", "C")):
                preparation_started = time.perf_counter()
                answer_ids = tokenizer.encode(case.answer, add_special_tokens=False)
                rendered = render_arm(
                    case,
                    tokenizer,
                    arm=arm,
                    max_prompt_tokens=SEQ_LEN - len(answer_ids),
                )
                preparation_ms = (time.perf_counter() - preparation_started) * 1_000
                scored = score_rendered(
                    model,
                    rendered,
                    device=device,
                    active_parameters=active_parameters,
                    loops=3,
                    repetitions=args.repetitions,
                )
                row = {
                    "case_id": case.case_id,
                    "task": case.task,
                    "history_distance": case.history_distance,
                    "arm": arm,
                    "raw_history_tokens": len(case.prefix_token_ids) + case.history_distance,
                    "raw_history_token_bytes": (len(case.prefix_token_ids) + case.history_distance) * 4,
                    "live_context_bytes": rendered["live_context_token_count"] * 4,
                    "retrieved_bytes": rendered["retrieved_bytes"],
                    "automata_state_bytes": rendered["automata_state_bytes"],
                    "transition_log_bytes": rendered["transition_log_bytes"],
                    "access_preparation_latency_ms": preparation_ms,
                    "active_parameters": active_parameters,
                    **scored,
                }
                rows.append(row)
                logger.log_metrics(
                    {
                        "eval/answer_token_accuracy": row["answer_token_accuracy"],
                        "eval/answer_exact_match": float(row["answer_exact_match"]),
                        "eval/answer_mean_nll": row["answer_mean_nll"],
                        "eval/latency_ms": row["latency_ms_median"],
                        "eval/retrieved_bytes": row["retrieved_bytes"],
                        "eval/automata_state_bytes": row["automata_state_bytes"],
                        "eval/transition_log_bytes": row["transition_log_bytes"],
                    },
                    step=case_index * 3 + arm_index,
                    phase=f"arm_{arm}",
                )
    except BaseException:
        status = "FAILED"
        raise
    finally:
        logger.finish(status=status)

    aggregates: dict[str, dict[str, Any]] = {}
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["history_distance"], row["arm"])].append(row)
    for key, values in grouped.items():
        task, distance, arm = key
        aggregates[f"{task}:{distance}:{arm}"] = {
            "answer_token_accuracy": statistics.mean(value["answer_token_accuracy"] for value in values),
            "answer_exact_match_rate": statistics.mean(float(value["answer_exact_match"]) for value in values),
            "answer_mean_nll": statistics.mean(value["answer_mean_nll"] for value in values),
            "latency_ms_median": statistics.median(value["latency_ms_median"] for value in values),
        }
    terminal = {
        "schema": "helix.long-memory-abc-terminal.v0",
        "status": "PASS",
        "manifest_root": manifest["manifest_root"],
        "mlflow_run_id": logger.run_id,
        "mlflow_errors": logger.mlflow_errors,
        "rows": rows,
        "accuracy_by_history_distance": aggregates,
        "production_effect": "none",
    }
    terminal["terminal_root"] = canonical_root(terminal)
    (run_root / "terminal.json").write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n")
    print(json.dumps(terminal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
