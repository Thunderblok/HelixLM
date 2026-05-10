#!/usr/bin/env python3
"""
nas_helixlm.py  v2.3  —  Neural Architecture Search for HelixLM

DATA PATH (native):
  load_dataset(streaming) -> Python list of strings -> Trainer(train_texts=...)
  which internally builds DocumentAwareDataset / create_document_loader.
  No custom collators, no foreign masking logic.

SCREENING PHILOSOPHY:
  - bfloat16 on GPU (fp32 on CPU). bf16 has the same exponent range as fp32,
    so it avoids the NaN/underflow problems of fp16 on SSM scans and Titans
    memory updates, while giving 16-bit speed/memory savings.
  - Viable-config table replaces blind TPE over interdependent params.
  - SSM + Titans combined config included (VRAM impact of Titans is ~0.3 MB).

FAILURE MODES CAUGHT:
  1. Unlimited fail tolerance      — wrapper catches Exception, returns inf
  2. Proper Optuna pruning         — TrialPruned is re-raised, NOT swallowed
  3. Frozen-trial watchdog         — SIGALRM aborts if epoch > 3× expected
  4. NaN / all-batches-skipped     — loss=0 & PPL=1 is caught as dead trial
  5. Timestamped MLflow experiment — prevents history clutter & leakage
"""
SCRIPT_VERSION = "2.3.0-20260510"
SCRIPT_REVISION_NOTE = (
    "v2.3: bf16 on GPU / fp32 on CPU, proper TrialPruned propagation, "
    "native DocumentAwareDataset via Trainer(train_texts), "
    "SSM+Titans combo config, removed fake search params, "
    "torch.compile enabled for all configs via @torch.compiler.disable"
)

import argparse
import csv
import gc
import json
import math
import os
import signal
import sys
import time
import traceback
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import optuna
import torch
from datasets import load_dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer, Trainer

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
FROZEN_TRIAL_MULTIPLIER = 3.0

# ---------------------------------------------------------------------------
# Round configs (overrideable via CLI --max-samples / --epochs)
# ---------------------------------------------------------------------------
ROUNDS: Dict[str, Dict[str, Any]] = {
    "screening": {
        "max_samples": 5000,
        "epochs": 3,
        "n_trials": None,
        "n_parallel": 2,
        "instance_cost_hr": 1.52,
    },
    "validation": {
        "max_samples": 50000,
        "epochs": 5,
        "n_trials": 10,
        "n_parallel": 2,
        "instance_cost_hr": 1.52,
    },
    "final": {
        "max_samples": None,
        "epochs": 10,
        "n_trials": 3,
        "n_parallel": 1,
        "instance_cost_hr": 32.00,
    },
}

# ---------------------------------------------------------------------------
# VIABLE CONFIG TABLE — thought-experiment screened
# ---------------------------------------------------------------------------
VIABLE_CONFIGS: List[Dict[str, Any]] = [
    # ---- T4-friendly (<= 16 GB) ----
    {
        "name": "xs_linear_128",
        "d_model": 128, "n_columns": 2, "n_loops": 1,
        "attention_mode": "linear", "hybrid_full_attention_interval": None,
        "use_ssm": False, "use_titans_memory": False,
        "ffn_expansion": 2.0, "dropout": 0.05,
        "lr": 3e-3, "weight_decay": 0.01,
        "grad_clip": 1.0, "warmup_ratio": 0.05,
        "batch_size": 16, "grad_accum": 2,
    },
    {
        "name": "xs_hybrid_128",
        "d_model": 128, "n_columns": 2, "n_loops": 1,
        "attention_mode": "hybrid", "hybrid_full_attention_interval": 2,
        "use_ssm": False, "use_titans_memory": False,
        "ffn_expansion": 2.0, "dropout": 0.05,
        "lr": 3e-3, "weight_decay": 0.01,
        "grad_clip": 1.0, "warmup_ratio": 0.05,
        "batch_size": 16, "grad_accum": 2,
    },
    {
        "name": "xs_linear_256",
        "d_model": 256, "n_columns": 2, "n_loops": 1,
        "attention_mode": "linear", "hybrid_full_attention_interval": None,
        "use_ssm": False, "use_titans_memory": False,
        "ffn_expansion": 2.0, "dropout": 0.05,
        "lr": 3e-3, "weight_decay": 0.01,
        "grad_clip": 1.0, "warmup_ratio": 0.05,
        "batch_size": 8, "grad_accum": 2,
    },
    {
        "name": "xs_hybrid_256",
        "d_model": 256, "n_columns": 2, "n_loops": 1,
        "attention_mode": "hybrid", "hybrid_full_attention_interval": 2,
        "use_ssm": False, "use_titans_memory": False,
        "ffn_expansion": 2.0, "dropout": 0.05,
        "lr": 3e-3, "weight_decay": 0.01,
        "grad_clip": 1.0, "warmup_ratio": 0.05,
        "batch_size": 8, "grad_accum": 2,
    },
    # ---- L4-friendly (<= 24 GB) ----
    {
        "name": "sm_linear_256",
        "d_model": 256, "n_columns": 3, "n_loops": 2,
        "attention_mode": "linear", "hybrid_full_attention_interval": None,
        "use_ssm": False, "use_titans_memory": False,
        "ffn_expansion": 2.5, "dropout": 0.05,
        "lr": 3e-3, "weight_decay": 0.01,
        "grad_clip": 1.0, "warmup_ratio": 0.05,
        "batch_size": 8, "grad_accum": 2,
    },
    {
        "name": "sm_hybrid_256",
        "d_model": 256, "n_columns": 3, "n_loops": 2,
        "attention_mode": "hybrid", "hybrid_full_attention_interval": 2,
        "use_ssm": False, "use_titans_memory": False,
        "ffn_expansion": 2.5, "dropout": 0.05,
        "lr": 3e-3, "weight_decay": 0.01,
        "grad_clip": 1.0, "warmup_ratio": 0.05,
        "batch_size": 8, "grad_accum": 2,
    },
    {
        "name": "sm_full_256",
        "d_model": 256, "n_columns": 2, "n_loops": 1,
        "attention_mode": "full", "hybrid_full_attention_interval": None,
        "use_ssm": False, "use_titans_memory": False,
        "ffn_expansion": 2.5, "dropout": 0.05,
        "lr": 3e-3, "weight_decay": 0.01,
        "grad_clip": 1.0, "warmup_ratio": 0.05,
        "batch_size": 8, "grad_accum": 2,
    },
    {
        "name": "sm_ssm_256",
        "d_model": 256, "n_columns": 2, "n_loops": 1,
        "attention_mode": "linear", "hybrid_full_attention_interval": None,
        "use_ssm": True, "use_titans_memory": False,
        "ssm_d_state": 64, "ssm_dt_rank": 16, "ssm_d_conv": 3, "ssm_expand": 2,
        "ffn_expansion": 2.0, "dropout": 0.05,
        "lr": 3e-3, "weight_decay": 0.01,
        "grad_clip": 1.0, "warmup_ratio": 0.05,
        "batch_size": 4, "grad_accum": 2,
    },
    {
        "name": "sm_titans_256",
        "d_model": 256, "n_columns": 2, "n_loops": 1,
        "attention_mode": "linear", "hybrid_full_attention_interval": None,
        "use_ssm": False, "use_titans_memory": True,
        "ffn_expansion": 2.0, "dropout": 0.05,
        "lr": 3e-3, "weight_decay": 0.01,
        "grad_clip": 1.0, "warmup_ratio": 0.05,
        "batch_size": 4, "grad_accum": 2,
    },
    {
        "name": "sm_ssm_titans_256",
        "d_model": 256, "n_columns": 2, "n_loops": 1,
        "attention_mode": "linear", "hybrid_full_attention_interval": None,
        "use_ssm": True, "use_titans_memory": True,
        "ssm_d_state": 64, "ssm_dt_rank": 16, "ssm_d_conv": 3, "ssm_expand": 2,
        "ffn_expansion": 2.0, "dropout": 0.05,
        "lr": 3e-3, "weight_decay": 0.01,
        "grad_clip": 1.0, "warmup_ratio": 0.05,
        "batch_size": 4, "grad_accum": 2,
    },
    {
        "name": "sm_linear_384",
        "d_model": 384, "n_columns": 2, "n_loops": 1,
        "attention_mode": "linear", "hybrid_full_attention_interval": None,
        "use_ssm": False, "use_titans_memory": False,
        "ffn_expansion": 2.0, "dropout": 0.05,
        "lr": 3e-3, "weight_decay": 0.01,
        "grad_clip": 1.0, "warmup_ratio": 0.05,
        "batch_size": 4, "grad_accum": 2,
    },
    {
        "name": "sm_hybrid_384",
        "d_model": 384, "n_columns": 2, "n_loops": 1,
        "attention_mode": "hybrid", "hybrid_full_attention_interval": 2,
        "use_ssm": False, "use_titans_memory": False,
        "ffn_expansion": 2.0, "dropout": 0.05,
        "lr": 3e-3, "weight_decay": 0.01,
        "grad_clip": 1.0, "warmup_ratio": 0.05,
        "batch_size": 4, "grad_accum": 2,
    },
]


# ---------------------------------------------------------------------------
# Data loading: native HF streaming -> list of strings -> Trainer
# ---------------------------------------------------------------------------
def load_texts(repo_id: str, split: str, max_samples: Optional[int] = None) -> List[str]:
    print(f"  Loading '{repo_id}' [{split}] (max={max_samples}) ...")
    ds = load_dataset(repo_id, split=split, streaming=True)
    texts = []
    for i, item in enumerate(tqdm(ds, desc=f"  {split}", unit="smpl", leave=False)):
        if max_samples is not None and i >= max_samples:
            break
        texts.append(item.get("text", ""))
    print(f"  -> {len(texts):,} samples")
    return texts


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(dev)
        print(f"  GPU: {props.name} | VRAM: {props.total_memory / 1e9:.1f}GB")
        return dev
    print("  WARNING: No CUDA available, falling back to CPU")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# MLflow safety wrappers
# ---------------------------------------------------------------------------
def safe_log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    clean: Dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and math.isfinite(v):
            clean[k] = float(v)
    if clean:
        try:
            mlflow.log_metrics(clean, step=step)
        except Exception as e:
            print(f"  [MLflow] metrics log failed: {repr(e)}")


def safe_log_param(key: str, value: Any) -> None:
    try:
        s = str(value)
        if len(s) > 500:
            s = s[:497] + "..."
        mlflow.log_param(key, s)
    except Exception as e:
        print(f"  [MLflow] param log failed for {key}: {repr(e)}")


def safe_log_tag(key: str, value: Any) -> None:
    try:
        mlflow.set_tag(key, str(value)[:4000])
    except Exception as e:
        print(f"  [MLflow] tag log failed for {key}: {repr(e)}")


# ---------------------------------------------------------------------------
# VRAM estimator (heuristic but catches obvious OOMs)
# ---------------------------------------------------------------------------
def estimate_vram(params: Dict[str, Any]) -> float:
    d = params["d_model"]
    n_col = params["n_columns"]
    loops = params["n_loops"]
    seq = params["seq_len"]
    batch = params.get("batch_size", 32)
    vocab = 50257

    embed_mb = vocab * d * 4 / (1024 ** 2)

    # Graph params: ~3.5 nodes/col, ~2 d² matrices each
    nodes_per_col = 3.5
    graph_params_mb = n_col * nodes_per_col * 2 * (d ** 2) * 4 / (1024 ** 2)

    # Activations
    base_activations_mb = batch * seq * d * loops * n_col * 4 / (1024 ** 2)

    # Full attention O(n²)
    full_attn_cols = 0
    if params["attention_mode"] == "full":
        full_attn_cols = n_col
    elif params["attention_mode"] == "hybrid":
        interval = params.get("hybrid_full_attention_interval") or 2
        full_attn_cols = (n_col + interval - 1) // interval

    attn_act_mb = 0.0
    if full_attn_cols > 0:
        heads = max(2, d // 64)
        attn_act_mb = full_attn_cols * batch * heads * (seq ** 2) * 4 / (1024 ** 2)

    # SSM autograd chain
    ssm_autograd_mb = 0.0
    if params.get("use_ssm", False):
        ssm_expand = params.get("ssm_expand", 2)
        ssm_d_state = params.get("ssm_d_state", 64)
        d_inner = ssm_expand * d
        ssm_nodes = n_col
        ssm_autograd_mb = ssm_nodes * seq * batch * d_inner * ssm_d_state * 4 / (1024 ** 2)
        graph_params_mb *= 1.3

    # Titans persistent memory M: (batch, feature_dim, d_model)
    titans_mem_mb = 0.0
    if params.get("use_titans_memory", False):
        feature_dim = 64  # TitansMemoryNode default
        titans_mem_mb = batch * feature_dim * d * 4 / (1024 ** 2)
        graph_params_mb *= 1.05

    optimizer_mb = 2 * (embed_mb + graph_params_mb)
    gradients_mb = embed_mb + graph_params_mb

    total_mb = (
        embed_mb + graph_params_mb + base_activations_mb +
        attn_act_mb + ssm_autograd_mb + titans_mem_mb +
        optimizer_mb + gradients_mb + base_activations_mb
    ) * 1.15
    return total_mb


# ---------------------------------------------------------------------------
# Cost estimator
# ---------------------------------------------------------------------------
def estimate_training_cost(
    params: Dict[str, Any],
    dataset_tokens: int,
    epochs: int,
    instance_cost_per_hour: float,
    tok_per_sec_assumed: Optional[float] = None,
) -> Dict[str, Any]:
    effective_batch = params["batch_size"] * params["grad_accum"]
    seq_len = params["seq_len"]
    steps_per_epoch = max(1, dataset_tokens // (effective_batch * seq_len))
    total_steps = steps_per_epoch * epochs

    if tok_per_sec_assumed is None:
        base_tok_per_sec = 25000
        loop_penalty = 1.0 / max(1, params["n_loops"] ** 0.7)
        ssm_penalty = 0.7 if params.get("use_ssm", False) else 1.0
        titans_penalty = 0.9 if params.get("use_titans_memory", False) else 1.0
        if params["attention_mode"] == "full":
            attn_penalty = 0.4
        elif params["attention_mode"] == "hybrid":
            attn_penalty = 0.7
        else:
            attn_penalty = 1.0
        seq_penalty = 256 / seq_len
        tok_per_sec = (
            base_tok_per_sec * loop_penalty * ssm_penalty *
            titans_penalty * attn_penalty * seq_penalty
        )
    else:
        tok_per_sec = tok_per_sec_assumed

    total_tokens = dataset_tokens * epochs
    wall_seconds = total_tokens / max(tok_per_sec, 1)
    wall_hours = wall_seconds / 3600
    cost = wall_hours * instance_cost_per_hour

    return {
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "estimated_tok_per_sec": round(tok_per_sec, 1),
        "wall_seconds": wall_seconds,
        "wall_hours": round(wall_hours, 2),
        "wall_days": round(wall_hours / 24, 2),
        "estimated_cost_usd": round(cost, 2),
        "instance_cost_per_hour": instance_cost_per_hour,
    }


# ---------------------------------------------------------------------------
# Config builder (sacred: align with real HelixConfig fields)
# ---------------------------------------------------------------------------
def build_helix_config(
    params: Dict[str, Any],
    vocab_size: int,
    device: str,
) -> Tuple[Any, bool]:
    # bfloat16 on GPU gives 16-bit speed without fp16 dynamic-range problems.
    # fp32 on CPU is the safe fallback (bfloat16 has limited CPU support).
    if torch.cuda.is_available():
        dtype_str = "bfloat16"
        use_amp = True
    else:
        dtype_str = "float32"
        use_amp = False

    effective_batch = params["batch_size"] * params["grad_accum"]
    steps_per_epoch = max(1, 20000 // effective_batch)
    warmup_steps = max(1, int(steps_per_epoch * params.get("warmup_ratio", 0.05)))

    cfg_kwargs: Dict[str, Any] = dict(
        vocab_size=vocab_size,
        d_model=params["d_model"],
        n_columns=params["n_columns"],
        n_loops=params["n_loops"],
        seq_len=params["seq_len"],
        tokenizer_name="gpt2",
        attention_mode=params["attention_mode"],
        use_ssm=params.get("use_ssm", False),
        use_titans_memory=params.get("use_titans_memory", False),
        use_rope=True,
        ffn_expansion=params["ffn_expansion"],
        dropout=params["dropout"],
        lr=params["lr"],
        weight_decay=params["weight_decay"],
        grad_clip=params["grad_clip"],
        warmup_steps=warmup_steps,
        batch_size=params["batch_size"],
        dtype=dtype_str,
        device=device,
    )

    if params.get("hybrid_full_attention_interval") is not None:
        cfg_kwargs["hybrid_full_attention_interval"] = params["hybrid_full_attention_interval"]

    if params.get("use_ssm", False):
        cfg_kwargs.update(
            ssm_d_state=params.get("ssm_d_state", 64),
            ssm_dt_rank=params.get("ssm_dt_rank", "auto"),
            ssm_d_conv=params.get("ssm_d_conv", 4),
            ssm_expand=params.get("ssm_expand", 2),
        )

    cfg = HelixConfig.tiny(**cfg_kwargs)
    return cfg, use_amp


def params_to_flat_dict(params: Dict[str, Any]) -> Dict[str, str]:
    flat: Dict[str, str] = {}
    for k, v in params.items():
        if isinstance(v, (list, tuple, dict)):
            flat[k] = str(v)
        elif v is None:
            flat[k] = "null"
        else:
            flat[k] = str(v)
    return flat


# ---------------------------------------------------------------------------
# Smoke test (forward+backward on dummy data)
# ---------------------------------------------------------------------------
def _smoke_test_model(model: HelixForCausalLM, device: torch.device, seq_len: int) -> None:
    model.train()
    dummy_len = min(seq_len, 64)
    dummy_ids = torch.zeros((2, dummy_len), dtype=torch.long, device=device)
    dummy_labels = dummy_ids.clone()

    outputs = model(dummy_ids, labels=dummy_labels)
    loss = outputs["loss"] if hasattr(outputs, "loss") else outputs[0]
    if loss is None:
        raise RuntimeError("Model returned None loss on smoke test")
    if torch.isnan(loss) or torch.isinf(loss):
        raise RuntimeError(f"Smoke test produced non-finite loss: {loss.item()}")

    loss.backward()
    model.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# Frozen-trial watchdog
# ---------------------------------------------------------------------------
class FrozenTrialError(Exception):
    pass


def _install_watchdog(timeout_seconds: float):
    timeout_seconds = max(30.0, timeout_seconds)
    if hasattr(signal, 'SIGALRM'):
        def _handler(signum, frame):
            raise FrozenTrialError(
                f"Trial frozen: exceeded {timeout_seconds:.0f}s wall-clock for this epoch."
            )
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(int(timeout_seconds))
        return timeout_seconds
    return None


def _disable_watchdog():
    if hasattr(signal, 'SIGALRM'):
        signal.alarm(0)


# ---------------------------------------------------------------------------
# torch.compile helper
# ---------------------------------------------------------------------------
def try_compile_model(
    model: HelixForCausalLM,
    device: torch.device,
    seq_len: int,
    use_ssm: bool,
    use_titans: bool,
) -> Tuple[HelixForCausalLM, bool, Optional[str]]:
    """Apply torch.compile to the full model.

    Since the custom Python loops in _ssd_chunked_scan and TitansMemoryNode
    are decorated with ``@torch.compiler.disable``, the inductor backend
    now treats them as opaque ops instead of trying to unroll them.  This
    means SSM/Titans configs can also be compiled — the loops run in eager
    mode, everything else is compiled.
    """
    try:
        import torch._inductor.config as inductor_cfg
        inductor_cfg.triton.cudagraphs = False
        compiled = torch.compile(model, mode="default", fullgraph=False)
        _smoke_test_model(compiled, device, seq_len)
        print("  [COMPILE] torch.compile(mode='default', cudagraphs=False) applied and verified")
        return compiled, True, None
    except Exception as e:
        err_msg = repr(e)
        print(f"  [COMPILE] Compiled model failed smoke test: {err_msg}")
        return model, False, err_msg


# ---------------------------------------------------------------------------
# Viable-config sampler
# ---------------------------------------------------------------------------
def sample_viable_config(trial: optuna.Trial, seq_len: int, gpu_mem_gb: Optional[int]) -> Dict[str, Any]:
    if trial.params:
        params = dict(trial.params)
        params["seq_len"] = seq_len
        return params

    eligible = []
    for cfg in VIABLE_CONFIGS:
        c = dict(cfg)
        c["seq_len"] = seq_len
        vram = estimate_vram(c)
        if gpu_mem_gb is None or vram <= gpu_mem_gb * 1024 * 0.90:
            eligible.append((c, vram))

    if not eligible:
        c = dict(VIABLE_CONFIGS[0])
        c["seq_len"] = seq_len
        eligible = [(c, estimate_vram(c))]
        print(f"  [WARN] No configs fit in {gpu_mem_gb}GB; forcing smallest config")

    idx = trial.number % len(eligible)
    chosen, vram = eligible[idx]
    chosen = dict(chosen)
    chosen["seq_len"] = seq_len
    chosen["estimated_vram_mb"] = round(vram, 1)
    return chosen


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial, args: argparse.Namespace, round_cfg: Dict[str, Any]) -> float:
    experiment_name = f"helixlm_nas_{args.round}_{TIMESTAMP}"
    mlflow.set_experiment(experiment_name)

    params = sample_viable_config(trial, args.seq_len, args.gpu_mem)

    tokenizer = HelixTokenizer("gpt2")
    vocab_size = len(tokenizer)
    device = get_device()

    # VRAM pre-check
    est_vram = estimate_vram(params)
    if args.gpu_mem and est_vram > args.gpu_mem * 1024:
        print(f"  [VRAM SKIP] est={est_vram:.0f}MB > {args.gpu_mem*1024:.0f}MB limit")
        safe_log_param("skipped_vram", f"{est_vram:.0f}MB > {args.gpu_mem*1024:.0f}MB")
        return float("inf")

    # Build config
    try:
        cfg, use_amp = build_helix_config(params, vocab_size, str(device))
    except Exception as e:
        warnings.warn(f"Config build failed for trial {trial.number}: {repr(e)}\n{traceback.format_exc()}")
        safe_log_tag("config_build_error", traceback.format_exc())
        return float("inf")

    # Instantiate model
    try:
        model = HelixForCausalLM(cfg).to(device)
        param_count = model.count_parameters()["total"]
    except Exception as e:
        warnings.warn(f"Model instantiation failed for trial {trial.number}: {repr(e)}\n{traceback.format_exc()}")
        safe_log_tag("model_instantiation_error", traceback.format_exc())
        return float("inf")

    # torch.compile attempt (now works for ALL configs)
    compile_applied = False
    compile_error = None
    if args.try_compile:
        model, compile_applied, compile_error = try_compile_model(
            model, device, cfg.seq_len,
            params.get("use_ssm", False),
            params.get("use_titans_memory", False),
        )

    run_name = (
        f"trial_{trial.number:03d}_"
        f"{params.get('name', 'cfg')}_"
        f"seq{params['seq_len']}_d{params['d_model']}"
    )

    with mlflow.start_run(run_name=run_name, log_system_metrics=True):
        for k, v in params_to_flat_dict(params).items():
            safe_log_param(k, v)
        safe_log_param("round", args.round)
        safe_log_param("trial_number", trial.number)
        safe_log_param("param_count", param_count)
        safe_log_param("estimated_vram_mb", round(est_vram, 1))
        safe_log_param("torch_compile", compile_applied)
        if compile_error:
            safe_log_tag("torch_compile_error", compile_error)

        cost_pred = estimate_training_cost(
            params,
            dataset_tokens=5_000_000 if args.round == "screening" else 400_000_000,
            epochs=round_cfg["epochs"],
            instance_cost_per_hour=round_cfg["instance_cost_hr"],
        )
        for k, v in cost_pred.items():
            safe_log_param(f"cost_pred_{k}", v)

        print(f"\n{'='*60}")
        print(f"  TRIAL {trial.number} | {args.round.upper()} | {params.get('name', 'unknown')}")
        print(f"  d_model={params['d_model']}  loops={params['n_loops']}  "
              f"seq={params['seq_len']}  lr={params['lr']}")
        print(f"  attention={params['attention_mode']}  "
              f"ssm={params.get('use_ssm', False)}  titans={params.get('use_titans_memory', False)}")
        print(f"  params={param_count:,}  est_vram={est_vram:.0f}MB  "
              f"batch={params['batch_size']}  accum={params['grad_accum']}")
        print(f"  est_cost=${cost_pred['estimated_cost_usd']}  "
              f"est_wall={cost_pred['wall_days']} days")
        print(f"  torch_compile={compile_applied}")
        print(f"{'='*60}")

        train_max = round_cfg["max_samples"]
        val_max = max(500, train_max // 10) if train_max else 5000

        # Watchdog timeout: per-epoch, min 5 minutes
        expected_epoch_sec = max(60.0, cost_pred["wall_seconds"] / round_cfg["epochs"])
        watchdog_timeout = max(300.0, expected_epoch_sec * FROZEN_TRIAL_MULTIPLIER)

        try:
            train_texts = load_texts(args.dataset_repo, "pretrain_train", train_max)
            val_texts = load_texts(args.dataset_repo, "pretrain_val", val_max) if val_max else None

            trainer = Trainer(
                model=model,
                cfg=cfg,
                train_texts=train_texts,
                val_texts=val_texts,
                tokenizer=tokenizer,
                output_dir=os.path.join(args.output_dir, f"trial_{trial.number:03d}"),
                example_prompts=["The next day", "In 1492,"],
                generated_example_length=30,
                grad_accum_steps=params["grad_accum"],
                use_amp=use_amp,
            )
        except Exception as e:
            safe_log_param("failed", "data_loading")
            safe_log_tag("data_loading_error", traceback.format_exc())
            warnings.warn(f"Data loading failed: {repr(e)}\n{traceback.format_exc()}")
            return float("inf")

        best_val_ppl = float("inf")
        tok_per_sec_list: List[float] = []
        start_time = time.time()
        any_valid_epoch = False

        for epoch in range(1, round_cfg["epochs"] + 1):
            epoch_start = time.time()
            _install_watchdog(watchdog_timeout)

            try:
                train_m = trainer.train_epoch(epoch)
            except FrozenTrialError as fte:
                _disable_watchdog()
                print(f"  [FROZEN] {fte}")
                safe_log_param("failed", "frozen_trial")
                safe_log_tag("frozen_trial_error", str(fte))
                return float("inf")
            except Exception as e:
                _disable_watchdog()
                safe_log_param(f"train_epoch_{epoch}_failed", repr(e))
                safe_log_tag(f"epoch_{epoch}_traceback", traceback.format_exc())
                warnings.warn(f"Train epoch failed: {repr(e)}\n{traceback.format_exc()}")
                return float("inf")
            finally:
                _disable_watchdog()

            epoch_time = time.time() - epoch_start
            tokens_per_epoch = (
                cost_pred["steps_per_epoch"] *
                params["grad_accum"] *
                cfg.batch_size *
                cfg.seq_len
            )
            tok_per_sec = tokens_per_epoch / max(epoch_time, 1e-6)
            tok_per_sec_list.append(tok_per_sec)

            # ----- NaN / all-batches-skipped detection -----
            train_loss = train_m.get("loss", float("inf"))
            train_ppl = train_m.get("perplexity", float("inf"))
            skipped_batches = train_m.get("skipped_batches", 0)

            is_all_skipped = (train_loss == 0.0 and train_ppl == 1.0)

            if is_all_skipped:
                print(f"  [NaN GUARD] Trial {trial.number} epoch {epoch}: "
                      f"ALL batches skipped (loss={train_loss}, ppl={train_ppl}, "
                      f"skipped={skipped_batches}). Killing trial.")
                safe_log_param("failed", f"all_nan_epoch_{epoch}")
                safe_log_param("nan_epoch_skipped_batches", skipped_batches)
                return float("inf")

            if not math.isfinite(train_loss) or train_loss > 50000 or train_ppl > 50000:
                print(f"  [EXPLODE] Trial {trial.number} epoch {epoch} "
                      f"(loss={train_loss:.2f}, ppl={train_ppl:.2f})")
                safe_log_param("failed", f"exploded_epoch_{epoch}")
                safe_log_param("exploded_loss", train_loss)
                return float("inf")

            any_valid_epoch = True

            safe_log_metrics({
                "train_loss": train_loss,
                "train_ppl": train_ppl,
                "tok_per_sec": tok_per_sec,
                "epoch_time_sec": epoch_time,
                "skipped_batches": skipped_batches,
            }, step=epoch)

            # Validation
            val_ppl = float("inf")
            if trainer.val_loader and epoch % max(1, round_cfg["epochs"] // 2) == 0:
                try:
                    val_m = trainer.evaluate()
                    val_loss = val_m.get("loss", float("inf"))
                    val_ppl = val_m.get("perplexity", float("inf"))
                    best_val_ppl = min(best_val_ppl, val_ppl)

                    safe_log_metrics({
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                    }, step=epoch)
                except Exception as e:
                    safe_log_tag("validation_error", traceback.format_exc())
                    warnings.warn(f"Validation failed: {repr(e)}\n{traceback.format_exc()}")

            # Optuna pruning
            report_value = best_val_ppl if math.isfinite(best_val_ppl) else train_ppl
            trial.report(report_value, epoch)
            if trial.should_prune():
                print(f"  [PRUNE] Trial {trial.number} pruned at epoch {epoch}")
                safe_log_param("pruned_at_epoch", epoch)
                raise optuna.TrialPruned()

        wall_time = time.time() - start_time
        avg_tok_per_sec = float(np.mean(tok_per_sec_list)) if tok_per_sec_list else 0.0

        if not any_valid_epoch:
            print(f"  [NaN GUARD] Trial {trial.number}: no epoch produced valid batches.")
            safe_log_param("failed", "all_epochs_zero_batches")
            return float("inf")

        safe_log_metrics({
            "best_val_ppl": best_val_ppl if math.isfinite(best_val_ppl) else 99999.0,
            "avg_tok_per_sec": avg_tok_per_sec,
            "wall_time_sec": wall_time,
        })

        actual_cost = estimate_training_cost(
            params,
            dataset_tokens=5_000_000 if args.round == "screening" else 400_000_000,
            epochs=round_cfg["epochs"],
            instance_cost_per_hour=round_cfg["instance_cost_hr"],
            tok_per_sec_assumed=avg_tok_per_sec,
        )
        for k, v in actual_cost.items():
            safe_log_param(f"actual_cost_{k}", v)

        print(f"  [DONE] Trial {trial.number} | best_val_ppl={best_val_ppl:.2f} | "
              f"avg_tok/s={avg_tok_per_sec:.0f} | wall={wall_time/60:.1f}min")

        del model, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return best_val_ppl if math.isfinite(best_val_ppl) else 99999.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HelixLM NAS with Optuna v2.3")
    parser.add_argument("--round", choices=["screening", "validation", "final"], required=True)
    parser.add_argument("--output-dir", default="./nas_results")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--search-seq-len", action="store_true")
    parser.add_argument("--study-name", default="helixlm_nas")
    parser.add_argument("--dataset-repo", default="david-thrower/HelixLM-small-50.0Mt-91250pt-7143it-20260427")
    parser.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    parser.add_argument("--enqueue-top", type=int, default=None)
    parser.add_argument("--gpu-mem", type=int, default=None,
                        help="GPU memory in GB; filters viable-config table (e.g. 16 for T4, 24 for L4)")
    parser.add_argument("--try-compile", action="store_true",
                        help="Attempt torch.compile on all configs (SSM/Titans loops are opaque to inductor)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Override round-config max_samples (smoke-test friendly)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override round-config epochs")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"\n{'='*70}")
    print(f"  HelixLM NAS  |  Script v{SCRIPT_VERSION}")
    print(f"  {SCRIPT_REVISION_NOTE}")
    print(f"  Timestamp: {TIMESTAMP}")
    print(f"{'='*70}\n")

    args = parse_args()
    round_cfg = dict(ROUNDS[args.round])

    if args.max_samples is not None:
        round_cfg["max_samples"] = args.max_samples
    if args.epochs is not None:
        round_cfg["epochs"] = args.epochs

    if args.round == "screening" and args.n_trials is None:
        n_trials = len(VIABLE_CONFIGS)
    else:
        n_trials = args.n_trials or round_cfg["n_trials"]
    n_jobs = args.n_jobs or round_cfg["n_parallel"]

    os.makedirs(args.output_dir, exist_ok=True)

    # Start MLflow with a file-based backend store if the URI is the default
    # and no server is running (common in CI / sandbox environments).
    if args.mlflow_uri == "http://localhost:5000":
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("localhost", 5000))
            sock.close()
            if result != 0:
                file_store = os.path.join(os.path.abspath(args.output_dir), "mlflow")
                os.makedirs(file_store, exist_ok=True)
                args.mlflow_uri = f"file://{file_store}"
                print(f"  [MLflow] No server on :5000; switching to file store: {args.mlflow_uri}")
        except Exception:
            pass

    mlflow.set_tracking_uri(args.mlflow_uri)
    experiment_name = f"helixlm_nas_{args.round}_{TIMESTAMP}"
    mlflow.set_experiment(experiment_name)

    storage_path = os.path.join(args.output_dir, f"{args.study_name}_{args.round}_{TIMESTAMP}.db")
    storage = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name=f"{args.study_name}_{args.round}_{TIMESTAMP}",
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            multivariate=True,
            n_startup_trials=min(5, n_trials // 4),
            seed=42,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=1,
            interval_steps=1,
        ),
        load_if_exists=True,
    )

    # Enqueue winners from previous round
    if args.round in ("validation", "final"):
        prev_round = "screening" if args.round == "validation" else "validation"
        import glob
        prev_pattern = os.path.join(args.output_dir, f"{args.study_name}_{prev_round}_*.db")
        prev_files = sorted(glob.glob(prev_pattern))
        if prev_files:
            prev_storage = prev_files[-1]
            try:
                prev_fname = os.path.basename(prev_storage).replace(".db", "")
                prev_study = optuna.load_study(
                    study_name=prev_fname,
                    storage=f"sqlite:///{prev_storage}",
                )
                completed = [
                    t for t in prev_study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
                ]
                top_n = args.enqueue_top or round_cfg["n_trials"]
                top_trials = sorted(completed, key=lambda t: t.value)[:top_n]

                for t in top_trials:
                    if t.params:
                        study.enqueue_trial(t.params)
                print(f"Enqueued {len(top_trials)} top configs from {prev_round}")
            except Exception as e:
                print(f"Warning: could not load previous study for enqueuing: {e}")

    print(f"\n{'='*70}")
    print(f" HELIXLM NAS — {args.round.upper()} ROUND")
    print(f" Trials: {n_trials}  |  Parallel: {n_jobs}  |  SeqLen: {args.seq_len}")
    print(f" Dataset: {args.dataset_repo}")
    if args.gpu_mem:
        print(f" GPU memory limit: {args.gpu_mem}GB")
    print(f" torch.compile: {'enabled (all configs)' if args.try_compile else 'disabled'}")
    print(f" Viable configs in table: {len(VIABLE_CONFIGS)}")
    print(f" MLflow experiment: {experiment_name}")
    print(f" Storage: {storage_path}")
    print(f"{'='*70}\n")

    print("Viable config table (filtered by GPU memory):")
    for i, cfg in enumerate(VIABLE_CONFIGS):
        c = dict(cfg)
        c["seq_len"] = args.seq_len
        vram = estimate_vram(c)
        eligible = (args.gpu_mem is None) or (vram <= args.gpu_mem * 1024 * 0.90)
        mark = "✓" if eligible else "✗"
        print(f"  {mark} {i:2d}: {cfg['name']:22s}  d={cfg['d_model']:3d}  "
              f"cols={cfg['n_columns']}  loops={cfg['n_loops']}  "
              f"attn={cfg['attention_mode']:7s}  ssm={str(cfg['use_ssm']):5s}  "
              f"titans={str(cfg.get('use_titans_memory', False)):5s}  "
              f"batch={cfg['batch_size']:2d}  est_vram={vram:6.0f}MB")
    print()

    # Wrapper ensures TrialPruned is NOT swallowed by a broad catch tuple.
    def _objective_wrapper(trial: optuna.Trial) -> float:
        try:
            return objective(trial, args, round_cfg)
        except optuna.TrialPruned:
            raise  # Optuna must see this to record PRUNED correctly
        except Exception as e:
            print(f"[CATCH] Trial {trial.number} failed: {repr(e)}")
            traceback.print_exc()
            return float("inf")

    study.optimize(
        _objective_wrapper,
        n_trials=n_trials,
        n_jobs=n_jobs,
        catch=(),  # handled inside wrapper
        show_progress_bar=True,
    )

    print(f"\n{'='*70}")
    print(f" NAS {args.round.upper()} COMPLETE")
    print(f"{'='*70}")

    if study.best_trial is not None:
        bt = study.best_trial
        print(f"Best trial    : #{bt.number}")
        print(f"Best val PPL  : {bt.value:.2f}")
        print(f"Best params   :")
        for k, v in bt.params.items():
            print(f"  {k:30s} = {v}")
    else:
        print("No successful trials completed.")

    results: List[Dict[str, Any]] = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None:
            results.append({
                "trial": t.number,
                "value": t.value,
                "params": t.params,
                "datetime_start": str(t.datetime_start) if t.datetime_start else None,
                "datetime_complete": str(t.datetime_complete) if t.datetime_complete else None,
                "duration_sec": (t.datetime_complete - t.datetime_start).total_seconds()
                if t.datetime_complete and t.datetime_start else None,
            })

    json_path = os.path.join(args.output_dir, f"nas_{args.round}_{TIMESTAMP}_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "round": args.round,
            "best_trial": study.best_trial.number if study.best_trial else None,
            "best_value": study.best_trial.value if study.best_trial else None,
            "best_params": study.best_trial.params if study.best_trial else None,
            "n_trials_completed": len(results),
            "trials": results,
            "script_version": SCRIPT_VERSION,
            "timestamp": TIMESTAMP,
        }, f, indent=2)
    print(f"\nJSON results : {json_path}")

    csv_path = os.path.join(args.output_dir, f"nas_{args.round}_{TIMESTAMP}_results.csv")
    if results:
        all_keys = sorted(set().union(*(r["params"].keys() for r in results)))
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trial", "val_ppl", "duration_sec"] + all_keys)
            for r in results:
                writer.writerow([
                    r["trial"],
                    r["value"],
                    r.get("duration_sec", ""),
                ] + [r["params"].get(k, "") for k in all_keys])
        print(f"CSV results  : {csv_path}")

    if results and args.search_seq_len:
        print(f"\nCost-ranked summary (seq_len grid):")
        for seq in [128, 256, 512]:
            seq_results = [r for r in results if r["params"].get("seq_len") == seq]
            if seq_results:
                best = min(seq_results, key=lambda r: r["value"])
                print(f"  seq={seq:3d} | best_ppl={best['value']:.2f} | trial={best['trial']}")


if __name__ == "__main__":
    main()
