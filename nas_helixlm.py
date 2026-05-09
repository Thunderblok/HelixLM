#!/usr/bin/env python3
"""
nas_helixlm.py  (place in project root)

Neural Architecture Search for HelixLM using Optuna.
Single-file refactoring of:
  - helix_lm/nas_search_space.py
  - scripts/nas_helixlm.py

3-round escalation protocol:
  Screening   : 5K samples, 3 epochs, ~80 trials, 5 parallel  -> eliminate losers
  Validation  : 50K samples, 5 epochs, ~15 trials, 3 parallel -> confirm top configs
  Final       : Full dataset, 10 epochs, 3 trials, 1 parallel  -> convergence

Usage:
  python nas_helixlm.py --round screening --n-jobs 5 --seq-len 256
  python nas_helixlm.py --round validation --n-jobs 3 --seq-len 256
  python nas_helixlm.py --round final --n-jobs 1 --seq-len 256
"""

import argparse
import csv
import gc
import json
import math
import os
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
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup (script is in root, so add its directory to path)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer, Trainer
from helix_lm.dataset import create_document_loader

# ---------------------------------------------------------------------------
# Round configurations
# ---------------------------------------------------------------------------
ROUNDS: Dict[str, Dict[str, Any]] = {
    "screening": {
        "max_samples": 5000,
        "epochs": 3,
        "n_trials": 80,
        "n_parallel": 5,
        "instance": "g6e.2xlarge",
        "instance_cost_hr": 1.52,
    },
    "validation": {
        "max_samples": 50000,
        "epochs": 5,
        "n_trials": 15,
        "n_parallel": 3,
        "instance": "g6e.2xlarge",
        "instance_cost_hr": 1.52,
    },
    "final": {
        "max_samples": None,
        "epochs": 10,
        "n_trials": 3,
        "n_parallel": 1,
        "instance": "p4d.24xlarge",
        "instance_cost_hr": 32.00,
    },
}

DATASET_REPO = "david-thrower/HelixLM-small-50.0Mt-91250pt-7143it-20260427"

# ---------------------------------------------------------------------------
# Streaming loader (lightweight, no full-RAM materialization)
# ---------------------------------------------------------------------------
class _StreamingTextDataset(IterableDataset):
    def __init__(
        self,
        hf_stream,
        tokenizer,
        seq_len: int,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.hf_stream = hf_stream
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_samples = max_samples

    def __iter__(self):
        count = 0
        buffer: List[int] = []
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0

        for item in self.hf_stream:
            text = item.get("text", "")
            if not text:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            buffer.extend(ids)
            while len(buffer) >= self.seq_len:
                chunk = buffer[: self.seq_len]
                buffer = buffer[self.seq_len :]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                labels = input_ids.clone()
                attention_mask = torch.ones(self.seq_len, dtype=torch.long)
                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask,
                    "is_natural_stop": torch.tensor(False, dtype=torch.bool),
                }
                count += 1
                if self.max_samples is not None and count >= self.max_samples:
                    return

        if 0 < len(buffer) < self.seq_len:
            pad_len = self.seq_len - len(buffer)
            chunk = buffer + [pad_id] * pad_len
            input_ids = torch.tensor(chunk, dtype=torch.long)
            labels = input_ids.clone()
            labels[-pad_len:] = -100
            attention_mask = (input_ids != pad_id).long()
            yield {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
                "is_natural_stop": torch.tensor(True, dtype=torch.bool),
            }


def create_streaming_loader(
    dataset_stream,
    tokenizer,
    seq_len: int,
    batch_size: int,
    max_samples: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    ds = _StreamingTextDataset(dataset_stream, tokenizer, seq_len, max_samples)

    def collate_fn(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        is_natural_stop = torch.stack([b["is_natural_stop"] for b in batch])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "is_natural_stop": is_natural_stop,
        }

    return DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_texts(repo_id: str, split_name: str, max_samples: Optional[int] = None) -> List[str]:
    print(f"  Streaming '{split_name}' ...")
    ds = load_dataset(repo_id, split=split_name, streaming=True)
    texts = []
    for i, item in enumerate(tqdm(ds, desc=f"  {split_name}", unit="smpl", leave=False)):
        if max_samples is not None and i >= max_samples:
            break
        texts.append(item["text"])
    print(f"  -> {len(texts):,} samples loaded")
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
# MLflow safety wrappers (never let a NaN or long string kill the run)
# ---------------------------------------------------------------------------
def safe_log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrics to MLflow, silently dropping NaN/Inf/non-finite values."""
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
    """Log param to MLflow, truncating to avoid length limits."""
    try:
        s = str(value)
        if len(s) > 500:
            s = s[:497] + "..."
        mlflow.log_param(key, s)
    except Exception as e:
        print(f"  [MLflow] param log failed for {key}: {repr(e)}")


def safe_log_tag(key: str, value: Any) -> None:
    """Log tag to MLflow (tags have higher length limits than params)."""
    try:
        mlflow.set_tag(key, str(value)[:4000])
    except Exception as e:
        print(f"  [MLflow] tag log failed for {key}: {repr(e)}")


# ---------------------------------------------------------------------------
# Search Space
# ---------------------------------------------------------------------------
def get_search_space_bounds() -> Dict[str, List[Any]]:
    return {
        "d_model": [128, 256, 384],
        "n_columns": [2, 3, 4],
        "nodes_per_column": ["(2,2)", "(2,3,2)", "(3,4,4,3)", "(3,4,4,4,3)"],
        "n_loops": [1, 2, 3, 4],
        "use_ssm": [False],
        "use_titans": [False],
        "attention_mode": ["linear", "hybrid", "full"],
        "lr": [1e-3, 3e-3, 5e-3, 1e-2],
        "seq_len": [128, 256, 512],
        "ffn_expansion": [2.0, 2.5, 3.0],
        "dropout": [0.0, 0.05, 0.1],
        "weight_decay": [0.0, 0.01, 0.05, 0.1],
        "beta1": [0.9, 0.95],
        "beta2": [0.999, 0.98],
        "adam_eps": [1e-8, 1e-6],
        "grad_clip": [0.5, 1.0, 2.0],
        "warmup_ratio": [0.01, 0.05, 0.1],
        "grad_accum": [1, 2],
    }


def sample_params(trial: optuna.Trial, seq_len: Optional[int] = None) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    # 1. Core Architecture
    params["d_model"] = trial.suggest_categorical("d_model", [128, 256, 384])
    params["n_columns"] = trial.suggest_categorical("n_columns", [2, 3, 4])
    params["nodes_per_column"] = trial.suggest_categorical(
        "nodes_per_column", ["(2,2)", "(2,3,2)", "(3,4,4,3)", "(3,4,4,4,3)"]
    )
    params["n_loops"] = trial.suggest_categorical("n_loops", [1, 2, 3, 4])
    params["ffn_expansion"] = trial.suggest_categorical("ffn_expansion", [2.0, 2.5, 3.0])
    params["dropout"] = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])

    # 2. Attention Topology
    params["attention_mode"] = trial.suggest_categorical("attention_mode", ["linear", "hybrid", "full"])
    if params["attention_mode"] == "hybrid":
        params["hybrid_full_attention_interval"] = trial.suggest_categorical(
            "hybrid_full_attention_interval", [1, 2, 4]
        )
    else:
        params["hybrid_full_attention_interval"] = None

    # 3. Optional Modules — STATIC categorical spaces (never dynamic lists)
    params["use_ssm"] = trial.suggest_categorical("use_ssm", [False, True])
    if params["use_ssm"]:
        params["ssm_d_state"] = trial.suggest_categorical("ssm_d_state", [64, 128])
        params["ssm_dt_rank"] = trial.suggest_categorical("ssm_dt_rank", ["auto", 16, 32])
        params["ssm_d_conv"] = trial.suggest_categorical("ssm_d_conv", [3, 4])
        params["ssm_expand"] = trial.suggest_categorical("ssm_expand", [2, 3])
    else:
        params["ssm_d_state"] = None
        params["ssm_dt_rank"] = None
        params["ssm_d_conv"] = None
        params["ssm_expand"] = None

    params["use_titans"] = trial.suggest_categorical("use_titans", [False, True])
    if params["use_titans"]:
        # FIXED ratio choices — actual dim computed later from d_model * ratio
        params["titans_memory_dim_ratio"] = trial.suggest_categorical(
            "titans_memory_dim_ratio", [0.5, 1.0]
        )
        params["titans_num_memories"] = trial.suggest_categorical("titans_num_memories", [4, 8, 16])
        params["titans_memory_lr"] = trial.suggest_categorical("titans_memory_lr", [1e-4, 3e-4, 1e-3])
    else:
        params["titans_memory_dim_ratio"] = None
        params["titans_num_memories"] = None
        params["titans_memory_lr"] = None

    # 4. AdamW & Training
    params["lr"] = trial.suggest_categorical("lr", [1e-3, 3e-3, 5e-3, 1e-2])
    params["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 0.01, 0.05, 0.1])
    params["beta1"] = trial.suggest_categorical("beta1", [0.9, 0.95])
    params["beta2"] = trial.suggest_categorical("beta2", [0.999, 0.98])
    params["adam_eps"] = trial.suggest_categorical("adam_eps", [1e-8, 1e-6])
    params["grad_clip"] = trial.suggest_categorical("grad_clip", [0.5, 1.0, 2.0])
    params["warmup_ratio"] = trial.suggest_categorical("warmup_ratio", [0.01, 0.05, 0.1])

    # 5. Seq Len
    if seq_len is not None:
        params["seq_len"] = seq_len
    else:
        params["seq_len"] = trial.suggest_categorical("seq_len", [128, 256, 512])

    # 6. Batch / Grad Accum — STATIC categorical (was dynamic based on VRAM)
    params["grad_accum"] = trial.suggest_categorical("grad_accum", [1, 2])
    params["batch_size"] = trial.suggest_categorical("batch_size", [4, 8, 16, 24, 32])

    # 7. Derived flags
    params["force_fp32"] = params["seq_len"] <= 128
    params["effective_batch"] = params["batch_size"] * params["grad_accum"]

    return params


def estimate_vram(params: Dict[str, Any]) -> float:
    d = params["d_model"]
    n_col = params["n_columns"]
    loops = params["n_loops"]
    seq = params["seq_len"]
    batch = params.get("batch_size", 32)
    vocab = 50257

    embed_mb = vocab * d * 4 / (1024 ** 2)
    nodes_tuple = eval(params["nodes_per_column"])
    total_nodes = sum(nodes_tuple)
    avg_nodes_per_col = total_nodes / len(nodes_tuple)
    graph_params_mb = n_col * avg_nodes_per_col * 2 * (d ** 2) * 4 / (1024 ** 2)
    activations_mb = batch * seq * d * loops * n_col * 4 / (1024 ** 2)

    if params["use_ssm"]:
        ssm_state = params.get("ssm_d_state", 64)
        ssm_expand = params.get("ssm_expand", 2)
        graph_params_mb *= 1.3
        activations_mb *= 1.2
        activations_mb += batch * seq * ssm_state * ssm_expand * 4 / (1024 ** 2)

    if params["use_titans"]:
        mem_dim = int(params.get("titans_memory_dim_ratio", 1.0) * d)
        n_mem = params.get("titans_num_memories", 8)
        graph_params_mb *= 1.1
        activations_mb += n_mem * mem_dim * seq * 4 / (1024 ** 2)

    optimizer_mb = 2 * (embed_mb + graph_params_mb)
    gradients_mb = embed_mb + graph_params_mb
    total_mb = (embed_mb + graph_params_mb + activations_mb + optimizer_mb + gradients_mb) * 1.2
    return total_mb


def estimate_training_cost(
    params: Dict[str, Any],
    dataset_tokens: int,
    epochs: int,
    instance_cost_per_hour: float,
    tok_per_sec_assumed: Optional[float] = None,
) -> Dict[str, Any]:
    effective_batch = params["effective_batch"]
    seq_len = params["seq_len"]
    steps_per_epoch = max(1, dataset_tokens // (effective_batch * seq_len))
    total_steps = steps_per_epoch * epochs

    if tok_per_sec_assumed is None:
        base_tok_per_sec = 25000
        loop_penalty = 1.0 / max(1, params["n_loops"] ** 0.7)
        ssm_penalty = 0.7 if params["use_ssm"] else 1.0
        titans_penalty = 0.85 if params["use_titans"] else 1.0
        seq_penalty = 256 / seq_len
        tok_per_sec = base_tok_per_sec * loop_penalty * ssm_penalty * titans_penalty * seq_penalty
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
        "wall_hours": round(wall_hours, 2),
        "wall_days": round(wall_hours / 24, 2),
        "estimated_cost_usd": round(cost, 2),
        "instance_cost_per_hour": instance_cost_per_hour,
    }


def assign_node_type(
    col_idx: int,
    node_idx: int,
    n_nodes_in_col: int,
    attention_mode: str,
    use_ssm: bool,
    n_columns: int,
    use_titans: bool,
    hybrid_interval: Optional[int] = None,
) -> str:
    if node_idx == n_nodes_in_col - 1 and (col_idx == 0 or col_idx == n_columns - 1):
        return "gate"

    if use_titans and node_idx == 0 and col_idx == n_columns // 2:
        return "titans_memory"

    if attention_mode == "full":
        if node_idx % 2 == 0:
            return "full_attn"
        else:
            return "swiglu"

    elif attention_mode == "hybrid":
        interval = hybrid_interval or 2
        if (col_idx + node_idx) % interval == 0:
            return "full_attn"
        else:
            return "linear_attn"
    else:
        if node_idx % 3 == 0 and use_ssm:
            return "mamba2"
        elif node_idx % 3 == 1:
            return "linear_attn"
        else:
            return "swiglu"

    return "swiglu"


def build_node_config(node_type: str, params: Dict[str, Any], col_idx: int, node_idx: int) -> Dict[str, Any]:
    d = params["d_model"]
    n_col = params["n_columns"]
    ffn = params["ffn_expansion"]

    if node_type == "linear_attn":
        feature_dim = d // 2 if params["n_loops"] >= 3 else d
        return {
            "type": "LinearAttnNode",
            "num_heads": max(2, d // 64),
            "feature_dim": feature_dim,
            "dropout": params["dropout"],
            "use_rope": True,
            "feature_map": "elu",
        }

    elif node_type == "full_attn":
        heads = max(4, d // 32)
        if n_col >= 4:
            heads = max(2, heads // 2)
        return {
            "type": "FullAttnNode",
            "num_heads": heads,
            "attn_dropout": params["dropout"],
            "use_rope": True,
            "causal": True,
        }

    elif node_type == "swiglu":
        col_depth = eval(params["nodes_per_column"])[col_idx]
        adjusted_ffn = ffn * (1.0 - 0.05 * (col_depth - 2))
        adjusted_ffn = max(1.5, adjusted_ffn)
        return {
            "type": "SwiGLUNode",
            "expansion_factor": round(adjusted_ffn, 2),
            "dropout": params["dropout"],
            "activation": "silu",
        }

    elif node_type == "mamba2":
        d_state = params.get("ssm_d_state", 64)
        d_state = min(d_state, d // 2)
        return {
            "type": "Mamba2Node",
            "d_state": d_state,
            "d_conv": params.get("ssm_d_conv", 4),
            "dt_rank": params.get("ssm_dt_rank", "auto"),
            "expand": params.get("ssm_expand", 2),
            "use_mem_eff_path": True,
        }

    elif node_type == "gate":
        return {
            "type": "GateNode",
            "aggregation": "learned_softmax",
            "max_inputs": 4,
            "dropout": 0.0,
        }

    elif node_type == "titans_memory":
        return {
            "type": "TitansMemoryNode",
            "memory_dim": params.get("titans_memory_dim", d),
            "num_memories": params.get("titans_num_memories", 8),
            "memory_lr": params.get("titans_memory_lr", 3e-4),
        }

    else:
        return {"type": "SwiGLUNode", "expansion_factor": ffn}


def build_helix_config(
    params: Dict[str, Any],
    vocab_size: int,
    device: str,
    tokenizer_name: str = "gpt2",
) -> Any:
    from helix_lm import HelixConfig

    # Compute derived Titans dims BEFORE building node configs
    if params.get("use_titans") and params.get("titans_memory_dim_ratio") is not None:
        params["titans_memory_dim"] = int(params["d_model"] * params["titans_memory_dim_ratio"])
    else:
        params["titans_memory_dim"] = None

    nodes_per_column: Tuple[int, ...] = eval(params["nodes_per_column"])

    column_specs: List[List[Dict[str, Any]]] = []
    for col_idx, n_nodes in enumerate(nodes_per_column):
        nodes: List[Dict[str, Any]] = []
        for node_idx in range(n_nodes):
            node_type = assign_node_type(
                col_idx=col_idx,
                node_idx=node_idx,
                n_nodes_in_col=n_nodes,
                attention_mode=params["attention_mode"],
                use_ssm=params["use_ssm"],
                n_columns=params["n_columns"],
                use_titans=params["use_titans"],
                hybrid_interval=params.get("hybrid_full_attention_interval"),
            )
            node_cfg = build_node_config(node_type, params, col_idx, node_idx)
            nodes.append(node_cfg)
        column_specs.append(nodes)

    if params["seq_len"] <= 128:
        dtype_str = "float32"
        use_amp = False
    else:
        dtype_str = "float16"
        use_amp = True

    effective_batch = params["batch_size"] * params["grad_accum"]
    steps_per_epoch = max(1, 20000 // effective_batch)
    warmup_steps = max(1, int(steps_per_epoch * params["warmup_ratio"]))

    # Pass nodes_per_column as tuple (HelixConfig validation does tuple arithmetic)
    cfg = HelixConfig.tiny(
        vocab_size=vocab_size,
        d_model=params["d_model"],
        n_columns=params["n_columns"],
        nodes_per_column=nodes_per_column,
        n_loops=params["n_loops"],
        seq_len=params["seq_len"],
        tokenizer_name=tokenizer_name,
        attention_mode=params["attention_mode"],
        hybrid_full_attention_interval=params.get("hybrid_full_attention_interval", 2),
        use_ssm=params["use_ssm"],
        use_titans_memory=params["use_titans"],
        use_rope=True,
        ffn_expansion=params["ffn_expansion"],
        dropout=params["dropout"],
        lr=params["lr"],
        weight_decay=params["weight_decay"],
        grad_clip=params["grad_clip"],
        warmup_steps=warmup_steps,
        batch_size=params["batch_size"],
        grad_accum_steps=params["grad_accum"],
        use_amp=use_amp,
        dtype=dtype_str,
        device=device,
        adam_beta1=params["beta1"],
        adam_beta2=params["beta2"],
        adam_eps=params["adam_eps"],
        ssm_d_state=params.get("ssm_d_state"),
        ssm_dt_rank=params.get("ssm_dt_rank"),
        ssm_d_conv=params.get("ssm_d_conv"),
        ssm_expand=params.get("ssm_expand"),
        titans_memory_dim=params.get("titans_memory_dim"),
        titans_num_memories=params.get("titans_num_memories"),
        titans_memory_lr=params.get("titans_memory_lr"),
        column_specs=column_specs,
    )

    return cfg


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
# Fail-fast model validation (never load data for a broken model)
# ---------------------------------------------------------------------------
def _smoke_test_model(model: HelixForCausalLM, device: torch.device, seq_len: int) -> None:
    """
    Run one forward+backward pass on dummy data.
    Raises on failure so caller can bail before wasting IO/time on data loading.
    """
    model.train()
    dummy_len = min(seq_len, 64)
    dummy_ids = torch.zeros((2, dummy_len), dtype=torch.long, device=device)
    dummy_labels = dummy_ids.clone()

    outputs = model(dummy_ids, labels=dummy_labels)
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    if loss is None:
        raise RuntimeError("Model returned None loss on smoke test")

    if torch.isnan(loss) or torch.isinf(loss):
        raise RuntimeError(f"Smoke test produced non-finite loss: {loss.item()}")

    loss.backward()
    model.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial, args: argparse.Namespace, round_cfg: Dict[str, Any]) -> float:
    mlflow.set_experiment(f"helixlm_nas_{args.round}-0002")

    if args.search_seq_len:
        seq_len = trial.suggest_categorical("seq_len", [128, 256, 512])
    else:
        seq_len = args.seq_len

    params = sample_params(trial, seq_len=seq_len)

    tokenizer = HelixTokenizer("gpt2")
    vocab_size = len(tokenizer)
    device = get_device()

    try:
        cfg = build_helix_config(params, vocab_size, str(device))
    except Exception as e:
        warnings.warn(f"Config build failed for trial {trial.number}: {repr(e)}\n{traceback.format_exc()}")
        return float("inf")

    try:
        model = HelixForCausalLM(cfg).to(device)
        param_count = model.count_parameters()["total"]
    except Exception as e:
        warnings.warn(f"Model instantiation failed for trial {trial.number}: {repr(e)}\n{traceback.format_exc()}")
        return float("inf")

    # -----------------------------------------------------------------------
    # torch.compile as requested, with full forward+backward smoke test.
    # If compiled model fails, fallback to uncompiled. If that fails, bail.
    # -----------------------------------------------------------------------
    compile_applied = False
    # try:
    #     # model = torch.compile(model, mode="reduce-overhead")
    #    _smoke_test_model(model, device, cfg.seq_len)
    #     compile_applied = True
    #     print(f"  [COMPILE] torch.compile(mode='reduce-overhead') applied and verified")
    # except Exception as e:
    #    print(f"  [COMPILE] Compiled model failed smoke test: {repr(e)}")
    #    del model
    #    gc.collect()
    #    if torch.cuda.is_available():
    #        torch.cuda.empty_cache()
    #    try:
    #        model = HelixForCausalLM(cfg).to(device)
    #        param_count = model.count_parameters()["total"]
    #        _smoke_test_model(model, device, cfg.seq_len)
    #        print(f"  [COMPILE] Fallback to uncompiled model verified")
    #    except Exception as e2:
    #        print(f"  [FAIL FAST] Uncompiled model also failed smoke test: {repr(e2)}")
    #        safe_log_param("model_smoke_test_failed", repr(e2))
    #        safe_log_tag("traceback", traceback.format_exc())
    #        return float("inf")

    run_name = f"trial_{trial.number:03d}_seq{params['seq_len']}_d{params['d_model']}"
    with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
        # Log all params up front so the run is never empty
        for k, v in params_to_flat_dict(params).items():
            safe_log_param(k, v)
        safe_log_param("round", args.round)
        safe_log_param("trial_number", trial.number)
        safe_log_param("param_count", param_count)
        safe_log_param("estimated_vram_mb", round(estimate_vram(params), 1))
        safe_log_param("torch_compile", compile_applied)

        cost_pred = estimate_training_cost(
            params,
            dataset_tokens=5_000_000 if args.round == "screening" else 400_000_000,
            epochs=round_cfg["epochs"],
            instance_cost_per_hour=round_cfg["instance_cost_hr"],
        )
        for k, v in cost_pred.items():
            safe_log_param(f"cost_pred_{k}", v)

        print(f"\n{'='*60}")
        print(f"  TRIAL {trial.number} | {args.round.upper()}")
        print(f"  d_model={params['d_model']}  loops={params['n_loops']}  "
              f"seq={params['seq_len']}  lr={params['lr']}")
        print(f"  attention={params['attention_mode']}  "
              f"ssm={params['use_ssm']}  titans={params['use_titans']}")
        print(f"  adam: beta1={params['beta1']}  beta2={params['beta2']}  "
              f"wd={params['weight_decay']}  eps={params['adam_eps']}")
        print(f"  params={param_count:,}  est_vram={estimate_vram(params):.0f}MB  "
              f"batch={params['batch_size']}  accum={params['grad_accum']}")
        print(f"  est_cost=${cost_pred['estimated_cost_usd']}  "
              f"est_wall={cost_pred['wall_days']} days")
        print(f"{'='*60}")

        train_max = round_cfg["max_samples"]
        val_max = max(500, train_max // 10) if train_max else 5000

        try:
            if train_max is None:
                print("  [DATA] Using streaming loader (full dataset, no RAM materialization)")
                train_stream = load_dataset(args.dataset_repo, split="pretrain_train", streaming=True)
                val_stream = load_dataset(args.dataset_repo, split="pretrain_val", streaming=True)

                train_loader = create_streaming_loader(
                    train_stream, tokenizer, cfg.seq_len,
                    batch_size=cfg.batch_size,
                    max_samples=None,
                    num_workers=2,
                    pin_memory=True,
                )
                val_loader = create_streaming_loader(
                    val_stream, tokenizer, cfg.seq_len,
                    batch_size=cfg.batch_size,
                    max_samples=val_max,
                    num_workers=2,
                    pin_memory=True,
                )

                trainer = Trainer(
                    model=model,
                    cfg=cfg,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    tokenizer=tokenizer,
                    output_dir=os.path.join(args.output_dir, f"trial_{trial.number:03d}"),
                    example_prompts=["The next day", "In 1492,"],
                    generated_example_length=30,
                    grad_accum_steps=params["grad_accum"],
                    use_amp=params["seq_len"] > 128,
                    min_tail_len=1,
                    num_workers=0,
                    pin_memory=True,
                )
            else:
                train_texts = load_texts(args.dataset_repo, "pretrain_train", train_max)
                val_texts = load_texts(args.dataset_repo, "pretrain_val", val_max)

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
                    use_amp=params["seq_len"] > 128,
                    min_tail_len=1,
                )
        except Exception as e:
            safe_log_param("failed", "data_loading")
            safe_log_tag("data_loading_error", traceback.format_exc())
            warnings.warn(f"Data loading failed: {repr(e)}\n{traceback.format_exc()}")
            return float("inf")

        best_val_ppl = float("inf")
        tok_per_sec_list: List[float] = []
        start_time = time.time()

        for epoch in range(1, round_cfg["epochs"] + 1):
            epoch_start = time.time()

            try:
                train_m = trainer.train_epoch(epoch)
            except Exception as e:
                safe_log_param(f"train_epoch_{epoch}_failed", repr(e))
                safe_log_tag(f"epoch_{epoch}_traceback", traceback.format_exc())
                warnings.warn(f"Train epoch failed: {repr(e)}\n{traceback.format_exc()}")
                return float("inf")

            epoch_time = time.time() - epoch_start
            # Trainer doesn't return tok_per_sec; estimate from tokens / time
            # We know batch_size, seq_len, and number of batches (approx)
            n_batches = max(1, len(trainer.train_loader))
            tokens_per_epoch = n_batches * cfg.batch_size * cfg.seq_len
            tok_per_sec = tokens_per_epoch / max(epoch_time, 1e-6)
            tok_per_sec_list.append(tok_per_sec)

            train_loss = train_m.get("loss", float("inf"))
            train_ppl = train_m.get("perplexity", float("inf"))
            if not math.isfinite(train_loss) or train_loss > 50000 or train_ppl > 50000:
                print(f"  [FAIL] Trial {trial.number} exploded at epoch {epoch} "
                      f"(loss={train_loss:.2f}, ppl={train_ppl:.2f})")
                safe_log_param("failed", f"exploded_epoch_{epoch}")
                safe_log_param("exploded_loss", train_loss)
                return float("inf")

            # Log epoch metrics immediately
            safe_log_metrics({
                "train_loss": train_loss,
                "train_ppl": train_ppl,
                "tok_per_sec": tok_per_sec,
                "epoch_time_sec": epoch_time,
                "skipped_batches": train_m.get("skipped_batches", 0),
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

            report_value = best_val_ppl if math.isfinite(best_val_ppl) else train_ppl
            trial.report(report_value, epoch)
            if trial.should_prune():
                print(f"  [PRUNE] Trial {trial.number} pruned at epoch {epoch}")
                safe_log_param("pruned_at_epoch", epoch)
                raise optuna.TrialPruned()

        wall_time = time.time() - start_time
        avg_tok_per_sec = float(np.mean(tok_per_sec_list)) if tok_per_sec_list else 0.0

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
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HelixLM NAS with Optuna")
    parser.add_argument("--round", choices=["screening", "validation", "final"], required=True)
    parser.add_argument("--output-dir", default="./nas_results")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--search-seq-len", action="store_true")
    parser.add_argument("--study-name", default="helixlm_nas")
    parser.add_argument("--dataset-repo", default=DATASET_REPO)
    parser.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    parser.add_argument("--enqueue-top", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    round_cfg = ROUNDS[args.round]
    n_trials = args.n_trials or round_cfg["n_trials"]
    n_jobs = args.n_jobs or round_cfg["n_parallel"]

    os.makedirs(args.output_dir, exist_ok=True)

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(f"helixlm_nas_{args.round}")

    storage_path = os.path.join(args.output_dir, f"{args.study_name}_{args.round}.db")
    storage = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name=f"{args.study_name}_{args.round}",
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            multivariate=True,
            n_startup_trials=min(10, n_trials // 4),
            seed=42,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=1,
            interval_steps=1,
        ),
        load_if_exists=True,
    )

    if args.round in ("validation", "final"):
        prev_round = "screening" if args.round == "validation" else "validation"
        prev_storage = os.path.join(args.output_dir, f"{args.study_name}_{prev_round}.db")
        if os.path.exists(prev_storage):
            try:
                prev_study = optuna.load_study(
                    study_name=f"{args.study_name}_{prev_round}",
                    storage=f"sqlite:///{prev_storage}",
                )
                completed = [t for t in prev_study.trials
                             if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
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
    print(f" Trials: {n_trials}  |  Parallel: {n_jobs}  |  SeqLen: {'searched' if args.search_seq_len else args.seq_len}")
    print(f" Dataset: {args.dataset_repo}")
    print(f" torch.compile: reduce-overhead (with full smoke-test fallback)")
    print(f" Storage: {storage_path}")
    print(f"{'='*70}\n")

    study.optimize(
        lambda trial: objective(trial, args, round_cfg),
        n_trials=n_trials,
        n_jobs=n_jobs,
        catch=(Exception,),
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

    json_path = os.path.join(args.output_dir, f"nas_{args.round}_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "round": args.round,
            "best_trial": study.best_trial.number if study.best_trial else None,
            "best_value": study.best_trial.value if study.best_trial else None,
            "best_params": study.best_trial.params if study.best_trial else None,
            "n_trials_completed": len(results),
            "trials": results,
        }, f, indent=2)
    print(f"\nJSON results : {json_path}")

    csv_path = os.path.join(args.output_dir, f"nas_{args.round}_results.csv")
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
