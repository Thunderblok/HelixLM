#!/usr/bin/env python3
"""
nas_400m_poc.py — Neural Architecture Search for HelixLM @ 400M tokens
======================================================================
Proof-of-concept on l40sx1 (48GB VRAM) with BF16, NO torch.compile.

Sacred constraints:
  - graph.py is NOT modified (the bug fix stays)
  - model architecture is fixed: d_model=384, n_columns=2, nodes=(2,2)
  - only hyperparameters and training dynamics are searched
  
Goal: find hyperparams that make the fixed-graph model train well on 400M tokens.
"""
SCRIPT_VERSION = "1.0.0-20260512"

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

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer, Trainer

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Search space: hyperparameters only — architecture is sacred
# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    # Learning rate: the old 3e-3 may be too aggressive with all params active
    "lr": [1e-3, 2e-3, 3e-3, 5e-4],
    
    # Weight decay: old was 0.01, new branch tried 0.1 — need to find sweet spot
    "weight_decay": [0.01, 0.05, 0.1, 0.15],
    
    # Warmup ratio: old was 2000 steps fixed, new branch uses dynamic
    "warmup_ratio": [0.05, 0.1, 0.15, 0.2],
    
    # Batch size / grad accum combinations for effective batch
    "batch_size": [8, 16, 32],
    "grad_accum": [1, 2, 4],
    
    # Gradient clipping
    "grad_clip": [0.5, 1.0, 2.0],
    
    # Dropout: may need more regularization with full model
    "dropout": [0.0, 0.05, 0.1],
    
    # Sequence length
    "seq_len": [256, 512],
}

# Fixed architecture for all trials
ARCH_CONFIG = {
    "d_model": 384,
    "n_columns": 2,
    "n_loops": 1,
    "nodes_per_column": (2, 2),
    "attention_mode": "hybrid",
    "hybrid_full_attention_interval": 2,
    "use_ssm": False,
    "use_titans_memory": False,
    "ffn_expansion": 2.0,
}

# ---------------------------------------------------------------------------
# Data loading
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
# Config builder
# ---------------------------------------------------------------------------
def build_helix_config(params: Dict[str, Any], vocab_size: int, device: str, epochs: int, tokenizer) -> Tuple[Any, bool]:
    if torch.cuda.is_available():
        dtype_str = "bfloat16"
        use_amp = True
    else:
        dtype_str = "float32"
        use_amp = False

    effective_batch = params["batch_size"] * params["grad_accum"]
    # 400M tokens, estimate steps
    total_tokens = 400_000_000
    steps_per_epoch = max(1, total_tokens // (effective_batch * params["seq_len"]))
    warmup_steps = max(1, int(steps_per_epoch * params["warmup_ratio"]))

    cfg_kwargs: Dict[str, Any] = dict(
        vocab_size=vocab_size,
        d_model=ARCH_CONFIG["d_model"],
        n_columns=ARCH_CONFIG["n_columns"],
        n_loops=ARCH_CONFIG["n_loops"],
        nodes_per_column=ARCH_CONFIG["nodes_per_column"],
        seq_len=int(params["seq_len"]),
        tokenizer_name="gpt2",
        attention_mode=ARCH_CONFIG["attention_mode"],
        hybrid_full_attention_interval=ARCH_CONFIG["hybrid_full_attention_interval"],
        use_ssm=ARCH_CONFIG["use_ssm"],
        use_titans_memory=ARCH_CONFIG["use_titans_memory"],
        use_rope=True,
        ffn_expansion=ARCH_CONFIG["ffn_expansion"],
        dropout=float(params["dropout"]),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
        grad_clip=float(params["grad_clip"]),
        warmup_steps=int(warmup_steps),
        epochs=int(epochs),
        batch_size=int(params["batch_size"]),
        dtype=dtype_str,
        device=device,
    )

    cfg = HelixConfig(**cfg_kwargs)
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.eos_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        cfg.bos_token_id = tokenizer.bos_token_id
    return cfg, use_amp


# ---------------------------------------------------------------------------
# Simple grid search (no Optuna needed for hyperparam-only search)
# ---------------------------------------------------------------------------
def generate_grid_trials(n_trials: int, seed: int = 42) -> List[Dict[str, Any]]:
    rng = np.random.RandomState(seed)
    trials = []
    
    for i in range(n_trials):
        trial = {}
        trial["lr"] = rng.choice(SEARCH_SPACE["lr"])
        trial["weight_decay"] = rng.choice(SEARCH_SPACE["weight_decay"])
        trial["warmup_ratio"] = rng.choice(SEARCH_SPACE["warmup_ratio"])
        trial["batch_size"] = rng.choice(SEARCH_SPACE["batch_size"])
        trial["grad_accum"] = rng.choice(SEARCH_SPACE["grad_accum"])
        trial["grad_clip"] = rng.choice(SEARCH_SPACE["grad_clip"])
        trial["dropout"] = rng.choice(SEARCH_SPACE["dropout"])
        trial["seq_len"] = rng.choice(SEARCH_SPACE["seq_len"])
        trials.append(trial)
    
    return trials


# ---------------------------------------------------------------------------
# Single trial runner
# ---------------------------------------------------------------------------
def run_trial(trial_num: int, params: Dict[str, Any], args, epochs: int, max_samples: int) -> Dict[str, Any]:
    model = None
    trainer = None
    result = {"trial": trial_num, "params": params, "status": "failed", "best_val_ppl": float("inf")}
    
    try:
        tokenizer = HelixTokenizer("gpt2")
        vocab_size = len(tokenizer)
        device = get_device()
        
        # Build config
        cfg, use_amp = build_helix_config(params, vocab_size, str(device), epochs, tokenizer)
        
        # Instantiate model
        model = HelixForCausalLM(cfg).to(device)
        param_count = model.count_parameters()["total"]
        
        print(f"\n{'='*60}")
        print(f"  TRIAL {trial_num}")
        print(f"  lr={params['lr']:.0e}  wd={params['weight_decay']}  warmup={params['warmup_ratio']}")
        print(f"  batch={params['batch_size']}  accum={params['grad_accum']}  seq={params['seq_len']}")
        print(f"  dropout={params['dropout']}  grad_clip={params['grad_clip']}")
        print(f"  params={param_count:,}  device={device}")
        print(f"{'='*60}")
        
        # Load data
        train_texts = load_texts(args.dataset_repo, "pretrain_train", max_samples)
        val_max = max(500, max_samples // 10) if max_samples else 5000
        val_texts = load_texts(args.dataset_repo, "pretrain_val", val_max) if val_max else None
        
        # Trainer
        trainer = Trainer(
            model=model,
            cfg=cfg,
            train_texts=train_texts,
            val_texts=val_texts,
            tokenizer=tokenizer,
            output_dir=os.path.join(args.output_dir, f"trial_{trial_num:03d}"),
            example_prompts=["The next day", "In 1492,"],
            generated_example_length=30,
            grad_accum_steps=params["grad_accum"],
            use_amp=use_amp,
            min_tail_len=params["seq_len"] // 4,
        )
        
        best_val_ppl = float("inf")
        any_valid_epoch = False
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            try:
                train_m = trainer.train_epoch(epoch)
            except Exception as e:
                print(f"  [ERROR] Train epoch {epoch} failed: {e}")
                break
            
            train_loss = train_m.get("loss", float("inf"))
            train_ppl = train_m.get("perplexity", float("inf"))
            skipped_batches = train_m.get("skipped_batches", 0)
            
            # NaN guard
            if train_loss == 0.0 and train_ppl == 1.0:
                print(f"  [NaN GUARD] All batches skipped")
                break
            
            # EXPLODE guard — with fixed graph, PPL starts high but should decrease
            # Only explode if PPL is INCREASING significantly or hits NaN
            if not math.isfinite(train_loss) or train_ppl > 1000000:
                print(f"  [EXPLODE] loss={train_loss:.2f}, ppl={train_ppl:.2f}")
                break
            
            any_valid_epoch = True
            
            # Validation every epoch
            val_ppl = float("inf")
            if trainer.val_loader:
                try:
                    val_m = trainer.evaluate()
                    val_loss = val_m.get("loss", float("inf"))
                    val_ppl = val_m.get("perplexity", float("inf"))
                    best_val_ppl = min(best_val_ppl, val_ppl)
                    print(f"  Epoch {epoch}: train_ppl={train_ppl:.2f}  val_ppl={val_ppl:.2f}")
                except Exception as e:
                    print(f"  [WARN] Validation failed: {e}")
            else:
                print(f"  Epoch {epoch}: train_ppl={train_ppl:.2f}")
        
        wall_time = time.time() - start_time
        
        result["status"] = "complete" if any_valid_epoch else "failed"
        result["best_val_ppl"] = best_val_ppl if math.isfinite(best_val_ppl) else 99999.0
        result["wall_time"] = wall_time
        result["param_count"] = param_count
        
        print(f"  [DONE] Trial {trial_num} | best_val_ppl={result['best_val_ppl']:.2f} | wall={wall_time/60:.1f}min")
        
    except Exception as e:
        print(f"  [FATAL] Trial {trial_num} failed: {e}")
        traceback.print_exc()
        result["error"] = str(e)
    finally:
        if trainer is not None:
            del trainer
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HelixLM NAS @ 400M tokens POC")
    parser.add_argument("--output-dir", default="./nas_400m_results")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per trial (POC: 1 epoch)")
    parser.add_argument("--max-samples", type=int, default=50000, help="Samples per trial (POC subset)")
    parser.add_argument("--dataset-repo", default="david-thrower/HelixLM-tiny-400.0Mt-730000pt-57143it-20260430")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    print(f"\n{'='*70}")
    print(f"  HelixLM NAS 400M POC  |  Script v{SCRIPT_VERSION}")
    print(f"  Timestamp: {TIMESTAMP}")
    print(f"  ARCHITECTURE IS SACRED — only hyperparameters searched")
    print(f"  Fixed: d_model=384, n_columns=2, nodes=(2,2), hybrid attn")
    print(f"{'='*70}\n")
    
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Generate trials
    trials = generate_grid_trials(args.n_trials, seed=args.seed)
    
    print(f"Running {len(trials)} trials x {args.epochs} epoch(s) each")
    print(f"Dataset: {args.dataset_repo} (max_samples={args.max_samples})")
    print(f"Output: {args.output_dir}\n")
    
    results = []
    for i, params in enumerate(trials):
        result = run_trial(i, params, args, args.epochs, args.max_samples)
        results.append(result)
        
        # Save incremental results
        json_path = os.path.join(args.output_dir, f"nas_400m_{TIMESTAMP}_results.json")
        with open(json_path, "w") as f:
            json.dump({
                "script_version": SCRIPT_VERSION,
                "timestamp": TIMESTAMP,
                "n_trials": len(trials),
                "epochs_per_trial": args.epochs,
                "max_samples": args.max_samples,
                "dataset_repo": args.dataset_repo,
                "arch_config": ARCH_CONFIG,
                "search_space": {k: v for k, v in SEARCH_SPACE.items()},
                "results": results,
            }, f, indent=2, default=str)
        print(f"  Saved: {json_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f" NAS 400M POC COMPLETE")
    print(f"{'='*70}")
    
    complete = [r for r in results if r["status"] == "complete"]
    if complete:
        best = min(complete, key=lambda r: r["best_val_ppl"])
        print(f"\nBEST TRIAL: #{best['trial']}")
        print(f"  Val PPL: {best['best_val_ppl']:.2f}")
        print(f"  Params: {best['params']}")
        
        # Write best config for production
        best_cfg_path = os.path.join(args.output_dir, f"best_config_{TIMESTAMP}.json")
        with open(best_cfg_path, "w") as f:
            json.dump({
                "arch_config": ARCH_CONFIG,
                "hyperparams": best["params"],
                "best_val_ppl": best["best_val_ppl"],
                "production_command": build_production_command(best["params"]),
            }, f, indent=2)
        print(f"\nBest config saved: {best_cfg_path}")
    else:
        print("No successful trials completed.")
    
    # CSV
    csv_path = os.path.join(args.output_dir, f"nas_400m_{TIMESTAMP}_results.csv")
    if complete:
        keys = sorted(set().union(*(r["params"].keys() for r in complete)))
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trial", "val_ppl", "wall_time", "param_count"] + keys)
            for r in complete:
                writer.writerow([
                    r["trial"],
                    r["best_val_ppl"],
                    r.get("wall_time", ""),
                    r.get("param_count", ""),
                ] + [r["params"].get(k, "") for k in keys])
        print(f"CSV results: {csv_path}")


def build_production_command(params: Dict[str, Any]) -> str:
    """Build the production training command from best hyperparams."""
    nodes_str = ",".join(map(str, ARCH_CONFIG["nodes_per_column"]))
    cmd = (
        f"python train_helixlm_production.py \\\n"
        f"  --stage base \\\n"
        f"  --dataset-repo david-thrower/HelixLM-tiny-400.0Mt-730000pt-57143it-20260430 \\\n"
        f"  --d-model {ARCH_CONFIG['d_model']} \\\n"
        f"  --n-columns {ARCH_CONFIG['n_columns']} \\\n"
        f"  --n-loops {ARCH_CONFIG['n_loops']} \\\n"
        f"  --nodes-per-column {nodes_str} \\\n"
        f"  --attention-mode {ARCH_CONFIG['attention_mode']} \\\n"
        f"  --hybrid-full-attention-interval {ARCH_CONFIG['hybrid_full_attention_interval']} \\\n"
        f"  --seq-len {params['seq_len']} \\\n"
        f"  --batch-size {params['batch_size']} \\\n"
        f"  --grad-accum {params['grad_accum']} \\\n"
        f"  --lr {params['lr']} \\\n"
        f"  --weight-decay {params['weight_decay']} \\\n"
        f"  --warmup-ratio {params['warmup_ratio']} \\\n"
        f"  --grad-clip {params['grad_clip']} \\\n"
        f"  --dropout {params['dropout']} \\\n"
        f"  --dtype bfloat16 \\\n"
        f"  --epochs 3 \\\n"
        f"  --output-dir ./helixlm_400m_production"
    )
    return cmd


if __name__ == "__main__":
    main()
