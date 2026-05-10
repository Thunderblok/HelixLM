#!/usr/bin/env python3
"""
HelixLM Production Training Script v1.0
======================================
Open-source reproducible multi-stage training for HelixLM.

Stages:
  base    : 3 epochs on pretrain_train @ LR=3e-3  (high LR is a feature)
  grok    : N epochs on pretrain_train @ LR=3e-4   (default 10)
  instruct: M epochs on instruct_train @ LR=1e-5  (default 5)

Model name embeds architecture + training metadata for transparency.

Usage:
  # Ablations (L4, <1hr total if run in parallel)
  python scripts/train_helixlm_production.py --stage base --epochs 1 --max-samples 50000 \
    --d-model 384 --attention-mode hybrid --output-dir ./ablation_hybrid_384

  # Base training (L40S)
  python scripts/train_helixlm_production.py --stage base --torch-compile \
    --dataset-repo david-thrower/HelixLM-tiny-400.0Mt-730000pt-57143it-20260430 \
    --d-model 384 --attention-mode hybrid --output-dir ./helixlm_base --hf-repo-id <name>

  # Grokking (L40S)
  python scripts/train_helixlm_production.py --stage grok --torch-compile \
    --resume-from <base-model-repo> --output-dir ./helixlm_grok --hf-repo-id <name>

  # Instruction tuning (L40S or L4)
  python scripts/train_helixlm_production.py --stage instruct --torch-compile \
    --resume-from <grok-model-repo> --output-dir ./helixlm_instruct --hf-repo-id <name>
"""
import argparse
import gc
import math
import os
import sys
import time
import traceback
import warnings
from datetime import datetime

import torch
from datasets import load_dataset
from tqdm import tqdm


from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer, Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="HelixLM Production Training")

    # Stage
    parser.add_argument("--stage", choices=["base", "grok", "instruct"], required=True,
                        help="Training stage")

    # Resume from previous stage
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Local path or HF hub repo ID to resume from")

    # Dataset
    parser.add_argument("--dataset-repo", type=str,
                        default="david-thrower/HelixLM-tiny-400.0Mt-730000pt-57143it-20260430")
    parser.add_argument("--train-split", type=str, default=None,
                        help="Auto-set by stage if omitted")
    parser.add_argument("--val-split", type=str, default="pretrain_val")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap training samples (for quick ablations)")

    # Architecture
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-columns", type=int, default=2)
    parser.add_argument("--n-loops", type=int, default=1)
    parser.add_argument("--nodes-per-column", type=str, default="2,2",
                        help="Comma-separated, e.g. '2,2' or '2,3,2'")
    parser.add_argument("--attention-mode", type=str, default="hybrid",
                        choices=["linear", "full", "hybrid"])
    parser.add_argument("--hybrid-full-attention-interval", type=int, default=2)
    parser.add_argument("--use-ssm", action="store_true")
    parser.add_argument("--use-titans-memory", action="store_true")
    parser.add_argument("--ffn-expansion", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.05)

    # Training
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override stage default")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override stage default")
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--min-tail-len", type=int, default=None)

    # Optimization
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="bfloat16 on Ampere+ (L4/L40S); float32 for debug")
    parser.add_argument("--torch-compile", action="store_true",
                        help="Enable torch.compile (big speedup on L40S)")

    # Output
    parser.add_argument("--output-dir", type=str, default="./helixlm_output")
    parser.add_argument("--hf-repo-id", type=str, default=None,
                        help="Push to HF hub (auto-generated if omitted)")
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN"))

    # System
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


def stage_defaults(stage):
    """Per-stage defaults. High LR is intentional for HelixLM."""
    if stage == "base":
        return {"epochs": 3, "lr": 3e-3, "train_split": "pretrain_train"}
    elif stage == "grok":
        return {"epochs": 10, "lr": 3e-4, "train_split": "pretrain_train"}
    elif stage == "instruct":
        return {"epochs": 5, "lr": 1e-5, "train_split": "instruct_train"}
    raise ValueError(f"Unknown stage: {stage}")


def build_model_name(args, param_count):
    """Embed metadata into model/repo name for open-source transparency."""
    short_attn = args.attention_mode[:3]
    ssm_tag = "-ssm" if args.use_ssm else ""
    titans_tag = "-titans" if args.use_titans_memory else ""
    params_m = round(param_count / 1e6)
    lr_str = f"{args.lr:.0e}".replace("+", "").replace("-", "")
    name = (
        f"HelixLM-{args.d_model}d-{short_attn}-{args.n_columns}c{args.n_loops}l"
        f"{ssm_tag}{titans_tag}-{params_m}M-400Mt-{args.stage}-{args.epochs}ep-{lr_str}lr"
    )
    return name


def load_split(repo_id, split_name, max_samples=None):
    """
    Load a dataset split into a list of strings.
    
    Uses streaming=True to match the NAS script and avoid disk cache.
    NOTE: List[str] still materializes in RAM; for 730k samples this is ~1-2 GB.
    The DocumentAwareDataset chunking step is the larger consumer (~2-3 GB).
    Both fit comfortably in L4/L40S RAM.
    """
    print(f"  Loading {repo_id}  split={split_name} ...")
    ds = load_dataset(repo_id, split=split_name, streaming=True, trust_remote_code=False)
    texts = []
    iterable = tqdm(ds, desc=f"  {split_name}", total=max_samples, leave=False)
    for i, item in enumerate(iterable):
        if max_samples is not None and i >= max_samples:
            break
        texts.append(item.get("text", ""))
    print(f"  -> {len(texts):,} samples")
    return texts


def main():
    args = parse_args()
    defaults = stage_defaults(args.stage)

    # Apply stage defaults, allow CLI override
    args.epochs = args.epochs if args.epochs is not None else defaults["epochs"]
    args.lr = args.lr if args.lr is not None else defaults["lr"]
    if args.train_split is None:
        args.train_split = defaults["train_split"]
    if args.min_tail_len is None:
        args.min_tail_len = args.seq_len // 4

    # Fix #4: Explicit nodes_per_column (never rely on .tiny() hardcode)
    args.nodes_per_column = tuple(int(x.strip()) for x in args.nodes_per_column.split(","))
    if len(args.nodes_per_column) != args.n_columns:
        raise ValueError(
            f"nodes_per_column length {len(args.nodes_per_column)} != n_columns {args.n_columns}"
        )

    torch.manual_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"HelixLM Production | Stage: {args.stage.upper()}")
    print(f"{'='*60}")
    print(f"Arch: d={args.d_model}, cols={args.n_columns}, loops={args.n_loops}")
    print(f"Nodes/col: {args.nodes_per_column}")
    print(f"Attn: {args.attention_mode}, interval={args.hybrid_full_attention_interval}")
    print(f"SSM: {args.use_ssm}, Titans: {args.use_titans_memory}")
    print(f"Train: epochs={args.epochs}, lr={args.lr}, warmup={args.warmup_steps}")
    print(f"Batch: {args.batch_size}, accum={args.grad_accum}, seq={args.seq_len}")
    print(f"{'='*60}\n")

    # Tokenizer
    tokenizer = HelixTokenizer("gpt2")
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")

    # Build config (Fix #1: real epochs passed into config for correct scheduler horizon)
    cfg = HelixConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_columns=args.n_columns,
        n_loops=args.n_loops,
        nodes_per_column=args.nodes_per_column,
        seq_len=args.seq_len,
        attention_mode=args.attention_mode,
        hybrid_full_attention_interval=args.hybrid_full_attention_interval,
        use_ssm=args.use_ssm,
        use_titans_memory=args.use_titans_memory,
        ffn_expansion=args.ffn_expansion,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        dtype=args.dtype,
        device=args.device,
        tokenizer_name="gpt2",
        use_rope=True,
    )

    # Fix #2: Sync token IDs from tokenizer (do not leave at 0)
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.bos_token_id = tokenizer.bos_token_id

    model = None
    try:
        # Resume or fresh
        if args.resume_from:
            print(f"Resuming from: {args.resume_from}")
            if os.path.isdir(args.resume_from):
                model = HelixForCausalLM.from_pretrained(args.resume_from)
            else:
                model = HelixForCausalLM.from_pretrained(
                    args.resume_from, trust_remote_code=True
                )
            # Override config for new stage
            model.config.epochs = args.epochs
            model.config.lr = args.lr
            model.config.warmup_steps = args.warmup_steps
            model.config.grad_clip = args.grad_clip
            model.config.weight_decay = args.weight_decay
        else:
            model = HelixForCausalLM(cfg)

        param_count = model.count_parameters()["total"]
        print(f"Parameters: {param_count:,}")

        # Torch compile (L40S loves this; L4 tolerates it)
        if args.torch_compile and torch.cuda.is_available():
            print("Applying torch.compile (reduce-overhead)...")
            model = torch.compile(model, mode="reduce-overhead")

        # Device
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        model = model.to(device)
        print(f"Device: {device}")

        # Data
        train_texts = load_split(args.dataset_repo, args.train_split, args.max_samples)

        val_texts = None
        if args.val_split:
            try:
                val_texts = load_split(args.dataset_repo, args.val_split)
            except Exception as e:
                print(f"  [WARN] Val split {args.val_split} not loaded: {e}")
                val_texts = None

        # Trainer (Fix #3: explicit min_tail_len passthrough)
        effective_cfg = model.config if args.resume_from else cfg
        trainer = Trainer(
            model=model,
            cfg=effective_cfg,
            train_texts=train_texts,
            val_texts=val_texts,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            example_prompts=[
                "The next day",
                "In 1492,",
                "Engine chip tuning is",
            ],
            generated_example_length=30,
            grad_accum_steps=args.grad_accum,
            use_amp=(args.dtype == "bfloat16" and torch.cuda.is_available()),
            min_tail_len=args.min_tail_len,
            verbose=True,
        )

        # Train
        print(f"\n>>> Starting {args.stage.upper()} training...")
        history = trainer.train(num_epochs=args.epochs, eval_every=max(1, args.epochs // 3))

        final_train_loss = history["train_loss"][-1] if history.get("train_loss") else float("inf")
        final_val_loss = history["val_loss"][-1] if history.get("val_loss") else float("inf")
        print(f"\nFinal train loss: {final_train_loss:.4f}")
        print(f"Final val PPL:   {math.exp(min(final_val_loss, 20)):.2f}")

        # Save locally
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Checkpoint saved to {args.output_dir}")

        # Push to hub with metadata-rich name
        if args.hf_token:
            repo_name = args.hf_repo_id or build_model_name(args, param_count)
            print(f"Pushing to HF hub: {repo_name}")
            model.push_to_hub(repo_name, token=args.hf_token)
            tokenizer.push_to_hub(repo_name, token=args.hf_token)
            print(f"Done: https://huggingface.co/{repo_name}")

        print(f"\n{'='*60}")
        print(f"STAGE {args.stage.upper()} COMPLETE")
        print(f"{'='*60}")

    finally:
        # Fix #5: Unconditional GPU cleanup (catches early returns, exceptions, NaN)
        if "trainer" in locals() and trainer is not None:
            del trainer
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print("GPU memory cleared.")


if __name__ == "__main__":
    main()
