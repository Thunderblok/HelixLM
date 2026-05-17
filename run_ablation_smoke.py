#!/usr/bin/env python3
"""
HelixLM Smoke Test + WeightWatcher Ablations

Tests at realistic seq_len=512 with the tiny production dataset:
1. Tied vs Untied parameter counts and forward pass
2. Quick train/val loop with DocumentAwareDataset (production scheme)
3. WeightWatcher analysis on both variants
4. Gradient health comparison

Usage:
    python run_ablation_smoke.py [--device cpu|cuda] [--max_samples N]
"""
import argparse
import os
import sys
import math
import time
import json
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
parser.add_argument("--max_samples", type=int, default=500,
                    help="Subset of tiny dataset for fast iteration")
parser.add_argument("--seq_len", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--epochs", type=int, default=2,
                    help="Quick smoke test epochs (use 3-5 for real measurement)")
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--output_dir", default="./ablation_outputs")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helix_lm.config import HelixConfig
from helix_lm.hf_model import HelixForCausalLM
from helix_lm.tokenizer import HelixTokenizer
from helix_lm.dataset import create_document_loader
from helix_lm.trainer import Trainer

# ---------------------------------------------------------------------------
# Helper: load tiny dataset
# ---------------------------------------------------------------------------
def load_tiny_dataset(max_samples=500):
    """Load a subset of the HelixLM tiny dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        ds = load_dataset("david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427", split="train")
        texts = [item["text"] for item in ds.select(range(min(max_samples, len(ds))))]
        print(f"  Loaded {len(texts)} samples from HF dataset")
        return texts
    except Exception as e:
        print(f"  WARNING: Could not load HF dataset: {e}")
        print("  Falling back to bible.txt")
        bible_path = os.path.join(os.path.dirname(__file__), "helix_lm", "bible.txt")
        if os.path.exists(bible_path):
            with open(bible_path, 'r', encoding='utf-8') as f:
                text = f.read()
            # Split into ~500 char chunks to simulate documents
            chunk_size = 500
            texts = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            texts = [t for t in texts if len(t) > 50][:max_samples]
            print(f"  Loaded {len(texts)} chunks from bible.txt")
            return texts
        return ["In the beginning God created the heaven and the earth."] * 100


# ---------------------------------------------------------------------------
# Helper: WeightWatcher analysis
# ---------------------------------------------------------------------------
def run_weightwatcher(model, name, output_dir):
    """Run WeightWatcher analysis and save metrics."""
    import weightwatcher as ww

    print(f"\n  [WeightWatcher] Analyzing {name}...")
    watcher = ww.WeightWatcher(model=model)

    # Analyze the model
    try:
        details = watcher.analyze(plot=False, mp_fit=True)
        summary = watcher.get_summary(details)

        print(f"    alpha (weight normality): {summary.get('alpha', 'N/A')}")
        print(f"    alpha_weighted: {summary.get('alpha_weighted', 'N/A')}")
        print(f"    log_norm (overall capacity): {summary.get('log_norm', 'N/A')}")
        print(f"    num_pl_spikes (PL outliers): {summary.get('num_pl_spikes', 'N/A')}")
        print(f"    num_mp_spikes (MP outliers): {summary.get('num_mp_spikes', 'N/A')}")

        # Per-layer analysis for gradient imbalance detection
        layer_metrics = []
        if details is not None and not details.empty:
            for _, row in details.iterrows().__iter__():
                try:
                    layer_id = row.get('layer_id', 'unknown')
                    layer_type = row.get('layer_type', 'unknown')
                    alpha = row.get('alpha', float('nan'))
                    log_norm = row.get('log_norm', float('nan'))
                    spectral_norm = row.get('spectral_norm', float('nan'))
                    stable_rank = row.get('stable_rank', float('nan'))

                    # Flag imbalanced layers
                    flags = []
                    if not math.isnan(alpha):
                        if alpha < 2:
                            flags.append("heavy_tailed (alpha<2)")
                        if alpha > 6:
                            flags.append("over_regularized (alpha>6)")
                    if not math.isnan(spectral_norm):
                        if spectral_norm > 10:
                            flags.append("large_spectral_norm")

                    layer_metrics.append({
                        'layer_id': str(layer_id),
                        'layer_type': str(layer_type),
                        'alpha': float(alpha) if not math.isnan(alpha) else None,
                        'log_norm': float(log_norm) if not math.isnan(log_norm) else None,
                        'spectral_norm': float(spectral_norm) if not math.isnan(spectral_norm) else None,
                        'stable_rank': float(stable_rank) if not math.isnan(stable_rank) else None,
                        'flags': flags,
                    })
                except Exception as e:
                    pass

        # Sort by alpha to find most imbalanced layers
        layer_metrics.sort(key=lambda x: x['alpha'] if x['alpha'] is not None else 999)

        print(f"\n    Top 10 most imbalanced layers (by alpha):")
        for lm in layer_metrics[:10]:
            flag_str = f" [{', '.join(lm['flags'])}]" if lm['flags'] else ""
            print(f"      {lm['layer_type']:20s} alpha={lm['alpha']:.3f} log_norm={lm['log_norm']:.3f} spec_norm={lm['spectral_norm']:.3f}{flag_str}")

        # Save detailed results
        results = {
            'name': name,
            'summary': {k: float(v) if isinstance(v, (int, float)) else str(v)
                        for k, v in summary.items()},
            'layer_metrics': layer_metrics,
        }
        out_path = os.path.join(output_dir, f"ww_{name.replace(' ', '_')}.json")
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"    Saved to {out_path}")

        return summary, layer_metrics

    except Exception as e:
        print(f"    WeightWatcher analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, []


# ---------------------------------------------------------------------------
# Helper: compute gradient norms per layer
# ---------------------------------------------------------------------------
def compute_gradient_norms(model):
    """Compute gradient norms per layer to detect imbalance."""
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    return grad_norms


# ---------------------------------------------------------------------------
# Helper: train and evaluate
# ---------------------------------------------------------------------------
def train_variant(variant_name, cfg, tokenizer, train_texts, val_texts, device):
    """Train one model variant and return results."""
    print(f"\n{'='*60}")
    print(f"Variant: {variant_name}")
    print(f"  d_model={cfg.d_model}, n_columns={cfg.n_columns}, "
          f"n_loops={cfg.n_loops}, tie={cfg.tie_word_embeddings}, "
          f"buffer={getattr(cfg, 'grad_buffer_ratio', 'N/A')}")
    print(f"{'='*60}")

    # Model
    model = HelixForCausalLM(cfg)
    print(f"  Parameters: {model.count_parameters()['total']:,}")

    # Data loaders with production scheme (DocumentAwareDataset)
    train_loader = create_document_loader(
        train_texts, tokenizer, cfg.seq_len, cfg.batch_size,
        shuffle=True, min_tail_len=1, lazy=True,
    )
    val_loader = create_document_loader(
        val_texts, tokenizer, cfg.seq_len, cfg.batch_size,
        shuffle=False, drop_last=False, min_tail_len=1, lazy=True,
    )
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Trainer
    trainer = Trainer(
        model=model, cfg=cfg, tokenizer=tokenizer,
        output_dir=os.path.join(args.output_dir, variant_name.replace(" ", "_")),
        grad_accum_steps=1, use_amp=False, verbose=True,
        train_loader=train_loader, val_loader=val_loader,
        example_prompts=["In 1492, Columbus", "The quantum", "Once upon a time"],
        generated_example_length=15,
    )

    # Train
    start = time.time()
    history = trainer.train(num_epochs=cfg.epochs, eval_every=1)
    elapsed = time.time() - start

    # Final eval
    final_val_ppl = history['perplexity'][-1] if history['perplexity'] else float('inf')
    final_train_loss = history['train_loss'][-1] if history['train_loss'] else float('inf')

    print(f"\n  [{variant_name}] Results:")
    print(f"    Train loss: {final_train_loss:.4f}")
    print(f"    Val PPL: {final_val_ppl:.2f}")
    print(f"    Time: {elapsed:.1f}s")
    print(f"    Params: {model.count_parameters()['total']:,}")

    return {
        'model': model,
        'variant': variant_name,
        'cfg': cfg,
        'history': history,
        'final_train_loss': final_train_loss,
        'final_val_ppl': final_val_ppl,
        'elapsed': elapsed,
        'params': model.count_parameters()['total'],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("HelixLM Smoke Test + WeightWatcher Ablations")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Dataset: {args.max_samples} samples")
    print(f"Seq len: {args.seq_len}, Batch: {args.batch_size}, Epochs: {args.epochs}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    print("\n--- Loading dataset ---")
    all_texts = load_tiny_dataset(args.max_samples)

    # Split: 80/20 train/val
    split_idx = int(len(all_texts) * 0.8)
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]
    print(f"  Train: {len(train_texts)}, Val: {len(val_texts)}")

    # -----------------------------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------------------------
    print("\n--- Tokenizer ---")
    tokenizer = HelixTokenizer("gpt2")
    print(f"  Vocab: {len(tokenizer)}")

    # -----------------------------------------------------------------------
    # Configs
    # -----------------------------------------------------------------------
    print("\n--- Configs ---")

    base_kwargs = dict(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        use_titans_memory=False,
        use_ssm=False,
    )

    # Variant 1: Tied with buffer
    cfg_tied = HelixConfig.micro(**base_kwargs, tie_word_embeddings=True, grad_buffer_ratio=0.5)

    # Variant 2: Untied (sacred baseline)
    cfg_untied = HelixConfig.micro(**base_kwargs, tie_word_embeddings=False)

    print(f"  Tied (buffer=0.5):   {cfg_tied.d_model}d, {cfg_tied.n_columns}cols, {cfg_tied.n_loops}loops")
    print(f"  Untied:              {cfg_untied.d_model}d, {cfg_untied.n_columns}cols, {cfg_untied.n_loops}loops")

    # -----------------------------------------------------------------------
    # Train variants
    # -----------------------------------------------------------------------
    results = {}

    # Tied
    results['tied'] = train_variant(
        "Tied buffer=0.5", cfg_tied, tokenizer, train_texts, val_texts, args.device
    )

    # Untied
    results['untied'] = train_variant(
        "Untied baseline", cfg_untied, tokenizer, train_texts, val_texts, args.device
    )

    # -----------------------------------------------------------------------
    # WeightWatcher analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("WeightWatcher Analysis")
    print("=" * 60)

    ww_results = {}
    for key, res in results.items():
        model = res['model']
        # Put model in eval mode for WW
        model.eval()
        summary, layers = run_weightwatcher(model, res['variant'], args.output_dir)
        ww_results[key] = {'summary': summary, 'layers': layers}

    # -----------------------------------------------------------------------
    # Gradient norm comparison
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Gradient Norm Comparison")
    print("=" * 60)

    for key, res in results.items():
        model = res['model']
        cfg = res['cfg']

        # Build a single training batch for gradient computation
        loader = create_document_loader(
            train_texts[:args.batch_size*2], tokenizer, cfg.seq_len, cfg.batch_size,
            shuffle=False, min_tail_len=1, lazy=True,
        )
        batch = next(iter(loader))
        input_ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)

        model.train()
        model.zero_grad()
        out = model(input_ids, labels=labels)
        loss = out["loss"]
        loss.backward()

        grad_norms = compute_gradient_norms(model)

        # Aggregate by layer type
        by_type = {}
        for name, norm in grad_norms.items():
            # Categorize
            if 'embed' in name:
                cat = 'embedding'
            elif 'lm_head' in name or 'buffer' in name:
                cat = 'output_head'
            elif 'graph' in name and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name):
                cat = 'attention_qkv'
            elif 'graph' in name and 'out_proj' in name:
                cat = 'attention_out'
            elif 'graph' in name and ('gate' in name or 'swiglu' in name):
                cat = 'ffn'
            elif 'graph' in name:
                cat = 'graph_other'
            elif 'lti' in name or 'act' in name:
                cat = 'recurrent'
            else:
                cat = 'other'

            by_type.setdefault(cat, []).append(norm)

        print(f"\n  [{res['variant']}] Gradient norms by component:")
        for cat, norms in sorted(by_type.items()):
            avg = sum(norms) / len(norms)
            max_n = max(norms)
            print(f"    {cat:20s}: avg={avg:.6f}, max={max_n:.6f}, n_layers={len(norms)}")

        # Specifically compare embedding gradient
        embed_norm = grad_norms.get('model.embed.weight', 0)
        head_norm = grad_norms.get('lm_head.weight', 0) or grad_norms.get('lm_head.buffer.weight', 0)
        print(f"    Embedding grad norm: {embed_norm:.6f}")
        print(f"    Head/buffer grad norm: {head_norm:.6f}")
        if embed_norm > 0 and head_norm > 0:
            print(f"    Head/Embed ratio: {head_norm/embed_norm:.3f}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for key, res in results.items():
        print(f"\n  {res['variant']}:")
        print(f"    Params:      {res['params']:,}")
        print(f"    Train loss:  {res['final_train_loss']:.4f}")
        print(f"    Val PPL:     {res['final_val_ppl']:.2f}")
        print(f"    Time:        {res['elapsed']:.1f}s")

    tied = results['tied']
    untied = results['untied']
    print(f"\n  Parameter reduction: {(1 - tied['params']/untied['params'])*100:.1f}%")
    print(f"  Val PPL tied:   {tied['final_val_ppl']:.2f}")
    print(f"  Val PPL untied: {untied['final_val_ppl']:.2f}")
    ppl_diff = tied['final_val_ppl'] - untied['final_val_ppl']
    print(f"  PPL delta:      {ppl_diff:+.2f} ({'tied better' if ppl_diff < 0 else 'untied better'})")

    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETE")
    print("=" * 60)

    # Save summary
    summary = {
        'tied': {k: v for k, v in results['tied'].items() if k not in ('model', 'cfg')},
        'untied': {k: v for k, v in results['untied'].items() if k not in ('model', 'cfg')},
    }
    with open(os.path.join(args.output_dir, 'ablation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {args.output_dir}/ablation_summary.json")


if __name__ == "__main__":
    main()
