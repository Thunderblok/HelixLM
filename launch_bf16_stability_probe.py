"""
HelixLM BF16 Numerical Stability Probe
========================================

Tests whether the fixed topology (mask propagation, CCA gate +2.0,
min_scale=0.05, merge normalization) trains stably with torch.bfloat16
at scale that previously produced NaN.

Previous NaN triggers at 5M/50M tokens, 512 seq, d=256, n_loops=2.
This probe tests multiple configurations systematically:
  - Pure AdamW vs Muon+AdamW hybrid
  - AMP on/off
  - torch.compile on/off
  - Grad clip values

Uses 5M subset for fast iteration. Expected: ~15-30 min per config on L4.

Usage:
    python launch_bf16_stability_probe.py [--config A|B|C|D|E|F]
"""
import argparse
import json
import math
import os
import random
import sys
import time

SEED = 42
random.seed(SEED)

import torch

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)

from helix_lm import (
    HelixConfig, HelixForCausalLM, HelixTokenizer,
    Trainer, build_hybrid_muon_adamw,
)
from helix_lm.dataset import create_document_loader
from datasets import load_dataset

# -- Probe configurations ------------------------------------------------------
# Each config varies a single axis to isolate the NaN trigger
PROBE_CONFIGS = {
    "A": {  # Baseline: FP32, no compile, pure AdamW (known good)
        "dtype": "float32",
        "use_amp": False,
        "use_compile": False,
        "use_muon": False,
        "grad_clip": 1.0,
        "label": "FP32_baseline",
    },
    "B": {  # BF16 weights, no AMP (weights in bf16, compute in bf16)
        "dtype": "bfloat16",
        "use_amp": False,
        "use_compile": False,
        "use_muon": False,
        "grad_clip": 1.0,
        "label": "BF16_noAMP",
    },
    "C": {  # BF16 weights + AMP autocast (most aggressive)
        "dtype": "bfloat16",
        "use_amp": True,
        "use_compile": False,
        "use_muon": False,
        "grad_clip": 1.0,
        "label": "BF16_AMP",
    },
    "D": {  # BF16 + Muon hybrid (test Muon stability in bf16)
        "dtype": "bfloat16",
        "use_amp": False,
        "use_compile": False,
        "use_muon": True,
        "grad_clip": 1.0,
        "label": "BF16_Muon",
    },
    "E": {  # BF16 + torch.compile (test compile stability)
        "dtype": "bfloat16",
        "use_amp": False,
        "use_compile": True,
        "use_muon": False,
        "grad_clip": 1.0,
        "label": "BF16_compile",
    },
    "F": {  # Full combo: BF16 + AMP + compile + Muon (most aggressive)
        "dtype": "bfloat16",
        "use_amp": True,
        "use_compile": True,
        "use_muon": True,
        "grad_clip": 1.0,
        "label": "BF16_full_combo",
    },
}


def run_probe(probe_cfg, dataset_name, output_root):
    """Run a single BF16 stability probe."""
    label = probe_cfg["label"]
    print(f"\n{'='*70}")
    print(f"PROBE: {label}")
    print(f"  dtype={probe_cfg['dtype']} amp={probe_cfg['use_amp']} "
          f"compile={probe_cfg['use_compile']} muon={probe_cfg['use_muon']}")
    print(f"{'='*70}")

    t0 = time.time()

    # -- Dataset ----------------------------------------------------------------
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)
    train_texts = list(ds["pretrain_train"]["text"])
    val_texts = list(ds["pretrain_val"]["text"])
    print(f"Train docs: {len(train_texts):,} | Val docs: {len(val_texts):,}")

    # -- Tokenizer --------------------------------------------------------------
    tok = HelixTokenizer("gpt2")
    vs = len(tok)
    print(f"Vocab={vs}")

    # -- Model config -----------------------------------------------------------
    cfg = HelixConfig(
        vocab_size=vs,
        d_model=256,
        n_columns=3,
        nodes_per_column=(2, 3, 2),
        n_heads=4,
        n_loops=2,
        seq_len=512,
        batch_size=8,
        use_titans_memory=False,
        attention_mode="hybrid",
        dropout=0.05,
        lr=1e-3,
        weight_decay=0.01,
        epochs=1,  # Single epoch for speed
        warmup_steps=200,
        grad_clip=probe_cfg["grad_clip"],
        device="auto",
        dtype=probe_cfg["dtype"],
        use_cca=True,
        cca_warmup_steps=3000,
        cca_ramp_mode="cubic_ease",
        cca_min_scale=0.05,
    )
    cfg.pad_token_id = tok.pad_token_id
    cfg.eos_token_id = tok.eos_token_id

    model = HelixForCausalLM(cfg)
    params = model.count_parameters()["total"]
    print(f"Params: {params:,}")

    # -- torch.compile ----------------------------------------------------------
    if probe_cfg["use_compile"] and torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="default")
            print("torch.compile enabled.")
        except Exception as e:
            print(f"torch.compile failed: {e}. Continuing eager.")

    # -- DataLoaders ------------------------------------------------------------
    train_loader = create_document_loader(
        train_texts, tok, seq_len=512, batch_size=8,
        shuffle=True, drop_last=True, lazy=True,
    )
    val_loader = create_document_loader(
        val_texts, tok, seq_len=512, batch_size=8,
        shuffle=False, drop_last=False, lazy=True,
    )

    # -- Optimizer --------------------------------------------------------------
    optimizer = None
    if probe_cfg["use_muon"]:
        optimizer = build_hybrid_muon_adamw(model, lr=cfg.lr, weight_decay=cfg.weight_decay)
        print("Using Muon + AdamW hybrid optimizer.")

    # -- Trainer ----------------------------------------------------------------
    output_dir = os.path.join(output_root, f"probe_{label}")
    os.makedirs(output_dir, exist_ok=True)

    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tok,
        output_dir=output_dir,
        example_prompts=["Once upon a time"],
        generated_example_length=10,
        grad_accum_steps=1,
        use_amp=probe_cfg["use_amp"],
        verbose=True,
        optimizer=optimizer,
    )
    # Ensure correct betas for AdamW
    target = trainer.optimizer
    if isinstance(target, (list, tuple)):
        target = target[-1]  # AdamW is the second optimizer in hybrid
    for group in target.param_groups:
        group["betas"] = (0.9, 0.999)

    # -- Train ------------------------------------------------------------------
    nan_detected = False
    skipped_batches = 0
    try:
        history = trainer.train(num_epochs=1, eval_every=1)
        final_train_loss = history["train_loss"][-1]
        final_val_loss = history["val_loss"][-1] if history.get("val_loss") else float("inf")
        skipped_batches = history.get("skipped_batches", 0)
        train_ppl = math.exp(min(final_train_loss, 20))
        val_ppl = math.exp(min(final_val_loss, 20))

        # NaN check
        if math.isnan(train_ppl) or math.isnan(val_ppl):
            nan_detected = True

    except RuntimeError as e:
        if "NaN" in str(e) or "nan" in str(e):
            nan_detected = True
            final_train_loss = float("nan")
            final_val_loss = float("nan")
            train_ppl = float("inf")
            val_ppl = float("inf")
        else:
            raise

    elapsed = time.time() - t0

    # -- Results ----------------------------------------------------------------
    stable = (
        not nan_detected
        and val_ppl < 500
        and not math.isnan(val_ppl)
        and not math.isinf(val_ppl)
        and skipped_batches < 10  # Allow a few skipped batches
    )

    result = {
        "probe": label,
        "config": probe_cfg,
        "stable": stable,
        "nan_detected": nan_detected,
        "train_ppl": train_ppl,
        "val_ppl": val_ppl,
        "train_loss": final_train_loss,
        "val_loss": final_val_loss,
        "skipped_batches": skipped_batches,
        "elapsed_sec": elapsed,
        "params": params,
    }

    print(f"\n{'='*70}")
    if stable:
        print(f"PASS: {label} is STABLE")
    else:
        print(f"FAIL: {label} is UNSTABLE")
    print(f"  Train PPL: {train_ppl:.2f}")
    print(f"  Val PPL:   {val_ppl:.2f}")
    print(f"  Skipped:   {skipped_batches}")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"{'='*70}")

    # Save result
    with open(os.path.join(output_dir, "probe_result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Run single config (A-F). If None, runs all.")
    parser.add_argument("--dataset", type=str,
                        default="david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427")
    parser.add_argument("--output", type=str, default="./bf16_probe_results")
    args = parser.parse_args()

    print(f"{'='*70}")
    print("HELIXLM BF16 STABILITY PROBE")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Output:  {args.output}")
    print(f"Device:  {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*70}")

    os.makedirs(args.output, exist_ok=True)

    configs_to_run = [args.config] if args.config else list(PROBE_CONFIGS.keys())
    all_results = []

    for key in configs_to_run:
        if key not in PROBE_CONFIGS:
            print(f"Unknown config: {key}. Available: {list(PROBE_CONFIGS.keys())}")
            continue
        result = run_probe(PROBE_CONFIGS[key], args.dataset, args.output)
        all_results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("BF16 STABILITY PROBE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<20}{'Label':<20}{'Stable':<10}{'Val PPL':<12}{'Skipped':<10}")
    print("-" * 70)
    for r in all_results:
        status = "PASS" if r["stable"] else "FAIL"
        print(f"{r['config'].get('label',''):<20}{r['probe']:<20}{status:<10}"
              f"{r['val_ppl']:<12.2f}{r['skipped_batches']:<10}")

    # Save combined results
    with open(os.path.join(args.output, "probe_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Determine if BF16 is safe for production
    stable_configs = [r for r in all_results if r["stable"]]
    if stable_configs:
        print(f"\n{len(stable_configs)}/{len(all_results)} configs stable.")
        bf16_safe_for_50m = any(
            r["stable"] for r in all_results
            if r["config"].get("dtype") == "bfloat16"
        )
        if bf16_safe_for_50m:
            print("BF16 appears safe for 50M/400M runs.")
        else:
            print("FP32 configs stable but BF16 failed. Stay FP32 for production.")
    else:
        print("No configs stable. Investigate before scaling.")

    print(f"\nResults saved to: {args.output}/probe_summary.json")


if __name__ == "__main__":
    main()
