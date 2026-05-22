"""
d_model × ffn_expansion factorial on 5M tokens, seq_len=128, three-stage LR.
BF16 AMP for production parity.

4-way factorial:
  d_model (256 vs 320) × ffn_expansion (2.0 vs 3.0)

Fixed: seq_len=128, n_loops=2, dropout=0.1, grad_buffer_ratio=1/e,
batch_size=32, grad_accum=1 (effective 32), three-stage LR,
weight_decay=0.1, grad_clip=1.0, use_cca=False, use_ssm=False,
use_titans_memory=False.
"""
import math
import json
import sys
import os
import time
import warnings

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from safetensors.torch import load_file as load_safetensors

from helix_lm.config import HelixConfig
from helix_lm.hf_model import HelixForCausalLM
from helix_lm.dataset import create_document_loader

# Try importing weightwatcher; if unavailable, skip weight analysis
try:
    import weightwatcher as ww
    WW_AVAILABLE = True
except Exception:
    WW_AVAILABLE = False
    warnings.warn("weightwatcher not available; skipping spectral analysis.")

DATASET = "david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427"
SEED = 42
OUTPUT_JSON = "/app/HelixLM/dmodel_ffn_results.json"
LOG_DIR = "/app/HelixLM/ablation_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Fixed hyperparameters
# ---------------------------------------------------------------------------
SEQ_LEN = 128
N_LOOPS = 2
DROPOUT = 0.1
GRAD_BUFFER_RATIO = 1.0 / math.e
BATCH_SIZE = 32
GRAD_ACCUM = 1
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
USE_CCA = False
USE_SSM = False
USE_TITANS = False
WARMUP1 = 50
WARMUP2 = 10
WARMUP3 = 10
EPOCHS_PER_STAGE = 1
TOTAL_EPOCHS = 3
AMP_DTYPE = torch.bfloat16
USE_AMP = True

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            if USE_AMP:
                with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE):
                    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            else:
                outputs = model(input_ids, labels=labels, attention_mask=attention_mask)

            loss = outputs.loss
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                num_batches += 1
    return total_loss / max(num_batches, 1)


def train_one_epoch(model, train_loader, optimizer, device, epoch_num, lr):
    model.train()
    epoch_loss = 0.0
    raw_count = 0
    accum_count = 0
    skipped = 0
    epoch_start = time.time()
    tokens_seen = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        tokens_seen += input_ids.numel()

        if USE_AMP:
            with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE):
                outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
                loss = outputs.loss
        else:
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            skipped += 1
            if skipped <= 5:
                print(f"  WARNING: NaN/Inf at batch {batch_idx}. Skipping.")
            continue

        # Gradient accumulation scaling
        divisor = GRAD_ACCUM
        is_last = (batch_idx + 1) == len(train_loader)
        if GRAD_ACCUM > 1 and is_last and accum_count < GRAD_ACCUM - 1:
            divisor = accum_count + 1
        loss = loss / divisor

        loss.backward()
        accum_count += 1
        epoch_loss += loss.item() * divisor
        raw_count += 1

        if accum_count >= GRAD_ACCUM or is_last:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0

        # Periodic logging
        if (batch_idx + 1) % 50 == 0 or is_last:
            avg = epoch_loss / max(raw_count, 1)
            ppl = math.exp(min(avg, 20))
            elapsed = time.time() - epoch_start
            tok_per_sec = tokens_seen / max(elapsed, 1e-6)
            print(
                f"  batch {batch_idx+1}/{len(train_loader)} | "
                f"loss={avg:.4f} ppl={ppl:.2f} lr={lr:.2e} "
                f"tok/s={tok_per_sec:.0f} skipped={skipped}"
            )

    train_avg = epoch_loss / max(raw_count, 1)
    train_ppl = math.exp(min(train_avg, 20))
    return {
        "train_loss": train_avg,
        "train_ppl": train_ppl,
        "time_s": time.time() - epoch_start,
        "tokens_seen": tokens_seen,
        "skipped_batches": skipped,
    }


def run_three_stage(d_model, n_heads, ffn_expansion, label, tokenizer,
                    train_texts, val_texts, vocab_size, device):
    """Three-stage LR: 2e-3 → 1e-3 → 3e-4, one epoch each."""
    print(f"\n{'='*70}")
    print(f"RUN: {label} | d={d_model} | heads={n_heads} | ffn={ffn_expansion}")
    print(f"{'='*70}")

    t0 = time.time()
    stage_results = []

    # Build config
    cfg_kwargs = dict(
        vocab_size=vocab_size,
        tokenizer_name="gpt2",
        d_model=d_model,
        n_heads=n_heads,
        n_loops=N_LOOPS,
        seq_len=SEQ_LEN,
        dropout=DROPOUT,
        ffn_expansion=ffn_expansion,
        weight_decay=WEIGHT_DECAY,
        grad_clip=GRAD_CLIP,
        grad_buffer_ratio=GRAD_BUFFER_RATIO,
        batch_size=BATCH_SIZE,
        use_cca=USE_CCA,
        use_ssm=USE_SSM,
        use_titans_memory=USE_TITANS,
        seed=SEED,
        device="auto",
        amp_dtype="bfloat16",
    )

    # --- Stage 1: LR = 2e-3 ---
    print(f"\n--- Stage 1: LR=2e-3 ---")
    cfg1 = HelixConfig.small_v2(lr=2e-3, epochs=1, warmup_steps=WARMUP1, **cfg_kwargs)
    cfg1.pad_token_id = tokenizer.pad_token_id
    cfg1.eos_token_id = tokenizer.eos_token_id
    cfg1.bos_token_id = tokenizer.bos_token_id

    model1 = HelixForCausalLM(cfg1).to(device)
    params1 = model1.count_parameters()
    print(f"Parameters: {params1['total']:,} total")

    train_loader1 = create_document_loader(
        train_texts, tokenizer, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
        shuffle=True, drop_last=True, lazy=True, stride=SEQ_LEN,
    )
    val_loader1 = create_document_loader(
        val_texts, tokenizer, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
        shuffle=False, drop_last=False, lazy=True, stride=SEQ_LEN,
    )

    optimizer1 = torch.optim.AdamW(
        model1.parameters(), lr=2e-3, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999)
    )

    s1 = train_one_epoch(model1, train_loader1, optimizer1, device, 1, 2e-3)
    val_loss_1 = evaluate(model1, val_loader1, device)
    val_ppl_1 = math.exp(min(val_loss_1, 20))
    print(f"Stage 1 — Train Loss: {s1['train_loss']:.4f}  Val Loss: {val_loss_1:.4f}  Val PPL: {val_ppl_1:.2f}")

    ckpt1_path = os.path.join(LOG_DIR, f"ckpt_{label}_s1")
    model1.save_pretrained(ckpt1_path)
    print(f"Saved checkpoint: {ckpt1_path}")

    stage_results.append({
        "stage": 1, "lr": 2e-3,
        "train_loss": s1["train_loss"], "train_ppl": s1["train_ppl"],
        "val_loss": val_loss_1, "val_ppl": val_ppl_1,
        "time_s": s1["time_s"],
    })

    # --- Stage 2: LR = 1e-3 ---
    print(f"\n--- Stage 2: LR=1e-3 ---")
    cfg2 = HelixConfig.small_v2(lr=1e-3, epochs=1, warmup_steps=WARMUP2, **cfg_kwargs)
    cfg2.pad_token_id = tokenizer.pad_token_id
    cfg2.eos_token_id = tokenizer.eos_token_id
    cfg2.bos_token_id = tokenizer.bos_token_id

    model2 = HelixForCausalLM(cfg2).to(device)
    st_path = os.path.join(ckpt1_path, "model.safetensors")
    if os.path.exists(st_path):
        sd = load_safetensors(st_path)
    else:
        sd = torch.load(os.path.join(ckpt1_path, "pytorch_model.bin"), map_location="cpu")
    model2.load_state_dict(sd, strict=False)
    model2 = model2.to(device)

    train_loader2 = create_document_loader(
        train_texts, tokenizer, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
        shuffle=True, drop_last=True, lazy=True, stride=SEQ_LEN,
    )
    val_loader2 = create_document_loader(
        val_texts, tokenizer, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
        shuffle=False, drop_last=False, lazy=True, stride=SEQ_LEN,
    )

    optimizer2 = torch.optim.AdamW(
        model2.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999)
    )

    s2 = train_one_epoch(model2, train_loader2, optimizer2, device, 2, 1e-3)
    val_loss_2 = evaluate(model2, val_loader2, device)
    val_ppl_2 = math.exp(min(val_loss_2, 20))
    print(f"Stage 2 — Train Loss: {s2['train_loss']:.4f}  Val Loss: {val_loss_2:.4f}  Val PPL: {val_ppl_2:.2f}")

    ckpt2_path = os.path.join(LOG_DIR, f"ckpt_{label}_s2")
    model2.save_pretrained(ckpt2_path)
    print(f"Saved checkpoint: {ckpt2_path}")

    stage_results.append({
        "stage": 2, "lr": 1e-3,
        "train_loss": s2["train_loss"], "train_ppl": s2["train_ppl"],
        "val_loss": val_loss_2, "val_ppl": val_ppl_2,
        "time_s": s2["time_s"],
    })

    # --- Stage 3: LR = 3e-4 ---
    print(f"\n--- Stage 3: LR=3e-4 ---")
    cfg3 = HelixConfig.small_v2(lr=3e-4, epochs=1, warmup_steps=WARMUP3, **cfg_kwargs)
    cfg3.pad_token_id = tokenizer.pad_token_id
    cfg3.eos_token_id = tokenizer.eos_token_id
    cfg3.bos_token_id = tokenizer.bos_token_id

    model3 = HelixForCausalLM(cfg3).to(device)
    st_path = os.path.join(ckpt2_path, "model.safetensors")
    if os.path.exists(st_path):
        sd = load_safetensors(st_path)
    else:
        sd = torch.load(os.path.join(ckpt2_path, "pytorch_model.bin"), map_location="cpu")
    model3.load_state_dict(sd, strict=False)
    model3 = model3.to(device)

    train_loader3 = create_document_loader(
        train_texts, tokenizer, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
        shuffle=True, drop_last=True, lazy=True, stride=SEQ_LEN,
    )
    val_loader3 = create_document_loader(
        val_texts, tokenizer, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
        shuffle=False, drop_last=False, lazy=True, stride=SEQ_LEN,
    )

    optimizer3 = torch.optim.AdamW(
        model3.parameters(), lr=3e-4, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999)
    )

    s3 = train_one_epoch(model3, train_loader3, optimizer3, device, 3, 3e-4)
    val_loss_3 = evaluate(model3, val_loader3, device)
    val_ppl_3 = math.exp(min(val_loss_3, 20))
    print(f"Stage 3 — Train Loss: {s3['train_loss']:.4f}  Val Loss: {val_loss_3:.4f}  Val PPL: {val_ppl_3:.2f}")

    ckpt3_path = os.path.join(LOG_DIR, f"ckpt_{label}_s3")
    model3.save_pretrained(ckpt3_path)
    print(f"Saved checkpoint: {ckpt3_path}")

    elapsed = time.time() - t0
    stage_results.append({
        "stage": 3, "lr": 3e-4,
        "train_loss": s3["train_loss"], "train_ppl": s3["train_ppl"],
        "val_loss": val_loss_3, "val_ppl": val_ppl_3,
        "time_s": s3["time_s"],
    })

    # WeightWatcher spectral analysis on final model
    ww_metrics = {}
    if WW_AVAILABLE:
        try:
            print(f"\nRunning WeightWatcher spectral analysis on {label}...")
            watcher = ww.WeightWatcher(model=model3)
            details = watcher.analyze()
            summary = watcher.get_summary()
            ww_metrics = {
                "alpha": summary.get("alpha"),
                "alpha_weighted": summary.get("alpha_weighted"),
                "log_norm": summary.get("log_norm"),
                "deltas": summary.get("deltas"),
                "num_layers": len(details),
            }
            print(f"  WeightWatcher: alpha={ww_metrics.get('alpha')}, log_norm={ww_metrics.get('log_norm')}")
        except Exception as e:
            print(f"  WeightWatcher failed: {e}")
            ww_metrics = {"error": str(e)}
    else:
        ww_metrics = {"skipped": "weightwatcher not available"}

    return {
        "label": label,
        "d_model": d_model,
        "n_heads": n_heads,
        "ffn_expansion": ffn_expansion,
        "params_total": params1["total"],
        "params_trainable": params1["trainable"],
        "final_train_loss": s3["train_loss"],
        "final_train_ppl": s3["train_ppl"],
        "final_val_loss": val_loss_3,
        "final_val_ppl": val_ppl_3,
        "best_val_loss": min(val_loss_1, val_loss_2, val_loss_3),
        "best_val_ppl": min(val_ppl_1, val_ppl_2, val_ppl_3),
        "total_time_s": elapsed,
        "stage_details": stage_results,
        "weightwatcher": ww_metrics,
    }


def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"BF16 support: {torch.cuda.is_bf16_supported()}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")

    # Dataset
    print(f"Loading dataset: {DATASET}")
    hf_ds = load_dataset(DATASET)
    train_texts = [ex["text"] for ex in hf_ds["pretrain_train"]]
    val_texts = [ex["text"] for ex in hf_ds["pretrain_val"]]
    print(f"Train docs: {len(train_texts):,} | Val docs: {len(val_texts):,}")

    runs = [
        (256, 4, 2.0, "d256_f2"),
        (256, 4, 3.0, "d256_f3"),
        (320, 5, 2.0, "d320_f2"),
        (320, 5, 3.0, "d320_f3"),
    ]

    all_results = []
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r") as f:
                all_results = json.load(f)
        except Exception:
            all_results = []

    completed_labels = {r["label"] for r in all_results}

    for d_model, n_heads, ffn, label in runs:
        if label in completed_labels:
            print(f"\nSkipping {label} — already in results.")
            continue

        result = run_three_stage(
            d_model, n_heads, ffn, label,
            tokenizer, train_texts, val_texts, vocab_size, device
        )
        all_results.append(result)

        # Save incremental results
        with open(OUTPUT_JSON, "w") as f:
            json.dump(all_results, f, indent=2)

        # Git commit after each run
        print("\nCommitting incremental results...")
        os.system(f"cd /app/HelixLM && git add -A && git commit -m \"ablation: {label} complete\" 2>&1 | tail -3")

    # Final report
    print(f"\n{'='*70}")
    print("d_model × FFN EXPANSION FACTORIAL RESULTS")
    print(f"{'='*70}")
    for r in sorted(all_results, key=lambda x: x["final_val_loss"]):
        print(
            f"  {r['label']:12s}  d={r['d_model']}  heads={r['n_heads']}  "
            f"ffn={r['ffn_expansion']}  params={r['params_total']:,}  "
            f"ppl={r['final_val_ppl']:.2f}  best_ppl={r['best_val_ppl']:.2f}  "
            f"time={r['total_time_s']:.0f}s"
        )

    best = min(all_results, key=lambda r: r["final_val_loss"])
    print(f"\nBest final val PPL: {best['label']} (ppl={best['final_val_ppl']:.2f})")
    print(f"Best any-stage val PPL: {best['label']} (ppl={best['best_val_ppl']:.2f})")

    # Save final JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
