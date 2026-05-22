"""
5M-token benchmark probe for HelixLM RNG + bf16 fixes.

Config: d_model=320, n_heads=5, n_loops=2, dropout=0.1, seed=42
Dataset: david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427
LR: step decay 2e-3 -> 1e-3 -> 3e-4
Batch: 48 + accum 2 (effective 96) on L4
AMP: bfloat16
seq_len: 128
"""
import math
import os
import sys
import time
from typing import Optional

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer
from datasets import load_dataset

from helix_lm.config import HelixConfig
from helix_lm.hf_model import HelixForCausalLM
from helix_lm.dataset import create_document_loader

# -- Config ---------------------------------------------------------------
cfg = HelixConfig.small_v2(
    vocab_size=50257,
    tokenizer_name="gpt2",
    d_model=320,
    n_heads=5,
    n_loops=2,
    dropout=0.1,
    grad_buffer_ratio=1.0 / math.e,
    seed=42,
    seq_len=128,
    batch_size=48,
    lr=2e-3,  # initial LR, overridden by manual schedule
    weight_decay=0.1,
    grad_clip=1.0,
    use_cca=False,
    tie_word_embeddings=True,
    attention_mode="hybrid",
    use_rope=True,
    use_titans_memory=False,
    use_ssm=False,
    amp_dtype="bfloat16",
    device="auto",
)

# -- Tokenizer ------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# -- Load Dataset ---------------------------------------------------------
print("Loading dataset...")
hf_ds = load_dataset("david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427")
train_texts = [ex["text"] for ex in hf_ds["pretrain_train"]]
val_texts = [ex["text"] for ex in hf_ds["pretrain_val"]]

print(f"Train docs: {len(train_texts):,} | Val docs: {len(val_texts):,}")

# -- Build Loaders --------------------------------------------------------
train_loader = create_document_loader(
    train_texts,
    tokenizer,
    seq_len=cfg.seq_len,
    batch_size=cfg.batch_size,
    shuffle=True,
    drop_last=True,
    lazy=True,
    stride=cfg.seq_len,
)
val_loader = create_document_loader(
    val_texts,
    tokenizer,
    seq_len=cfg.seq_len,
    batch_size=cfg.batch_size,
    shuffle=False,
    drop_last=False,
    lazy=True,
    stride=cfg.seq_len,
)

# -- Model ----------------------------------------------------------------
print("Building model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HelixForCausalLM(cfg).to(device)
params = model.count_parameters()
print(f"Parameters: {params['total']:,} total, {params['trainable']:,} trainable")

# -- Optimizer ------------------------------------------------------------
optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))

# -- AMP Setup ------------------------------------------------------------
use_amp = True
amp_dtype = torch.bfloat16
scaler = None  # bf16 doesn't need GradScaler

# -- LR Schedule (manual step decay) --------------------------------------
def set_lr(epoch: int):
    """Epoch 1: 2e-3, Epoch 2: 1e-3, Epoch 3+: 3e-4"""
    if epoch == 1:
        lr = 2e-3
    elif epoch == 2:
        lr = 1e-3
    else:
        lr = 3e-4
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr

# -- Training -------------------------------------------------------------
grad_accum_steps = 2
effective_batch = cfg.batch_size * grad_accum_steps
epochs = 3

log_file = "/app/sandbox_helix/5m_probe_log.txt"
with open(log_file, "w") as f:
    f.write("epoch,step,loss,ppl,lr,tokens_seen,time,skipped_batches\n")

best_val_ppl = float("inf")
best_epoch = 0
all_nan = False

total_tokens = 0
for epoch in range(1, epochs + 1):
    epoch_lr = set_lr(epoch)
    print(f"\n{'='*60}")
    print(f"Epoch {epoch}/{epochs} | LR: {epoch_lr:.2e} | Effective batch: {effective_batch}")
    print(f"{'='*60}")

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
        total_tokens += input_ids.numel()

        # Forward
        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
                loss = outputs.loss
        else:
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            skipped += 1
            if skipped <= 5:
                print(f"  WARNING: NaN/Inf loss at batch {batch_idx}. Skipping.")
            continue

        # Gradient accumulation scaling
        divisor = grad_accum_steps
        if grad_accum_steps > 1:
            is_last = (batch_idx + 1) == len(train_loader)
            if is_last and accum_count < grad_accum_steps - 1:
                divisor = accum_count + 1
        loss = loss / divisor

        loss.backward()
        accum_count += 1
        epoch_loss += loss.item() * divisor
        raw_count += 1

        # Step
        is_last = (batch_idx + 1) == len(train_loader)
        if accum_count >= grad_accum_steps or is_last:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0

        # Logging every 50 steps
        if (batch_idx + 1) % 50 == 0 or is_last:
            avg = epoch_loss / max(raw_count, 1)
            ppl = math.exp(min(avg, 20))
            elapsed = time.time() - epoch_start
            tok_per_sec = tokens_seen / max(elapsed, 1e-6)
            print(
                f"  batch {batch_idx+1}/{len(train_loader)} | "
                f"loss={avg:.4f} ppl={ppl:.2f} lr={epoch_lr:.2e} "
                f"tok/s={tok_per_sec:.0f} skipped={skipped}"
            )

    train_avg = epoch_loss / max(raw_count, 1)
    train_ppl = math.exp(min(train_avg, 20))
    epoch_time = time.time() - epoch_start

    # Validation
    model.eval()
    val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            else:
                outputs = model(input_ids, labels=labels, attention_mask=attention_mask)

            vloss = outputs.loss
            if not (torch.isnan(vloss) or torch.isinf(vloss)):
                val_loss += vloss.item()
                val_batches += 1

    val_avg = val_loss / max(val_batches, 1)
    val_ppl = math.exp(min(val_avg, 20))

    print(f"\nEpoch {epoch} Summary:")
    print(f"  Train Loss: {train_avg:.4f} | Train PPL: {train_ppl:.2f}")
    print(f"  Val Loss:   {val_avg:.4f} | Val PPL:   {val_ppl:.2f}")
    print(f"  Time: {epoch_time:.1f}s | Tokens: {tokens_seen:,} | Skipped: {skipped}")

    with open(log_file, "a") as f:
        f.write(f"{epoch},train_end,{train_avg:.6f},{train_ppl:.2f},{epoch_lr},{tokens_seen},{epoch_time:.1f},{skipped}\n")
        f.write(f"{epoch},val_end,{val_avg:.6f},{val_ppl:.2f},{epoch_lr},{tokens_seen},{epoch_time:.1f},{skipped}\n")

    # Save best
    if val_ppl < best_val_ppl:
        best_val_ppl = val_ppl
        best_epoch = epoch
        save_path = "/app/sandbox_helix/best_model_5m"
        model.save_pretrained(save_path)
        print(f"  [BEST] Saved checkpoint to {save_path}")

    # NaN check
    if skipped > 0 and epoch == 1 and raw_count < 500:
        all_nan = True
        print("FATAL: NaN/Inf in first epoch before step 500 -- stopping.")
        break

print(f"\n{'='*60}")
print(f"Training complete. Best val PPL: {best_val_ppl:.2f} at epoch {best_epoch}")
print(f"Total tokens processed: {total_tokens:,}")
print(f"Target: val PPL <= 289.10")
print(f"{'='*60}")

# Final save
final_path = "/app/sandbox_helix/final_model_5m"
model.save_pretrained(final_path)
print(f"Final model saved to {final_path}")
