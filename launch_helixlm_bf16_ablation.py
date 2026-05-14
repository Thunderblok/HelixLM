"""
HelixLM BF16 Numerical Stability Ablation
==========================================
Tests whether the fixed topology (mask propagation, CCA gate +2.0,
min_scale=0.05) trains stably with torch.bfloat16 AMP without NaN/Inf.

Uses mini dataset (1K docs, 1 epoch) for fast feedback.
Expected: ~2-5 min on L4 GPU.
"""
import os, sys, math, random, json, time

SEED = 42
random.seed(SEED)

import torch
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)

from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer
from helix_lm.trainer import Trainer
from helix_lm.dataset import create_document_loader
from datasets import load_dataset

# ── Dataset (1K subset) ─────────────────────────────────────────────────
DATASET = "david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427"
print(f"Loading dataset: {DATASET}")
ds = load_dataset(DATASET)

train_texts = list(ds["pretrain_train"]["text"])[:1000]
val_texts   = list(ds["pretrain_val"]["text"])[:100]

print(f"Train docs: {len(train_texts):,} | Val docs: {len(val_texts):,}")

# ── Tokenizer ───────────────────────────────────────────────────────────
tok = HelixTokenizer("gpt2")
vs = len(tok)
print(f"Vocab={vs}")

# ── Config ── SAME as winning mini ablation, but BF16 + AMP ────────────
cfg = HelixConfig(
    vocab_size=vs,
    d_model=256,
    n_columns=2,
    nodes_per_column=(2, 2),
    n_heads=4,
    n_loops=2,
    seq_len=512,
    batch_size=8,
    use_titans_memory=False,
    attention_mode="hybrid",
    dropout=0.05,
    lr=1e-3,
    weight_decay=0.01,
    epochs=1,
    warmup_steps=200,
    grad_clip=1.0,
    device="auto",
    dtype="bfloat16",          # ← BF16 weights
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

# ── DataLoaders ─────────────────────────────────────────────────────────
train_loader = create_document_loader(
    train_texts, tok, seq_len=512, batch_size=8,
    shuffle=True, drop_last=True, lazy=True,
)
val_loader = create_document_loader(
    val_texts, tok, seq_len=512, batch_size=8,
    shuffle=False, drop_last=False, lazy=True,
)

# ── Trainer with AMP=ON ────────────────────────────────────────────────
trainer = Trainer(
    model=model, cfg=cfg,
    train_loader=train_loader, val_loader=val_loader,
    tokenizer=tok,
    output_dir="./checkpoints_bf16_ablation",
    example_prompts=["Once upon a time"],
    generated_example_length=10,
    grad_accum_steps=1,
    use_amp=True,              # ← ENABLE AMP (fp16 autocast, fp32 master weights)
    verbose=True,
)
for group in trainer.optimizer.param_groups:
    group["betas"] = (0.9, 0.999)

# ── Train ───────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("HELIXLM BF16 ABLATION (1K docs, 1 epoch, AMP=ON, dtype=bf16)")
print(f"{'='*70}")

start = time.time()
history = trainer.train(num_epochs=1, eval_every=1)
elapsed = time.time() - start

# ── Results ─────────────────────────────────────────────────────────────
final_val_loss = history["val_loss"][-1] if history.get("val_loss") else float("inf")
final_train_loss = history["train_loss"][-1]

print(f"\n{'='*70}")
print("BF16 ABLATION RESULTS")
print(f"{'='*70}")
print(f"Train loss: {final_train_loss:.4f} (PPL={math.exp(min(final_train_loss,20)):.2f})")
print(f"Val loss:   {final_val_loss:.4f} (PPL={math.exp(min(final_val_loss,20)):.2f})")
print(f"Time:       {elapsed:.1f}s")
print(f"NaN/Inf skipped batches: {history.get('skipped_batches', 'N/A')}")

# Decision
val_ppl = math.exp(min(final_val_loss, 20))
stable = (val_ppl < 500) and not math.isnan(final_val_loss) and not math.isinf(final_val_loss)
print(f"\n{'='*70}")
if stable:
    print(f"✅ BF16 STABLE — val PPL={val_ppl:.2f}. Safe for 400M production run.")
else:
    print(f"❌ BF16 UNSTABLE — val PPL={val_ppl:.2f}. Stay FP32 for 5M/50M/400M.")
print(f"{'='*70}")

# Save
os.makedirs("./checkpoints_bf16_ablation", exist_ok=True)
model.save_pretrained("./checkpoints_bf16_ablation/final_model")
with open("./checkpoints_bf16_ablation/results.json", "w") as f:
    json.dump({
        "train_loss": final_train_loss, "val_loss": final_val_loss,
        "val_ppl": val_ppl, "time_sec": elapsed,
        "stable": stable, "history": history,
        "config": {"dtype": "bfloat16", "use_amp": True},
    }, f)
