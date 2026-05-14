"""
HelixLM Mini Ablation — 1 epoch on 1K docs for fast validation
===============================================================

Runs 1 epoch on a 1K document subset to quickly validate:
1. No NaN/Inf
2. CCA gates open correctly  
3. Loss converges from ~10.8 down toward <8.0

Expected: ~10-15 min on CPU, ~2-3 min on GPU.
"""
import os, sys, math, random, json

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
val_texts = list(ds["pretrain_val"]["text"])[:100]

print(f"Train docs: {len(train_texts):,} | Val docs: {len(val_texts):,}")

# ── Tokenizer ───────────────────────────────────────────────────────────
tok = HelixTokenizer("gpt2")
vs = len(tok)
print(f"Vocab={vs}")

# ── Config ───────────────────────────────────────────────────────────────
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
    dtype="float32",
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

# ── Trainer ──────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model, cfg=cfg,
    train_loader=train_loader, val_loader=val_loader,
    tokenizer=tok,
    output_dir="./checkpoints_mini_ablation",
    example_prompts=["Once upon a time"],
    generated_example_length=10,
    grad_accum_steps=1,
    use_amp=False,
    verbose=True,
)
for group in trainer.optimizer.param_groups:
    group["betas"] = (0.9, 0.999)

# ── Train ───────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("HELIXLM MINI ABLATION (1K docs, 1 epoch)")
print(f"{'='*70}")

history = trainer.train(num_epochs=1, eval_every=1)

# ── Results ─────────────────────────────────────────────────────────────
final_val_loss = history["val_loss"][-1] if history.get("val_loss") else float("inf")
final_train_loss = history["train_loss"][-1]

print(f"\n{'='*70}")
print("MINI ABLATION RESULTS")
print(f"{'='*70}")
print(f"Train loss: {final_train_loss:.4f} (PPL={math.exp(min(final_train_loss,20)):.2f})")
print(f"Val loss:   {final_val_loss:.4f} (PPL={math.exp(min(final_val_loss,20)):.2f})")

# Save
os.makedirs("./checkpoints_mini_ablation", exist_ok=True)
model.save_pretrained("./checkpoints_mini_ablation/final_model")
with open("./checkpoints_mini_ablation/results.json", "w") as f:
    json.dump({
        "train_loss": final_train_loss, "val_loss": final_val_loss,
        "history": history,
    }, f)
