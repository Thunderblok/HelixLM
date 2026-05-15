"""
HelixLM 5M Token Smoke Test — Optimal Configuration (Regression Fix Applied)
==============================================================================

Uses the best config from Card A/B/C/D ablations:
  - d_model=384, n_loops=1 (higher dim beats more loops when attention works)
  - lr=2e-3, wd=0.05 (aggressive LR, strong decay)
  - CCA quadratic ramp over 5000 steps (gradual attention wake-up)
  - FP32 ONLY (BF16 banned after ablation failure)
  - torch.compile for speedup (validated in compile ablation)

Target: val PPL < 120 (Gate 2 unlock).
"""
import os, sys, math, random, json, time, argparse

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

# ── Args ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    default="david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427")
parser.add_argument("--output_dir", type=str, default="./checkpoints_5m_optimal")
parser.add_argument("--push_to_hub", action="store_true")
parser.add_argument("--hub_model_id", type=str, default="")
args = parser.parse_args()

# ── Dataset ──────────────────────────────────────────────────────────────
print(f"Loading dataset: {args.dataset}")
ds = load_dataset(args.dataset)

train_texts = list(ds["pretrain_train"]["text"])
val_texts = list(ds["pretrain_val"]["text"])

print(f"Train docs: {len(train_texts):,} | Val docs: {len(val_texts):,}")

# ── Tokenizer ────────────────────────────────────────────────────────────
tok = HelixTokenizer("gpt2")
vs = len(tok)
print(f"Vocab={vs}")

# ── Config: small model for 5M tokens to avoid overfitting ─────────────────
# 42M params on 5M tokens = severe overfitting. Use ~5M params instead.
SEQ_LEN = 512
BATCH_SIZE = 8
EPOCHS = 3
LR = 1e-3
WD = 0.1
DROPOUT = 0.1
ATTN_DROPOUT = 0.2
D_MODEL = 256
N_LOOPS = 2
N_COLUMNS = 3
NODES_PER_COLUMN = (2, 3, 2)

CCA_WARMUP_STEPS = 1000000  # Fully suppress attention for entire 5M run (>> total steps)
CCA_RAMP_MODE = "quadratic"
CCA_MIN_SCALE = 0.05

cfg = HelixConfig(
    vocab_size=vs,
    d_model=D_MODEL,
    n_columns=N_COLUMNS,
    nodes_per_column=NODES_PER_COLUMN,
    n_heads=4,
    n_loops=N_LOOPS,
    seq_len=SEQ_LEN,
    batch_size=BATCH_SIZE,
    use_titans_memory=False,
    attention_mode="hybrid",
    dropout=DROPOUT,
    attn_dropout=ATTN_DROPOUT,
    lr=LR,
    weight_decay=WD,
    epochs=EPOCHS,
    warmup_steps=200,
    grad_clip=1.0,
    device="auto",
    dtype="float32",  # FORCE FP32 — BF16 banned
    use_cca=True,
    cca_warmup_steps=CCA_WARMUP_STEPS,
    cca_ramp_mode=CCA_RAMP_MODE,
    cca_min_scale=CCA_MIN_SCALE,
)
cfg.pad_token_id = tok.pad_token_id
cfg.eos_token_id = tok.eos_token_id

model = HelixForCausalLM(cfg)
params = model.count_parameters()["total"]
print(f"Params: {params:,}")

# ── torch.compile DISABLED ─────────────────────────────────────────────
# torch.compile causes InductorError with heterogeneous graph.
print("\n⚠ torch.compile DISABLED — eager mode for graph stability.")

# ── DataLoaders ──────────────────────────────────────────────────────────
train_loader = create_document_loader(
    train_texts, tok, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
    shuffle=True, drop_last=True, lazy=True,
)
val_loader = create_document_loader(
    val_texts, tok, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
    shuffle=False, drop_last=False, lazy=True,
)

# ── Trainer (FP32, no AMP) ───────────────────────────────────────────────
trainer = Trainer(
    model=model, cfg=cfg,
    train_loader=train_loader, val_loader=val_loader,
    tokenizer=tok,
    output_dir=args.output_dir,
    example_prompts=["Once upon a time", "The cat sat on the", "In 1492, Columbus"],
    generated_example_length=20,
    grad_accum_steps=1,
    use_amp=False,  # ← DISABLED: Full FP32
    verbose=True,
)
for group in trainer.optimizer.param_groups:
    group["betas"] = (0.9, 0.999)

# ── Train ────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("HELIXLM 5M SMOKE TEST — OPTIMAL CONFIG")
print(f"{'='*70}")
print(f"Config: d={D_MODEL}, loops={N_LOOPS}, lr={LR}, wd={WD}")
print(f"CCA: warmup={CCA_WARMUP_STEPS}, mode={CCA_RAMP_MODE}, min_scale={CCA_MIN_SCALE}")
print(f"Dtype: float32 | AMP: OFF | Batch: {BATCH_SIZE}")
print(f"Attention dropout: {ATTN_DROPOUT} (higher than FFN dropout)")
print(f"Dataset: {args.dataset}")
print(f"{'='*70}\n")

history = trainer.train(num_epochs=EPOCHS, eval_every=1)

# ── Results ──────────────────────────────────────────────────────────────
final_val_loss = history["val_loss"][-1] if history.get("val_loss") else float("inf")
final_ppl = math.exp(min(final_val_loss, 20))
final_train_loss = history["train_loss"][-1]

print(f"\n{'='*70}")
print("5M SMOKE TEST RESULTS")
print(f"{'='*70}")
print(f"Train loss: {final_train_loss:.4f}")
print(f"Val loss:   {final_val_loss:.4f}")
print(f"Val PPL:    {final_ppl:.2f}")
print(f"Params:     {params:,}")

# Decision gates
if final_ppl < 80:
    print(f"\n🚀 GO FOR 400M! PPL={final_ppl:.2f} < 80")
elif final_ppl < 120:
    print(f"\n✓ PROMISING: PPL={final_ppl:.2f} < 120")
    print("   Scale to 50M tokens for tuning, then 400M.")
elif final_ppl < 200:
    print(f"\n⚠ VIABLE: PPL={final_ppl:.2f} < 200")
    print("   Stable but needs hyperparameter tuning.")
else:
    print(f"\n✗ NEEDS WORK: PPL={final_ppl:.2f} >= 200")
    print("   Debug CCA/mask/topology further before scaling.")

# Save
os.makedirs(args.output_dir, exist_ok=True)
model.save_pretrained(f"{args.output_dir}/final_model")
with open(f"{args.output_dir}/results.json", "w") as f:
    json.dump({
        "train_loss": final_train_loss, "val_loss": final_val_loss,
        "val_ppl": final_ppl, "params": params, "history": history,
        "config": {
            "d_model": D_MODEL, "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
            "epochs": EPOCHS, "lr": LR, "wd": WD, "dropout": DROPOUT,
            "attn_dropout": ATTN_DROPOUT,
            "n_loops": N_LOOPS, "cca": True,
            "cca_warmup_steps": CCA_WARMUP_STEPS,
            "cca_ramp_mode": CCA_RAMP_MODE,
            "cca_min_scale": CCA_MIN_SCALE,
            "dtype": "float32", "use_amp": False,
            "dataset": args.dataset,
        }
    }, f)

if args.push_to_hub and args.hub_model_id:
    print(f"Pushing to {args.hub_model_id}...")
    try:
        model.push_to_hub(args.hub_model_id)
        print("Push successful!")
    except Exception as e:
        print(f"Push failed: {e}")

print(f"\nCheckpoint saved to {args.output_dir}/")
