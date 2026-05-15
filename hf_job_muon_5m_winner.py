"""
HF Job: Muon Winner Config at 5M Scale
=======================================

Runs the best config from Card A (A3 + Muon hybrid) at d=256 on 5M tokens.
Uses the full 5M dataset with proper CCA warmup.

Submit via:
  python hf_job_muon_5m_winner.py

Or via HF Job:
  apt-get update && apt-get install -y git
  git clone -b agent-2026-05-16-muon-ablation \
    https://github.com/david-thrower/HelixLM.git /tmp/HelixLM
  cd /tmp/HelixLM && pip install -r requirements.txt
  python hf_job_muon_5m_winner.py

Hardware: l4x1 (minimum) or a10g-small / t4-small
Expected: ~30-60 min. Target Val PPL < 200 (Gate 2).
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

from helix_lm import (
    HelixConfig, HelixForCausalLM, HelixTokenizer,
    Trainer, build_hybrid_muon_adamw,
)
from helix_lm.dataset import create_document_loader
from datasets import load_dataset

# ── Args ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    default="david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427")
parser.add_argument("--output_dir", type=str, default="./checkpoints_muon_5m")
parser.add_argument("--d_model", type=int, default=256)
parser.add_argument("--n_loops", type=int, default=1)
parser.add_argument("--lr", type=float, default=2e-3)
parser.add_argument("--wd", type=float, default=0.05)
parser.add_argument("--dropout", type=float, default=0.05)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--seq_len", type=int, default=512)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--cca_warmup", type=int, default=1000000)
parser.add_argument("--cca_min_scale", type=float, default=0.05)
parser.add_argument("--use_muon", action="store_true", default=True)
parser.add_argument("--no_muon", action="store_true", help="Use AdamW only (baseline)")
args = parser.parse_args()

use_muon = not args.no_muon
opt_name = "muon" if use_muon else "adamw"

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

# ── Config ───────────────────────────────────────────────────────────────
# Winning config from Card A: A3 (d=256, loops=1, lr=2e-3, wd=0.05, do=0.05)
cfg = HelixConfig(
    vocab_size=vs,
    d_model=args.d_model,
    n_columns=3,
    nodes_per_column=(2, 3, 2),
    n_heads=4,
    n_loops=args.n_loops,
    seq_len=args.seq_len,
    batch_size=args.batch_size,
    use_titans_memory=False,
    attention_mode="hybrid",
    dropout=args.dropout,
    attn_dropout=min(args.dropout + 0.1, 0.25),
    lr=args.lr,
    weight_decay=args.wd,
    epochs=args.epochs,
    warmup_steps=200,
    grad_clip=1.0,
    device="auto",
    dtype="float32",  # FORCE FP32
    use_cca=True,
    cca_warmup_steps=args.cca_warmup,
    cca_ramp_mode="quadratic",
    cca_min_scale=args.cca_min_scale,
)
cfg.pad_token_id = tok.pad_token_id
cfg.eos_token_id = tok.eos_token_id

model = HelixForCausalLM(cfg)
params = model.count_parameters()["total"]
print(f"Params: {params:,}")

# ── torch.compile ────────────────────────────────────────────────────────
if torch.cuda.is_available():
    print("\nEnabling torch.compile for speedup...")
    try:
        model = torch.compile(model, mode="default")
        print("torch.compile enabled.")
    except Exception as e:
        print(f"torch.compile failed: {e}. Continuing eager mode.")

# ── DataLoaders ──────────────────────────────────────────────────────────
train_loader = create_document_loader(
    train_texts, tok, seq_len=args.seq_len, batch_size=args.batch_size,
    shuffle=True, drop_last=True, lazy=True,
)
val_loader = create_document_loader(
    val_texts, tok, seq_len=args.seq_len, batch_size=args.batch_size,
    shuffle=False, drop_last=False, lazy=True,
)

# ── Configure Optimizer ──────────────────────────────────────────────────
optimizer = None
if use_muon:
    print("\nUsing Muon hybrid optimizer...")
    optimizer = build_hybrid_muon_adamw(model, lr=cfg.lr, weight_decay=cfg.weight_decay)
    muon_p_n = sum(p.numel() for g in optimizer[0].param_groups for p in g["params"])
    adam_p_n = sum(p.numel() for g in optimizer[1].param_groups for p in g["params"])
    print(f"  Muon params: {muon_p_n:,} ({muon_p_n/params*100:.1f}%)")
    print(f"  AdamW params: {adam_p_n:,} ({adam_p_n/params*100:.1f}%)")
else:
    print("\nUsing AdamW only (baseline)...")

# ── Trainer ──────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model, cfg=cfg,
    train_loader=train_loader, val_loader=val_loader,
    tokenizer=tok,
    output_dir=args.output_dir,
    example_prompts=["Once upon a time", "The cat sat on the", "In 1492, Columbus"],
    generated_example_length=20,
    grad_accum_steps=1,
    use_amp=False,
    verbose=True,
    optimizer=optimizer,
)

# ── Train ────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"HELIXLM 5M with {opt_name.upper()} — WINNER CONFIG FROM CARD A")
print(f"{'='*70}")
print(f"Config: d={args.d_model}, loops={args.n_loops}, lr={args.lr}, wd={args.wd}")
print(f"CCA: warmup={args.cca_warmup}, min_scale={args.cca_min_scale}")
print(f"Optimizer: {'Muon hybrid (2D matrices + AdamW for rest)' if use_muon else 'AdamW only'}")
print(f"Dtype: float32 | AMP: OFF | Batch: {args.batch_size}")
print(f"Dataset: {args.dataset}")
print(f"{'='*70}\n")

history = trainer.train(num_epochs=args.epochs, eval_every=1)

# ── Results ──────────────────────────────────────────────────────────────
final_val_loss = history["val_loss"][-1] if history.get("val_loss") else float("inf")
final_ppl = math.exp(min(final_val_loss, 20))
final_train_loss = history["train_loss"][-1]

print(f"\n{'='*70}")
print(f"5M RESULTS with {opt_name.upper()}")
print(f"{'='*70}")
print(f"Train loss: {final_train_loss:.4f}")
print(f"Val loss:   {final_val_loss:.4f}")
print(f"Val PPL:    {final_ppl:.2f}")
print(f"Params:     {params:,}")

if final_ppl < 120:
    print(f"\n🚀 GO FOR 50M! PPL={final_ppl:.2f} < 120 (Gate 2 passed)")
elif final_ppl < 200:
    print(f"\n✓ VIABLE: PPL={final_ppl:.2f} < 200 (Gate 2 acceptable)")
else:
    print(f"\n⚠ NEEDS WORK: PPL={final_ppl:.2f} >= 200")

# Save
os.makedirs(args.output_dir, exist_ok=True)
model.save_pretrained(f"{args.output_dir}/final_model")
with open(f"{args.output_dir}/results.json", "w") as f:
    json.dump({
        "train_loss": final_train_loss, "val_loss": final_val_loss,
        "val_ppl": final_ppl, "params": params, "history": history,
        "config": {
            "d_model": args.d_model, "n_loops": args.n_loops,
            "lr": args.lr, "wd": args.wd, "dropout": args.dropout,
            "batch_size": args.batch_size, "seq_len": args.seq_len,
            "epochs": args.epochs, "optimizer": opt_name,
            "cca_warmup": args.cca_warmup, "cca_min_scale": args.cca_min_scale,
            "dtype": "float32", "use_amp": False,
            "dataset": args.dataset,
        }
    }, f)

print(f"\nCheckpoint saved to {args.output_dir}/")
