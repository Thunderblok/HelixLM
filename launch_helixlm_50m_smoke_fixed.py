"""
HelixLM 50M Token Smoke Test — Fixed Configuration
===================================================

Conservative but scaled-up config for 50M-token validation.
Uses FP32, fixed CCA with minimum scale, real pretraining data.

Expected: ~30-90 min on A10G/L40S. Target PPL < 120 after epoch 3.
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
                    default="david-thrower/HelixLM-small-50.0Mt-91250pt-7143it-20260427")
parser.add_argument("--d_model", type=int, default=256)
parser.add_argument("--n_loops", type=int, default=2)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--seq_len", type=int, default=512)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--cca_warmup", type=int, default=10000)
parser.add_argument("--cca_min_scale", type=float, default=0.05)
parser.add_argument("--output_dir", type=str, default="./checkpoints_50m_smoke_fixed")
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

# ── Config ───────────────────────────────────────────────────────────────
cfg = HelixConfig(
    vocab_size=vs,
    d_model=args.d_model,
    n_columns=2,
    nodes_per_column=(2, 2),
    n_heads=4,
    n_loops=args.n_loops,
    seq_len=args.seq_len,
    batch_size=args.batch_size,
    use_titans_memory=False,
    attention_mode="hybrid",
    dropout=0.05,
    lr=args.lr,
    weight_decay=args.wd,
    epochs=args.epochs,
    warmup_steps=200,
    grad_clip=1.0,
    device="auto",
    dtype="float32",  # FORCE FP32
    use_cca=True,
    cca_warmup_steps=args.cca_warmup,
    cca_ramp_mode="cubic_ease",
    cca_min_scale=args.cca_min_scale,
)
cfg.pad_token_id = tok.pad_token_id
cfg.eos_token_id = tok.eos_token_id

model = HelixForCausalLM(cfg)
params = model.count_parameters()["total"]
print(f"Params: {params:,}")

# ── torch.compile (validated in compile ablation) ─────────────────────────
if torch.cuda.is_available():
    print("\nEnabling torch.compile for speedup...")
    try:
        model = torch.compile(model, mode="default")
        print("✅ torch.compile enabled.")
    except Exception as e:
        print(f"⚠ torch.compile failed: {e}. Continuing eager mode.")

# ── DataLoaders ──────────────────────────────────────────────────────────
train_loader = create_document_loader(
    train_texts, tok, seq_len=args.seq_len, batch_size=args.batch_size,
    shuffle=True, drop_last=True, lazy=True,
)
val_loader = create_document_loader(
    val_texts, tok, seq_len=args.seq_len, batch_size=args.batch_size,
    shuffle=False, drop_last=False, lazy=True,
)

# ── Trainer (use_amp=False for FP32 stability) ───────────────────────────
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
print("HELIXLM 50M SMOKE TEST (FIXED)")
print(f"{'='*70}")
print(f"Config: d={args.d_model}, loops={args.n_loops}, lr={args.lr}, wd={args.wd}")
print(f"CCA: warmup={args.cca_warmup}, min_scale={args.cca_min_scale}, cubic_ease")
print(f"Dtype: float32 | AMP: OFF | Batch: {args.batch_size}")
print(f"Dataset: {args.dataset}")
print(f"{'='*70}\n")

history = trainer.train(num_epochs=args.epochs, eval_every=1)

# ── Results ──────────────────────────────────────────────────────────────
final_val_loss = history["val_loss"][-1] if history.get("val_loss") else float("inf")
final_ppl = math.exp(min(final_val_loss, 20))
final_train_loss = history["train_loss"][-1]

print(f"\n{'='*70}")
print("50M SMOKE TEST RESULTS")
print(f"{'='*70}")
print(f"Train loss: {final_train_loss:.4f}")
print(f"Val loss:   {final_val_loss:.4f}")
print(f"Val PPL:    {final_ppl:.2f}")
print(f"Params:     {params:,}")

if final_ppl < 80:
    print(f"\n🚀 GO FOR 400M! PPL={final_ppl:.2f} < 80")
elif final_ppl < 120:
    print(f"\n✓ PROMISING: PPL={final_ppl:.2f} < 120")
    print("   Try d=384 with same settings, or lr=1.5e-3")
elif final_ppl < 200:
    print(f"\n⚠ VIABLE: PPL={final_ppl:.2f} < 200")
    print("   Stable but needs hyperparameter tuning.")
else:
    print(f"\n✗ NEEDS WORK: PPL={final_ppl:.2f} >= 200")
    print("   Consider topology or data audit.")

# Save
os.makedirs(args.output_dir, exist_ok=True)
model.save_pretrained(f"{args.output_dir}/final_model")
with open(f"{args.output_dir}/results.json", "w") as f:
    json.dump({
        "train_loss": final_train_loss, "val_loss": final_val_loss,
        "val_ppl": final_ppl, "params": params, "history": history,
        "config": vars(args),
    }, f)

if args.push_to_hub and args.hub_model_id:
    print(f"Pushing to {args.hub_model_id}...")
    model.push_to_hub(args.hub_model_id)

print(f"\nCheckpoint saved to {args.output_dir}/")
