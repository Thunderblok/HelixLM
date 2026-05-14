"""
HelixLM 5M Token Smoke Test — HF Job Wrapper
=============================================
Identical to launch_helixlm_tiny_smoke_fixed.py but with:
  - Explicit HF_TOKEN for dataset download
  - Pushes results to Hub on completion

Runs 5M tokens, 3 epochs, FP32.
Target val PPL < 200 (ideally < 120).
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

# ── Dataset ─────────────────────────────────────────────────────────────
DATASET = "david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427"
SPLIT_TRAIN = "pretrain_train"
SPLIT_VAL = "pretrain_val"
TEXT_COL = "text"

print(f"Loading dataset: {DATASET}")
ds = load_dataset(DATASET)

train_texts = list(ds[SPLIT_TRAIN][TEXT_COL])
val_texts   = list(ds[SPLIT_VAL][TEXT_COL])

print(f"Train docs: {len(train_texts):,} | Val docs: {len(val_texts):,}")

# ── Tokenizer ───────────────────────────────────────────────────────────
tok = HelixTokenizer("gpt2")
vs = len(tok)
print(f"Vocab={vs}")

# ── Conservative Config ─────────────────────────────────────────────────
SEQ_LEN = 512
BATCH_SIZE = 8
EPOCHS = 3
LR = 1e-3
WD = 0.01
DROPOUT = 0.05
D_MODEL = 256
N_LOOPS = 2
N_COLUMNS = 2
NODES_PER_COLUMN = (2, 2)

CCA_WARMUP_STEPS = 10000
CCA_RAMP_MODE = "cubic_ease"
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
    lr=LR,
    weight_decay=WD,
    epochs=EPOCHS,
    warmup_steps=200,
    grad_clip=1.0,
    device="auto",
    dtype="float32",
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

# ── DataLoaders ─────────────────────────────────────────────────────────
train_loader = create_document_loader(
    train_texts, tok, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
    shuffle=True, drop_last=True, lazy=True,
)
val_loader = create_document_loader(
    val_texts, tok, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
    shuffle=False, drop_last=False, lazy=True,
)

# ── Trainer (FP32, no AMP) ──────────────────────────────────────────────
trainer = Trainer(
    model=model, cfg=cfg,
    train_loader=train_loader, val_loader=val_loader,
    tokenizer=tok,
    output_dir="./checkpoints_tiny_smoke_fixed",
    example_prompts=["Once upon a time", "The cat sat on the", "In 1492, Columbus"],
    generated_example_length=20,
    grad_accum_steps=1,
    use_amp=False,
    verbose=True,
)
for group in trainer.optimizer.param_groups:
    group["betas"] = (0.9, 0.999)

# ── Train ───────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("HELIXLM 5M SMOKE TEST (FP32)")
print(f"{'='*70}")
print(f"Config: d={D_MODEL}, loops={N_LOOPS}, lr={LR}, wd={WD}")
print(f"CCA: warmup={CCA_WARMUP_STEPS}, mode={CCA_RAMP_MODE}, min_scale={CCA_MIN_SCALE}")
print(f"Dtype: float32 | AMP: OFF | Batch: {BATCH_SIZE}")
print(f"Dataset: {DATASET}")
print(f"{'='*70}\n")

history = trainer.train(num_epochs=EPOCHS, eval_every=1)

# ── Results ─────────────────────────────────────────────────────────────
final_val_loss = history["val_loss"][-1] if history.get("val_loss") else float("inf")
final_ppl = math.exp(min(final_val_loss, 20))
final_train_loss = history["train_loss"][-1]

print(f"\n{'='*70}")
print("SMOKE TEST RESULTS")
print(f"{'='*70}")
print(f"Train loss: {final_train_loss:.4f}")
print(f"Val loss:   {final_val_loss:.4f}")
print(f"Val PPL:    {final_ppl:.2f}")
print(f"Params:     {params:,}")

# Decision gates
if final_ppl < 80:
    print(f"\n🚀 SHIP IT! PPL={final_ppl:.2f} < 80")
    print("   Ready for 400M production run with this config.")
elif final_ppl < 120:
    print(f"\n✓ PROMISING: PPL={final_ppl:.2f} < 120")
    print("   Scale to 50M tokens for tuning, then 400M.")
elif final_ppl < 200:
    print(f"\n⚠ VIABLE: PPL={final_ppl:.2f} < 200")
    print("   Config is stable but needs tuning (try lr=1.5e-3 or d=384 with fp32).")
else:
    print(f"\n✗ NEEDS WORK: PPL={final_ppl:.2f} >= 200")
    print("   Debug CCA/mask/topology further before scaling.")

# Save
os.makedirs("./checkpoints_tiny_smoke_fixed", exist_ok=True)
model.save_pretrained("./checkpoints_tiny_smoke_fixed/final_model")
with open("./checkpoints_tiny_smoke_fixed/results.json", "w") as f:
    json.dump({
        "train_loss": final_train_loss, "val_loss": final_val_loss,
        "val_ppl": final_ppl, "params": params, "history": history,
        "config": {
            "d_model": D_MODEL, "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
            "epochs": EPOCHS, "lr": LR, "wd": WD, "dropout": DROPOUT,
            "n_loops": N_LOOPS, "cca": True,
            "cca_warmup_steps": CCA_WARMUP_STEPS,
            "cca_ramp_mode": CCA_RAMP_MODE,
            "cca_min_scale": CCA_MIN_SCALE,
            "dtype": "float32", "use_amp": False,
            "dataset": DATASET,
        }
    }, f)

# Push to hub
try:
    model.push_to_hub("david-thrower/helixlm-tiny-smoke-fixed")
    print("Pushed to david-thrower/helixlm-tiny-smoke-fixed")
except Exception as e:
    print(f"Hub push skipped: {e}")

print(f"\nCheckpoint saved to ./checkpoints_tiny_smoke_fixed/")
