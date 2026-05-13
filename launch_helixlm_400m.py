"""
HF Job launcher for HelixLM 400M token production run.
Clones repo, installs deps, runs train_production_cca.py with trackio logging.
"""
import os, subprocess, sys, math, json



# ── Import HelixLM ─────────────────────────────────────────────────────
from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer
from helix_lm.trainer import Trainer
from helix_lm.dataset import create_document_loader
from datasets import load_dataset
import torch
import random

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Config (Card A3 winner + CCA, configs were set on 96 seq len, tiny stories, need re-optimized) 
SEQ_LEN = 512
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-3
WD = 0.05
DROPOUT = 0.05
D_MODEL = 384
N_LOOPS = 1
N_COLUMNS = 2

DATASET = "david-thrower/HelixLM-tiny-400.0Mt-730000pt-57143it-20260430"
SPLIT_TRAIN = "pretrain_train"
SPLIT_VAL = "pretrain_val"
TEXT_COL = "text"

print(f"Loading dataset: {DATASET}")
ds = load_dataset(DATASET)

train_texts = list(ds[SPLIT_TRAIN][TEXT_COL])
val_texts = list(ds[SPLIT_VAL][TEXT_COL])

print(f"Train docs: {len(train_texts):,} | Val docs: {len(val_texts):,}")

tok = HelixTokenizer("gpt2")
vs = len(tok)
print(f"Vocab={vs}")

cfg = HelixConfig(
    vocab_size=vs, d_model=D_MODEL, n_columns=N_COLUMNS,
    nodes_per_column=(2, 2),
    n_heads=4, n_loops=N_LOOPS, seq_len=SEQ_LEN,
    batch_size=BATCH_SIZE,
    use_titans_memory=False, attention_mode="hybrid", dropout=DROPOUT,
    lr=LR, weight_decay=WD, epochs=EPOCHS,
    warmup_steps=200, grad_clip=1.0,
    device="auto",
    use_cca=True, cca_warmup_steps=5000, cca_ramp_mode="quadratic",
)
cfg.pad_token_id = tok.pad_token_id
cfg.eos_token_id = tok.eos_token_id

model = HelixForCausalLM(cfg)
params = model.count_parameters()["total"]
print(f"Params: {params:,}")


train_loader = create_document_loader(
    train_texts, tok, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
    shuffle=True, drop_last=True, lazy=True,
)
val_loader = create_document_loader(
    val_texts, tok, seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
    shuffle=False, drop_last=False, lazy=True,
)

trainer = Trainer(
    model=model, cfg=cfg,
    train_loader=train_loader, val_loader=val_loader,
    tokenizer=tok,
    output_dir="./checkpoints_400M",
    example_prompts=["Once upon a time", "The cat sat on the"],
    generated_example_length=20,
    grad_accum_steps=1,
    use_amp=torch.cuda.is_available(),
    verbose=True,
)
for group in trainer.optimizer.param_groups:
    group["betas"] = (0.9, 0.999)


history = trainer.train(num_epochs=EPOCHS, eval_every=1)

final_val_loss = history["val_loss"][-1] if history.get("val_loss") else float("inf")
final_ppl = math.exp(min(final_val_loss, 20))
final_train_loss = history["train_loss"][-1]

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Train loss: {final_train_loss:.4f}")
print(f"Val loss:   {final_val_loss:.4f}")
print(f"Val PPL:    {final_ppl:.2f}")
print(f"Params:     {params:,}")

# Save
os.makedirs("./checkpoints_400M", exist_ok=True)
model.save_pretrained("./checkpoints_400M/final_model")
with open("./checkpoints_400M/results.json", "w") as f:
    json.dump({
        "train_loss": final_train_loss, "val_loss": final_val_loss,
        "val_ppl": final_ppl, "params": params, "history": history,
        "config": {
            "d_model": D_MODEL, "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
            "epochs": EPOCHS, "lr": LR, "wd": WD, "dropout": DROPOUT,
            "n_loops": N_LOOPS, "cca": True, "cca_warmup": 5000,
        }
    }, f)

# Push to hub
HUB_ID = "david-thrower/HelixLM-384d-cca-400Mt-prod"
print(f"Pushing to {HUB_ID}...")
try:
    model.push_to_hub(HUB_ID)
    print("Push successful!")
except Exception as e:
    print(f"Push failed: {e}")

print(f"\n{'='*60}")
if final_ppl < 80:
    print(f"SHIP IT! PPL={final_ppl:.2f} < 80")
elif final_ppl < 120:
    print(f"PRODUCTION GATE OPEN: PPL={final_ppl:.2f} < 120")
elif final_ppl < 160:
    print(f"PROMISING: PPL={final_ppl:.2f}")
else:
    print(f"NEEDS WORK: PPL={final_ppl:.2f}")
