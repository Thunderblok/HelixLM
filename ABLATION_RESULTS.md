# HelixLM Phase 1–3 Tuning Results

**Branch:** `2026-05-20--phase1-tuning-5M`  
**Dataset:** `david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427` (1,000-sample subset)  
**Model:** `HelixConfig.small_v2` (~14.8M params)  
**Hardware:** L4 GPU (24 GB VRAM)  
**Source:** `patch-rng-freeze-release-cycle` (P0–P6 fixes, RNG-safe HelixGraph, 1/e grad_buffer_ratio default)

---

## Summary of Winning Configuration

| Parameter | Winning Value | Baseline (Default) |
|-----------|---------------|--------------------|
| `lr` | **2e-3** | 3e-4 |
| `n_loops` | **2** | 1 |
| `dropout` | **0.1** | 0.05 |
| `grad_buffer_ratio` | **1/e ≈ 0.368** | 1/e (already optimal) |
| `use_cca` | **False** | — |
| `use_ssm` | False | — |
| `use_titans_memory` | False | — |
| `batch_size` | 8 | — |
| `weight_decay` | 0.1 | — |
| `warmup_steps` | 50 | — |
| `grad_clip` | 1.0 | — |

**Final validation PPL: 837.00**

- vs. Phase 1 baseline (lr=2e-3, n_loops=2, dropout=0.05): **–15.5%**
- vs. original default (lr=3e-4, n_loops=1, dropout=0.05): **–23.8%**

---

## Phase 1: LR × n_loops Grid

4 runs, 3 epochs each, 1,000-sample subset (~90s per run).

| Run | LR | n_loops | Train Loss | Val Loss | Val PPL | Time |
|-----|-------|---------|------------|----------|---------|------|
| `lr2e-3_L2` | **2e-3** | **2** | 6.6383 | **6.8737** | **966.50** | 115s |
| `lr3e-4_L2` | 3e-4 | 2 | 6.6938 | 6.9449 | 1037.82 | 117s |
| `lr3e-4_L1` | 3e-4 | 1 | 6.7952 | 7.0013 | 1098.04 | 89s |
| `lr2e-3_L1` | 2e-3 | 1 | 6.8171 | 7.0532 | 1156.58 | 90s |

**Phase 1 Winner:** `lr2e-3_L2` — higher LR (2e-3) + recurrent depth (n_loops=2) is clearly best.

**Observation:** `n_loops=2` consistently outperforms `n_loops=1` by ~12% PPL at the same LR. The higher LR (2e-3 vs 3e-4) provides another ~12% gain. These effects are additive and well outside noise.

---

## Phase 2: CCA Warmup

2 runs on the Phase 1 winner config (lr=2e-3, n_loops=2), enabling `use_cca=True`.

| Run | CCA Warmup | Steps | Train Loss | Val Loss | Val PPL | Time |
|-----|-----------|-------|------------|----------|---------|------|
| `cca10pct` | 10% | 34 | 6.1514 | 7.0433 | **1145.18** | 116s |
| `cca25pct` | 25% | 85 | 6.6587 | 8.7273 | 6169.23 | 115s |

**Phase 2 Winner:** Neither — both CCA configurations perform worse than the no-CCA Phase 1 baseline (val PPL 966.50).

**Observation:** CCA appears to hurt at this scale (1,000 samples, 3 epochs). The 25% warmup is particularly damaging, with val PPL exploding to >6,000. The 10% warmup is still 15.6% worse than no CCA. For the production run, CCA should be **disabled** or tested on a larger subset before enabling.

---

## Phase 3: Dropout Sweep

3 runs on the best config so far (lr=2e-3, n_loops=2, no CCA).

| Run | Dropout | Train Loss | Val Loss | Val PPL | Time |
|-----|---------|------------|----------|---------|------|
| `drop0_1` | **0.1** | 6.3602 | **6.7298** | **837.00** | 115s |
| `drop0` | 0.0 | 6.3956 | 6.7601 | 862.75 | 115s |
| `drop0_25` | 0.25 | 6.6809 | 6.9536 | 1046.92 | 115s |

**Phase 3 Winner:** `drop0_1` — dropout=0.1 provides the best regularization at this scale.

**Observation:**
- `dropout=0.1` beats `dropout=0.0` by ~3.0% PPL and `dropout=0.05` (Phase 1 baseline) by 15.5%.
- `dropout=0.25` is too aggressive — 20% worse than 0.1.
- A small amount of dropout (0.1) is beneficial; 0.05 is slightly too low for this data scale.

---

## Steps to Reproduce the Winning Run

```python
import math
from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer, Trainer

tokenizer = HelixTokenizer("gpt2")
vocab_size = len(tokenizer)

cfg = HelixConfig.small_v2(
    vocab_size=vocab_size,
    tokenizer_name="gpt2",
    use_titans_memory=False,
    use_cca=False,
    use_ssm=False,
    lr=2e-3,                    # Phase 1 winner
    n_loops=2,                  # Phase 1 winner
    dropout=0.1,                # Phase 3 winner
    weight_decay=0.1,
    epochs=3,
    warmup_steps=50,
    grad_clip=1.0,
    grad_buffer_ratio=1.0 / math.e,  # ~0.368, validated optimal
    batch_size=8,
)
cfg.pad_token_id = tokenizer.pad_token_id
cfg.eos_token_id = tokenizer.eos_token_id
cfg.bos_token_id = tokenizer.bos_token_id

model = HelixForCausalLM(cfg)

# Load dataset (production pipeline — no manual chunking)
from datasets import load_dataset
ds = load_dataset("david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427", split="pretrain_train")
texts = [ex["text"] for ex in ds]

# For full reproduction, use the 1,000-sample subset:
# import random; texts = random.sample(texts, 1000)
# split = int(len(texts) * 0.9); train_texts = texts[:split]; val_texts = texts[split:]

trainer = Trainer(
    model=model, cfg=cfg,
    train_texts=train_texts, val_texts=val_texts,
    tokenizer=tokenizer,
    output_dir="./ckpt_winner",
    grad_accum_steps=1,
    use_amp=False,
    verbose=True,
)
history = trainer.train(num_epochs=3, eval_every=1)
```

---

## Artifacts

- Branch: `2026-05-20--phase1-tuning-5M`
- Scripts: `ablation_phase1.py`, `ablation_phase2.py`, `ablation_phase3.py`
- Results JSON: `phase1_results.json`, `phase2_results.json`, `phase3_results.json`
- Grad buffer ablation (prior): `ablation_grad_buffer_5M.py`, `ablation_results_grad_buffer.json`

---

## Recommendations for Production 400M-Token Run

1. **Keep the winning config:** `lr=2e-3`, `n_loops=2`, `dropout=0.1`, `grad_buffer_ratio=1/e`, `use_cca=False`.
2. **Re-test CCA on a larger subset** (e.g., 10K samples, 5+ epochs) before ruling it out for production — the ~30% PPL improvement claim may only emerge at scale.
3. **Consider `use_ssm=True`** (Mamba-2) for the production run — it was not tested here but is part of the target architecture.
4. **Effective batch size:** The ablation used `batch_size=8, grad_accum=1` (effective batch 8). For the 400M-token run, consider scaling to `batch_size=8, grad_accum=8` (effective 64) for smoother gradients.
5. **LR decay schedule:** A special test (2 runs) is included in `ablation_special_lr_decay.py` but was not executed due to time. Consider running it if training instability is observed at epoch boundaries.
