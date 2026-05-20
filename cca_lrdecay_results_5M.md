# CCA × LR Schedule Factorial Ablation — Full 5M Tokens (L4)

**Date:** 2026-05-20  
**Branch:** `2026-05-20--cca-lrdecay-factorial-5M`  
**Dataset:** `david-thrower/HelixLM-tiny-5.0Mt-9125pt-715it-20260427`  
**Splits:** pretrain_train (8,942 docs), pretrain_val (183 docs)

## What We Tested

| Run | CCA | LR Schedule | Label |
|-----|-----|-------------|-------|
| 1 | OFF | Constant 2e-3 (3 epochs) | `baseline_constant` |
| 2 | ON (10% warmup = 342 steps) | Constant 2e-3 (3 epochs) | `cca_constant` |
| 3 | OFF | Two-stage: 2e-3 (epoch 1) → 3e-4 (epochs 2–3) | `lr2stage` |
| 4 | ON (10% warmup) | Two-stage: 2e-3 (epoch 1) → 3e-4 (epochs 2–3) | `cca_lr2stage` |
| 5 | OFF | Three-stage: 2e-3 → 1e-3 → 3e-4 (1 epoch each) | `lr3stage` |

**Fixed config (all runs):** `small_v2`, `n_loops=2`, `dropout=0.1`, `lr=2e-3` (stage 1), `batch_size=8`, `warmup_steps=50`, `grad_buffer_ratio=1/e`, `weight_decay=0.1`, `grad_clip=1.0`, `use_ssm=False`, `use_titans_memory=False`.

---

## Results

| Run | Label | CCA | LR Schedule | Val PPL | Val Loss | Train Loss | Time (s) |
|-----|-------|-----|-------------|---------|----------|------------|----------|
| 5 | **lr3stage** | OFF | 2e-3 → 1e-3 → 3e-4 | **289.10** | 5.6668 | 5.3032 | 1205 |
| 3 | lr2stage | OFF | 2e-3 → 3e-4 | 301.97 | 5.7103 | 5.3750 | 1188 |
| 1 | baseline_constant | OFF | constant 2e-3 | 310.80 | 5.7391 | 5.4875 | 1165 |
| 4 | cca_lr2stage | ON | 2e-3 → 3e-4 | 453.93 | 6.1179 | 5.5146 | 1191 |
| 2 | cca_constant | ON | constant 2e-3 | 460.17 | 6.1316 | 5.5210 | 1172 |

*Sorted by Val PPL (lower is better).*

---

## Answers to Key Questions

### 1. Does CCA help or hurt at this scale?

**HURT — significantly.**

- With constant LR: CCA-on (run 2) → **+48.1% WORSE** Val PPL than CCA-off baseline (460 vs 311)
- With two-stage LR: CCA-on (run 4) → **+50.3% WORSE** Val PPL than CCA-off (454 vs 302)
- The harm is **additive, not interactive**: CCA hurts roughly equally under both LR schedules (~48-50% worse).

This result is consistent with the Phase 2 finding, but now validated with proper 10% CCA warmup (342 steps, not 34-85). The 5M-token scale does not reverse the negative CCA effect.

### 2. Does staged LR decay beat constant LR?

**YES — both two-stage and three-stage beat constant LR.**

- Two-stage (run 3) vs constant (run 1): **-2.8% better** Val PPL (302 vs 311)
- Three-stage (run 5) vs constant (run 1): **-7.0% better** Val PPL (289 vs 311)
- Three-stage vs two-stage: **-4.3% better** Val PPL (289 vs 302)

The three-stage schedule (2e-3 → 1e-3 → 3e-4) produced the **best overall result** at 289.10 Val PPL — a modest but consistent improvement over both constant and two-stage LR.

---

## Takeaways

1. **CCA remains harmful** on the full 5M-token dataset with properly computed 10% warmup (342 steps). It is not a short-step-count artifact.
2. **Staged LR decay helps**, and finer granularity (three-stage > two-stage > constant) yields monotonic PPL improvement.
3. **Best config for Phase 4 onward**: `lr3stage` (three-stage decay) with CCA=OFF. This is the new winning recipe.
4. Runtime: ~20 min/run × 5 runs = ~100 min total on L4 (23 GB VRAM).

---

## Raw JSON

See `cca_lrdecay_results.json` for machine-readable full results.
