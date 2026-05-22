# d_model × FFN Expansion Factorial — 5M Tokens, seq_len=128

**Date:** 2026-05-22  
**Branch:** `2026-05-22--dmodel-ffn-factorial-5M`  
**Source:** `2026-05-21--18-51--fix-bf16-dtype-compat-plus-seed-fix`  
**GPU:** NVIDIA L4  
**AMP:** bfloat16

## Executive Summary

| Run | d_model | n_heads | ffn | Params | Final Val PPL | Rank | Time |
|-----|---------|---------|-----|--------|---------------|------|------|
| **d256_f2** | 256 | 4 | 2.0 | 14.8M | **268.50** | 🥇 | 782s |
| d320_f2 | 320 | 5 | 2.0 | 19.1M | 288.23 | 🥈 | 910s |
| d256_f3 | 256 | 4 | 3.0 | 15.2M | 310.84 | 🥉 | 795s |
| d320_f3 | 320 | 5 | 3.0 | 19.8M | 314.13 | 4th | 919s |

**Key finding:** At 5M tokens and seq_len=128, **narrower (d=256) + shallower FFN (ffn=2.0) decisively wins.** Wider models (d=320) and deeper FFNs (ffn=3.0) both underperform. The recurrent architecture with n_loops=2 already provides sufficient effective FFN capacity; adding more FFN parameters is wasted at this token budget.

---

## Per-Run Stage Breakdown

### d256_f2 (Winner) — 14.8M params
| Stage | LR | Train PPL | Val PPL | Time |
|-------|----|-----------|---------|------|
| 1 | 2e-3 | 467.8 | 338.44 | 241s |
| 2 | 1e-3 | 218.3 | 283.46 | 242s |
| 3 | 3e-4 | 157.5 | **268.50** | 242s |

**Gap train→val:** 111 PPL (41%). Significant overfitting at this scale, but manageable.

### d320_f2 — 19.1M params (+29% params vs d256_f2)
| Stage | LR | Train PPL | Val PPL | Time |
|-------|----|-----------|---------|------|
| 1 | 2e-3 | 566.2 | 370.13 | 281s |
| 2 | 1e-3 | 237.3 | 305.68 | 282s |
| 3 | 3e-4 | 167.8 | **288.23** | 282s |

**vs d256_f2:** +7.3% PPL worse despite 29% more parameters. Wider model is **data-starved**.

### d256_f3 — 15.2M params (+2.6% params vs d256_f2)
| Stage | LR | Train PPL | Val PPL | Time |
|-------|----|-----------|---------|------|
| 1 | 2e-3 | 649.0 | 437.88 | 243s |
| 2 | 1e-3 | 292.3 | 337.80 | 244s |
| 3 | 3e-4 | 204.9 | **310.84** | 245s |

**vs d256_f2:** +15.8% PPL worse! FFN expansion 3.0 **hurts** at this scale. The extra FFN parameters steal capacity from what the model can afford to learn.

### d320_f3 — 19.8M params (+33% vs d256_f2)
| Stage | LR | Train PPL | Val PPL | Time |
|-------|----|-----------|---------|------|
| 1 | 2e-3 | 596.8 | 410.56 | 284s |
| 2 | 1e-3 | 268.5 | 335.97 | 285s |
| 3 | 3e-4 | 191.7 | **314.13** | 285s |

**vs d256_f2:** +17.0% PPL worse. Double-whammy: wider AND deeper FFN both lose.

---

## WeightWatcher Spectral Analysis

| Run | alpha | alpha_weighted | log_norm | Interpretation |
|-----|-------|----------------|----------|----------------|
| d256_f2 | **3.262** | **1.770** | 2.322 | Best regularized, most power-law-like |
| d256_f3 | 3.258 | 1.521 | 2.331 | Slightly higher log_norm, still well-regularized |
| d320_f2 | 3.264 | 1.690 | **2.461** | Wider model → higher log_norm (more parameter mass) |
| d320_f3 | **3.595** | 1.682 | **2.488** | **Highest alpha = worst regularization** |

**Theory (WeightWatcher / Martin-Mahoney):**  
- alpha ≈ 2–4 indicates well-trained, power-law-like weight matrices (good).  
- alpha > 4 indicates under-trained / over-regularized (bad).  
- alpha < 2 indicates over-trained / correlated weights (bad).  

All runs have alpha in the 3.26–3.60 range, which is acceptable but d320_f3 at 3.60 is approaching the boundary of concern. The **log_norm** metric measures total weight magnitude; d320 models have higher log_norm because they simply have more parameters — this is expected. The alpha values confirm that d256_f2 is the most "natural" (power-law) weight distribution.

**No disparity in saturation ratios** was detected that would warrant targeted regularization changes. All alphas are in the healthy 3.2–3.6 band; the performance differences come from **capacity vs data mismatch**, not from weight matrix pathology.

---

## Answers to Open Questions

### Q1: Does ffn_expansion=3.0 help?

**No. It hurts by 15–17% PPL at 5M tokens.**

With n_loops=2, the effective FFN is already applied twice per forward pass. Adding ffn_expansion=3.0 increases FFN capacity from 0.8M to 1.2M (d256) or 1.2M to 1.8M (d320), but those parameters are **severely underutilized** at 5M tokens. The recurrent depth already provides sufficient non-linear transformations.

**Production implication:** At 400M tokens, ffn_expansion=3.0 may become competitive — but this ablation does NOT support switching to 3.0 now. If anything, it reinforces keeping ffn_expansion=2.0 as the default for small/medium scales.

### Q2: Does d=320 earn its extra 4.3M params at seq_len=128?

**No. It loses by 7% PPL despite 29% more parameters.**

At 5M tokens, d=320 is severely data-starved. The Chinchilla-adjusted target for d=320 (19.1M params, n_loops=2) is ~573–764M tokens — we're at only 5M, which is **~1%** of optimal. The embedding table at d=320 consumes 16.1M (84% of params), leaving even less budget for the actual network to learn.

**Critical caveat:** This does NOT contradict the 400M-token projection. At 400M, d=320 will almost certainly close and possibly surpass d=256. This ablation tests architectural allocation at a tiny scale — the finding is: **don't widen before you have the tokens to feed it.**

---

## Parameter Allocation Analysis

```
d256_f2 (winner): embed=12.9M (87%)  attn=0.5M (3%)  ffn=0.8M (5%)  other=0.6M
d256_f3:          embed=12.9M (85%)  attn=0.5M (3%)  ffn=1.2M (8%)  other=0.6M
d320_f2:          embed=16.1M (84%)  attn=0.8M (4%)  ffn=1.2M (6%)  other=0.7M
d320_f3:          embed=16.1M (77%)  attn=0.8M (4%)  ffn=1.8M (9%)  other=0.7M
```

The embedding dominates everywhere (77–87%). Shifting 3–4% of param budget from embedding to FFN (via ffn=3.0) **does not help** when the total budget is so small. The FFN expansion essentially steals capacity that the embedding table could use to better represent the vocabulary.

---

## Recommendations Before 50M/400M Production Run

1. **Keep d_model=256, ffn_expansion=2.0 for 50M sanity check.** These results strongly validate the current default. Changing either dimension before scaling data is premature optimization.

2. **Do NOT increase ffn_expansion to 3.0 at 5M or 50M.** Wait until 200M+ tokens before revisiting.

3. **Consider d=320 for 400M run ONLY if Chinchilla budget is met.** At 400M tokens, d=320 (19.1M) is at ~21 tokens/param (vanilla) or ~21 adjusted (n_loops=2). That's still below the 30–40 target but closer. d=256 (14.8M) is at ~27 tokens/param — closer to optimal. For 400M, **d=256 remains the safer choice.**

4. **WeightWatcher confirms no need for targeted regularization changes.** All alphas are healthy (~3.26). Current dropout=0.1 and weight_decay=0.1 are appropriate.

5. **d=384 projection revised further down:** At 23.6M params, d=384 wants ~708–944M tokens. At 400M, it would be at ~17 tokens/param — severely underfed. Projected PPL at 400M: **~100–130** (not 70–95). Do NOT use d=384 for 400M unless you can secure 700M+ tokens.

6. **Next ablation before 400M:** Consider testing `seq_len=128 vs 256` on 50M tokens to verify that 128 remains optimal. Longer sequences reduce sample count but may improve context learning. The 50M sanity check is the right place to validate this.

---

## Raw Results JSON

See `dmodel_ffn_results.json` in this branch for full per-stage, per-metric data including WeightWatcher summaries.
