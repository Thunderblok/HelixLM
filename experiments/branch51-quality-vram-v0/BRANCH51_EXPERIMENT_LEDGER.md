# Branch51 Experiment Ledger

## 2026-08-30 setup

Objective:

```text
Create a clean Branch51 successor lane from Branch50 to test quality and VRAM knobs without mixing causes.
```

Starting branch:

```text
51-from-50-quality-vram-scaling
```

Baseline control reference:

```text
schema=helix.branch50.full-corpus-terminal-summary.v0
run_id=branch50-ablation-promoted-full-corpus-s512-b12-a7-t1501062500-20260829T142802Z
mlflow_run_id=98ecead0eaba4b438728d8ce51a926c5
best_validation_perplexity=47.378291172127675
final_validation_perplexity=47.47040872526028
optimizer_steps=34971
peak_vram_bytes=12655427072
```

Initial Branch51 factor set:

```text
CONTROL:
  batch12xaccum7
  ffn_expansion=2.5
  n_loops=3
  lr=1.5e-4
  scheduler=linear_warmup_then_constant

CANDIDATE_GEOMETRY_A:
  batch10xaccum6
  warmup_microbatches=1710
  warmup_optimizer_steps=285

CANDIDATE_GEOMETRY_B:
  batch8xaccum8
  warmup_microbatches=2280
  warmup_optimizer_steps=285

CANDIDATE_SCHEDULER:
  cosine_decay
  scheduler_min_lr_ratio=0.1

CANDIDATE_FFN:
  ffn_expansion=3.0

CANDIDATE_LOOPS:
  n_loops=2 or n_loops=4
```

Current rule:

```text
Non-control run must change exactly one factor family.
Promotion manifest required before combining winners.
```

Verification:

```text
python -m py_compile experiments/branch51-quality-vram-v0/run_branch51_quality_vram_ablation.py experiments/branch51-quality-vram-v0/test_branch51_quality_vram_controls.py
python experiments/branch51-quality-vram-v0/test_branch51_quality_vram_controls.py
```

## 2026-08-30 geometry smokes

Exact source:

```text
8a9d5988099d6307798595afd4ca03ca54b22945
```

All three 100-step runs passed numerical, checkpoint-readback, MLflow, and
source-custody courts with zero skipped batches and zero nonfinite events.

```text
12x7 control:
  peak_vram=12.04 GB
  step100_raw_tok_s=19,117.99
  mlflow=9e14dbe8bc414d5e913ca51c55d37789

10x6 candidate:
  peak_vram=10.20 GB
  step100_raw_tok_s=18,126.85
  mlflow=7acca60a8c524767a65e9a4dfd93a904

8x8 candidate:
  peak_vram=8.36 GB
  step100_raw_tok_s=16,737.16
  mlflow=b718c841f07f46dcb3101e4cabd13163
```

Decision:

```text
FIRST_FIXED_TOKEN_QUALITY_PILOT=batch10xaccum6

RATIONALE=
10x6 retains about 94.8% of control throughput while saving about 1.84 GB
of allocated VRAM. 8x8 is the lower-memory fallback, but pays a larger
throughput penalty. Startup validation PPL is not promotion evidence because
all three terminals occur inside the 285-step warmup.
```
