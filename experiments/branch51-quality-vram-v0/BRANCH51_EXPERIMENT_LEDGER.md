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

CANDIDATE_GEOMETRY_B:
  batch7xaccum13

CANDIDATE_GEOMETRY_C:
  batch12xaccum9

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
