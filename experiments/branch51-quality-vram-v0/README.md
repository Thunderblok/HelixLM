# Branch51 Quality/VRAM Scaling

Branch51 starts from the Branch50 full-corpus winner and tests whether we can improve quality or memory efficiency without mixing causes.

The control reference is the Branch50 terminal evidence committed at:

```text
experiments/branch50-linear-context-v0/evidence/full-corpus-terminal-summary.json
```

Bound control facts:

```text
best_val_ppl=47.378291172127675
final_val_ppl=47.47040872526028
seq_len=512
batch_size=12
gradient_accumulation=7
ffn_expansion=2.5
n_loops=3
learning_rate=1.5e-4
scheduler=linear_warmup_then_constant
master_dtype=float32
amp_dtype=bfloat16
grad_buffer_ratio=0.0
```

## Experiment law

Each Branch51 non-control run changes exactly one declared factor family:

```text
optimizer_geometry:
  batch12xaccum7
  batch10xaccum6
  batch8xaccum8

scheduler:
  linear_warmup_then_constant
  cosine_decay + scheduler_min_lr_ratio

learning_rate

ffn_expansion

n_loops
```

Mixed changes are refused unless a signed Branch51 promotion manifest is supplied.

## First GPU smokes

Use short bounded smokes before a 100M or 300M candidate:

```bash
/home/mo/DEV/experiments/helix-branch49-5080-scaling-v0/.venv/bin/python \
  experiments/branch51-quality-vram-v0/run_branch51_quality_vram_ablation.py \
  --ablation-id geometry-b10-a6 \
  --batch-size 10 \
  --grad-accum 6 \
  --warmup-microbatches 1710 \
  --target-causal-targets 5000000 \
  --max-optimizer-steps 100 \
  --skip-shard-sha256
```

```bash
/home/mo/DEV/experiments/helix-branch49-5080-scaling-v0/.venv/bin/python \
  experiments/branch51-quality-vram-v0/run_branch51_quality_vram_ablation.py \
  --ablation-id geometry-b8-a8 \
  --batch-size 8 \
  --grad-accum 8 \
  --warmup-microbatches 2280 \
  --target-causal-targets 5000000 \
  --max-optimizer-steps 100 \
  --skip-shard-sha256
```

## Courts

```bash
/home/mo/DEV/experiments/helix-branch49-5080-scaling-v0/.venv/bin/python \
  experiments/branch51-quality-vram-v0/test_branch51_quality_vram_controls.py
```
