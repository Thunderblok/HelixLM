# Branch52 Activation Checkpointing

Branch52 inherits the verified Branch51 head `d297a3c` and changes one factor:
real non-reentrant activation checkpointing around the complete recurrent graph.

The first GPU court preserves the Branch50 optimizer geometry:

```text
batch_size=12
gradient_accumulation=7
effective_sequences=84
sequence_length=512
activation_checkpointing=true
```

The run is admissible only when requested, instantiated, and executed evidence
agree. Execution requires backward recomputation calls to exceed forward calls.

Bounded smoke:

```bash
HELIX_BRANCH52_RUN_ROOT=/home/mo/DEV/experiments/helix-branch52-activation-checkpointing-v0 \
/home/mo/DEV/experiments/helix-branch49-5080-scaling-v0/.venv/bin/python \
  experiments/branch51-quality-vram-v0/run_branch51_quality_vram_ablation.py \
  --ablation-id activation-checkpointing-b12-a7 \
  --batch-size 12 \
  --grad-accum 7 \
  --warmup-microbatches 2000 \
  --activation-checkpointing \
  --target-causal-targets 5000000 \
  --max-optimizer-steps 100 \
  --skip-shard-sha256
```

Promotion requires numerical parity, observed recomputation, a lower live CUDA
residency, finite loss/gradients, and an acceptable throughput tradeoff.
