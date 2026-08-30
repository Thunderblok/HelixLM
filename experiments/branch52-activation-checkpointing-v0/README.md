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

## Bounded GPU result

Both 100-step courts passed with activation checkpointing requested,
instantiated, and executed. The `8x8` geometry achieved the Branch52 target:

```text
batch_size=8
gradient_accumulation=8
peak_pytorch_allocated=6493904384 bytes (6.05 GiB)
trainer_process_residency=7952 MiB (7.77 GiB)
strict_8_GiB_process_margin=240 MiB
raw_tokens_per_second=12435.888945826424
validation_loss=8.559529840946198
validation_ppl=5216.228138952979
nonfinite_events=0
skipped_batches=0
```

The validation result exactly matches the uncheckpointed Branch51 `8x8`
100-step court. Activation checkpointing saved `1,870,669,312` allocated bytes
at a `25.70%` raw-throughput cost. This establishes numerical parity and the
memory result for the bounded smoke; it does not establish fixed-token quality.

The under-8-GiB result applies to the trainer process. Total card usage was
`8876 MiB` because the display, browser, and editor also held VRAM; no claim is
made that whole-card residency was below 8 GiB.

Next gate: a fixed-token 100M `8x8` Branch52 pilot before any longer campaign.
