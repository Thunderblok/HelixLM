# Branch53 FFN Expansion V0

Branch53 changes one model-capacity knob relative to the admitted Branch52
runtime profile:

```text
ffn_expansion=2.5 -> 3.0
```

The following remain fixed:

```text
seq_len=512
batch_size=8
gradient_accumulation=8
n_loops=3
learning_rate=1.5e-4
scheduler=linear_warmup_then_constant
activation_checkpointing=true
target_causal_targets=400000000
```

The queued launcher refuses to start unless the exact Branch52 parent run has
exited and emitted a passing terminal with at least 400M causal targets, zero
nonfinite events, zero skipped batches, healthy checkpoints, healthy MLflow,
and healthy numerical status.

Manual verification:

```bash
python experiments/branch53-ffn-expansion-v0/test_branch53_ffn_expansion.py
bash -n experiments/branch53-ffn-expansion-v0/launch_after_branch52.sh
```
