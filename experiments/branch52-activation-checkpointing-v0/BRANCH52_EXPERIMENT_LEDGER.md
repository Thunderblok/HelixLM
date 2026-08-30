# Branch52 Experiment Ledger

## 2026-08-30 — lane opened

```text
parent_head=d297a3c633f04751bc9e0a0f7af28e2751c47853
branch=52-from-51-activation-checkpointing
factor=activation_checkpointing
checkpoint_boundary=complete_recurrent_graph
checkpoint_mode=non_reentrant
branch51_mutated=false
```

Pre-GPU courts:

```text
loss_parity=PASS
gradient_parity=PASS
backward_recomputation_observed=PASS
disabled_mechanism_no_execution=PASS
branch51_control_courts=PASS
```

GPU promotion fields remain pending until the bounded smoke completes:

```text
peak_allocated_vram=pending
live_cuda_process_residency=pending
raw_tokens_per_second=pending
nonfinite_events=pending
```
