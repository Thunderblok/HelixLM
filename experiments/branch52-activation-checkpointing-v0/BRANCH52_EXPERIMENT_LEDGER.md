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

## 2026-08-30 — bounded GPU courts

Branch50 geometry preservation court:

```text
geometry=12x7
optimizer_steps=100
status=SMOKE_PASS
peak_allocated_vram_bytes=9193714176
live_cuda_process_residency_mib=11456
raw_tokens_per_second=14527.847531663589
validation_loss=8.536221027374268
validation_ppl=5096.050094549297
checkpoint_forward_calls=700
checkpoint_function_calls=1400
checkpoint_recompute_calls=700
nonfinite_events=0
skipped_batches=0
mlflow_run_id=0543fb4cabf64475acae5d05be93e4f6
terminal_sha256=1156f76f752260f328cc392df41fd7062be289c2b47e15f19632498551695fbc
```

Strict process-residency court:

```text
geometry=8x8
optimizer_steps=100
status=SMOKE_PASS
peak_allocated_vram_bytes=6493904384
peak_allocated_vram_gib=6.04792
live_cuda_process_residency_mib=7952
live_cuda_process_residency_gib=7.76562
strict_8_gib_process_margin_mib=240
total_card_residency_mib=8876
raw_tokens_per_second=12435.888945826424
validation_loss=8.559529840946198
validation_ppl=5216.228138952979
checkpoint_forward_calls=800
checkpoint_function_calls=1600
checkpoint_recompute_calls=800
nonfinite_events=0
skipped_batches=0
mlflow_run_id=305be47eeef14735aaf4bcc51d9832c6
terminal_sha256=67d4baff797e3ac5ce8a7b278e207570fb8e42618aaa4f6d513afe0ba1148b27
```

Comparison with the uncheckpointed Branch51 `8x8` court:

```text
allocated_vram_saved_bytes=1870669312
raw_throughput_change_percent=-25.6989
validation_loss_equal=true
validation_ppl_equal=true
trainer_process_under_8_gib=PASS
whole_card_under_8_gib=NOT_CLAIMED
fixed_token_quality=UNAVAILABLE
```
