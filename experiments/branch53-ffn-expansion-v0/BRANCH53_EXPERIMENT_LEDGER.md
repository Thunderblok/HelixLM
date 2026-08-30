# Branch53 Experiment Ledger

## 2026-08-30 - queued FFN capacity pilot

```text
parent_branch=52-from-51-activation-checkpointing
parent_head=cd5d8f5292492cda7553727c5e76cc73d00e31d2
branch=53-from-52-ffn-expansion
single_new_factor=ffn_expansion_2.5_to_3.0
target_causal_targets=400000000
launch_gate=exact_branch52_terminal_pass
parent_mlflow_run_id=3b4d833fea24417eae11f87184e69ff1
gpu_overlap=forbidden
production_effect=none
```

Rationale:

```text
Branch52 established exact matched-step numerical parity while reducing the
trainer's driver-observed residency below 8 GiB. Branch51's frozen experiment
order names FFN expansion as the next capacity knob before loop-count changes.
The saved memory is spent on FFN width while all optimizer, scheduler, data,
loop, sequence, and checkpointing settings remain fixed.
```

Claims before terminal:

```text
quality_improvement=UNAVAILABLE
throughput=UNAVAILABLE
peak_vram=UNAVAILABLE
```
