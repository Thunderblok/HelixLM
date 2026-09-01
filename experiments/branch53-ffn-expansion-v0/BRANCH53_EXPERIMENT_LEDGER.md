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

## 2026-09-01 - completed full-corpus LR 2e-4 run

The completed 1.504B-raw-token terminal is documented in
`BRANCH53_FULL_CORPUS_LR2E4_REPORT_V0.md`.

```text
terminal=PASS
mlflow_run_id=edcae37fcfd045b29c19482ae948bf58
best_validation_perplexity=45.82256472942389
final_validation_perplexity=45.82256472942389
skipped_batches=0
nonfinite_events=0
valid_ablation=learning_rate_1.5e-4_to_2.0e-4
ffn_expansion_ablation=NOT_EXECUTED
next_gate=repair_profile_binding_then_matched_ffn_2.5_vs_3.0_preflight
```

The run-local contract resolved `ffn_expansion=2.5`, the parameter count stayed
at 53,592,340, and `changed_knobs` contained only `learning_rate`. The hard-coded
`ffn_expansion_3p0_v0` profile is therefore not FFN evidence.

## 2026-09-01 - profile repair and checkpoint evaluation gate

The profile/config identity repair and exact checkpoint export preflights are
documented in `BRANCH53_PROFILE_REPAIR_AND_EVAL_GATE_V0.md`.

```text
repair_head=bfe672b5939fd5c2b0f0b7871ef7004eb126b5cd
profile_identity=PASS
branch50_checkpoint_export=PASS
branch53_checkpoint_export=PASS
downstream_lighteval=UNAVAILABLE
matched_ffn_preflight=HELD
new_full_corpus_run=FORBIDDEN
```

No downstream benchmark was executed. The isolated Lighteval 0.13.0 attempt
was stopped before it could replace the working Torch/CUDA stack. A frozen
evaluator/task/data contract is the next gate.
