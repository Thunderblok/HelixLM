# Branch53 Full-Corpus LR 2e-4 Report V0

## Disposition

```text
RUN_TERMINAL=PASS
FULL_CORPUS_PASS=true
MLFLOW_SPOOL_FINISHED=PASS
NUMERICAL_HEALTH=PASS
CHECKPOINT_HEALTH=PASS
VALID_ABLATION=learning_rate_1.5e-4_to_2.0e-4
FFN_EXPANSION_ABLATION=NOT_EXECUTED
PRODUCTION_EFFECT=none
```

The completed run is a valid single-factor learning-rate ablation over the
53,592,340-parameter Branch50 model. It is not evidence for the queued
`ffn_expansion=3.0` hypothesis. The runtime resolved `ffn_expansion=2.5`, the
parameter count did not change, and the ablation contract lists only
`learning_rate` in `changed_knobs`.

## Exact subject

```text
source_head=6ffbe0fb49a719c2fba5878b8f494aa58e23fcb1
source_tree=50225c2643f565fab4e3d93c22be418feebcdcd5
source_dirty=false
run_id=branch53-ablation-lr2e4-b12a7-full1p504b-retry-allocator-s512-b12-a7-t1501062500-20260901T002405Z
mlflow_run_id=edcae37fcfd045b29c19482ae948bf58
initial_model_root=2c61d9764e6109f61a1c4a7f5c3ae56325301072cdf1fc602221be5279295f59
final_model_root=e25e0fe551a39180b2e67447b445420d0561f3b362d89bdcd2714e4884882f0d
terminal_sha256=278e5088c9db8dfcce2c1ad91351fcb554146ca7889213cc7be4916613cd7cbf
ablation_contract_sha256=2824f6b643b41906d4fb861c20674c0142f6a0949a6a2d797ee7906d255f9e77
resolved_config_sha256=0c36be53b4168f5b9b68a171671d5a83f9cc27fe9d4d4316b8eacb64d7174e15
mlflow_spool_sha256=8a39993c5a4fdf59952850d8bfab82e4279668e3a88a5590fefda1c03a25c590
terminal_checkpoint_sha256=3d75cd690ac44d923a7ec36370e87ebf9db55cca37d1ac74e3060707bf0c0d30
best_checkpoint_sha256=588ed9606ff2b185c00f2fb8238d74c0ec5877ffbdce055674489e397b2a82bf
```

The local MLflow HTTP server was unavailable during this report's readback.
The run-local append-only spool is therefore the custody source for the
`run_started` and `run_finished(status=FINISHED)` observations. It records no
MLflow errors.

## Executed configuration

```text
tokenizer=gpt2
sequence_length=512
batch_size=12
gradient_accumulation=7
causal_targets_per_optimizer_step=42924
optimizer_steps=34971
raw_tokens=1504000000
causal_targets=1501062500
learning_rate=2.0e-4
scheduler=linear_warmup_then_constant
warmup_microbatches=2000
weight_decay=0.05
gradient_clip=1.0
dropout=0.05
attention_dropout=0.05
ffn_expansion=2.5
n_loops=3
activation_checkpointing=false
amp_dtype=bfloat16
seed=42
```

## Matched full-corpus comparison

The comparator is the exact Branch50 full-corpus run
`branch50-ablation-promoted-full-corpus-s512-b12-a7-t1501062500-20260829T142802Z`
with MLflow run `98ecead0eaba4b438728d8ce51a926c5`. Both runs share the
same initial model root, model size, tokenizer, sequence length, optimizer
geometry, data order, target count, scheduler family, and seed. The declared
single changed knob is learning rate.

| Observation | Branch50 LR 1.5e-4 | Completed LR 2.0e-4 | Delta |
| --- | ---: | ---: | ---: |
| Best validation perplexity | 47.378291 | 45.822565 | -1.555726 (-3.2836%) |
| Final validation perplexity | 47.470409 | 45.822565 | -1.647844 (-3.4713%) |
| Best step | 34,700 | 34,971 | +271 |
| Peak allocated VRAM | 12,655,427,072 B | 12,630,845,440 B | -24,581,632 B (-0.1942%) |
| Final raw tokens/second | 20,474.09 | 20,324.67 | -0.7298% |
| Wall time | 73,463.38 s | 74,004.11 s | +540.73 s (+0.7360%) |
| Skipped batches | 0 | 0 | 0 |
| Nonfinite events | 0 | 0 | 0 |

The LR 2e-4 curve is lower at every inspected matched validation step from 100
through 34,971. The final five validation observations remain in the narrow
45.82-45.97 range and the best observation occurs at the terminal step.

## Scientific claim ceiling

Admitted:

```text
For this exact seed, corpus order, model, and one full-corpus pass, LR 2e-4
outperformed the LR 1.5e-4 control on the shared validation projection without
introducing skipped batches, nonfinite events, or material VRAM growth.
```

Not admitted:

```text
ffn_expansion=3.0 improved quality
LR 2e-4 is a universal optimum
the result generalizes across seeds or datasets
the checkpoint passed an independent downstream capability evaluation
```

## Contract defect exposed

The generic runner hard-codes:

```text
schema=helix.branch53.ffn-expansion-ablation.v0
branch53_profile=ffn_expansion_3p0_v0
```

even when the resolved single-factor ablation is learning rate and
`ffn_expansion=2.5`. The run itself remains usable because its resolved config,
parameter count, and `changed_knobs` are explicit. Its Branch53 FFN profile and
`promotion_eligible=true` projection are not admissible as FFN evidence.

## Next experiment order

### G0 - repair the experiment identity court

Before another GPU launch:

1. Derive the profile from resolved knobs or refuse a profile/config mismatch.
2. Require the FFN lane to observe `ffn_expansion=3.0` and
   `changed_knobs=["ffn_expansion"]` for a single-factor court.
3. Bind the exact expected 3.0 parameter count and require a nonzero delta from
   the 53,592,340-parameter control.
4. Persist the resolved invocation arguments in the run artifact.
5. Refuse FFN promotion when any of those observations disagree.
6. Add a hostile court in which the FFN flag is omitted and require RED.

### G1 - independent checkpoint evaluation

Replay the exact Branch50 control checkpoint and the completed LR 2e-4
checkpoint through the same held-out/downstream evaluation suite. Bind both
checkpoint hashes, tokenizer identity, evaluation dataset root, and evaluator
source. This is the cheapest next model-quality test because both checkpoints
already exist; no new training is required.

### G2 - matched allocator and numerical preflight

Run matched 2.5 and 3.0 FFN subjects from the same source and data roots:

```text
learning_rate=2.0e-4
batch_size=12
gradient_accumulation=7
sequence_length=512
n_loops=3
activation_checkpointing=false
seed=42
target=about 4.7M causal targets
```

Require exact parameter identity, zero skipped batches, zero nonfinite events,
checkpoint readback, MLflow health, and recorded allocated/process VRAM. If
3.0 does not fit, do not change only the candidate geometry: rerun both control
and candidate under one matched lower-memory geometry.

### G3 - fixed-token FFN quality gate

If G2 passes, run a matched 100M-causal-target pair at FFN 2.5 and 3.0. Do not
spend another full-corpus pass yet. Promotion requires the 3.0 candidate to
improve the shared validation projection while staying numerically healthy and
within an explicitly recorded throughput/VRAM ceiling.

### G4 - full-corpus capacity run

Only a passing G3 may authorize a 1.5B-token FFN 3.0 run. Preserve LR 2e-4 as
the admitted control setting and keep tokenizer, data order, scheduler, seed,
sequence length, loop count, and optimizer geometry fixed.

Training validation perplexity alone remains insufficient for a capability or
production-admission claim even if the later FFN gates pass.
