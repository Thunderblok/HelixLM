# Branch 50 Overnight Campaign Report V0

Status: `ACTIVE_NOT_PROMOTED`

This report is updated from admitted receipts. `PASS` means the named court ran
and passed. `PENDING` means the required evidence does not yet exist. No pending
item may be inferred from a plan, process launch, or partial metric stream.

## Frozen model contract

```text
sequence_length=512
parameter_count_total=53592340
parameter_count_trainable=53592340
batch_size=12
gradient_accumulation=7
causal_targets_per_optimizer_step=42924
raw_tokens_per_optimizer_step=43008
dtype=torch.float32
amp_dtype=bfloat16
strict_nan_check=true
seed=42
```

Architecture, corpus, tokenizer, validation set, batch geometry, dtype path,
weight decay, clipping, dropout, and seed are held. The campaign may tune only:

```text
learning_rate
warmup_microbatches
scheduler_policy_and_minimum_ratio
checkpoint_cadence
```

## Source custody

```text
branch=50-from-49-linear-context-scaling

ablation_source_head=
ec85aa45841873926e88114bd12572c41223014b

ablation_source_tree=
4eed826a90968edc67708e1c5e03624ac45da253

campaign_source_head=
d708ef5da65a9ad67cc3df32c2b6a7586a44cd6f

campaign_source_tree=
8a2ff129c923139e3ab7da2b74a61a1e6e3b390a
```

The moving documentation branch is not the admitted execution source. The
ablation queue and successor campaign execute from separate clean exact-head
checkouts.

## Data custody

Training corpus:

```text
dataset=david-thrower/HelixLM-medium-1500.0Mt-2988750pt-20260528
split=pretrain_train
tokenizer=gpt2
dtype=uint16_le
eos_token_id=50256
raw_tokens=1504000000
shards=188
manifest_file_sha256=
b67f33931c0e545c8701166dbf990a7af64cf1c3966c5500d20bd2381bc9b115
```

Fixed validation corpus:

```text
dataset=david-thrower/HelixLM-medium-1500.0Mt-2988750pt-20260528
split=pretrain_val
tokenizer=gpt2
dtype=uint16_le
eos_token_id=50256
raw_tokens=8000000
shards=1
manifest_file_sha256=
2c15971275e2834e378ea358fc2acf05f7251d2199ebfa8854e5974a29f7932b
```

Known identity limitation:

```text
dataset_revision=null
dataset_config=null
dataset_column=null
tokenizer_revision=null
```

Those values are absent from the historical materialization manifest. They are
reported as unavailable; this report does not invent them.

The exact full pass contains 2,937,500 samples and 1,501,062,500 causal
next-token targets. With 84 full samples per optimizer update it requires
34,970 full optimizer steps plus one partial-tail optimizer step.

## Completed 300M control

```text
status=PASS
ablation_id=control
learning_rate=0.00015
warmup_microbatches=2000
steps=6990
causal_targets_seen=300038760
raw_tokens_seen=300625920
best_val_step=6990
best_val_loss=4.360077083110809
best_val_ppl=78.26316695894894
skipped_batches=0
nonfinite_events=0
peak_vram_bytes=12655427072
checkpoint_readback=PASS
best_checkpoint_readback=PASS
mlflow_run_id=2169f744bef24ac08bf2b5cea23ab0b2
final_model_root=
004a6c0c763214b5c6d0821d9154ee1966b1398aab696829a08d004bf2bcb08c
```

## Active and pending stages

```text
lr_1e_4_300m=ACTIVE
warmup_500_300m=PENDING
primary_300m_comparison=PENDING

exact_resume_live_court=PENDING
rotating_checkpoint_live_court=PENDING
hostile_resume_refusal_live_court=PENDING
diminishing_return_live_court=PENDING

operational_control_100m=PENDING
cosine_scheduler_r0p1_100m=PENDING
checkpoint_every_250_100m=PENDING
operational_100m_comparison=PENDING

combined_promotion_manifest=PENDING
combined_100m_pilot=PENDING
full_corpus_promotion_manifest=PENDING
full_1504m_raw_token_pass=PENDING
final_checkpoint_preflight=PENDING
lighteval_execution=PENDING
promotion_recommendation=PENDING
```

The live GPU courts are intentionally serialized behind the ablation queue.
They refuse to run while another compute process owns the device.

## Static and proxy courts

```text
full_campaign_control_courts=PASS
live_campaign_static_courts=PASS
overnight_campaign_static_courts=PASS
lighteval_helper_courts=PASS
```

A CPU-only Lighteval consumer preflight over the completed control checkpoint
passed. See `BRANCH50_LIGHTEVAL_READINESS_V0.md`. Lighteval itself has not run,
and no benchmark score is established.

## Promotion law

A candidate may be promoted only from completed terminals and exact comparison
receipts. A smoke terminal, diminishing-return stop, healthy partial run, or
writer-reported metric is not promotion evidence by itself.

The full-corpus stage remains held until:

```text
primary_single_knob_comparison=PASS
operational_single_knob_comparison=PASS
combined_pilot=PASS
promotion_manifest_exact_match=PASS
resume_and_rotation_live_courts=PASS
```

Final disposition vocabulary:

```text
PROMOTE
HOLD
REVISE
REFUSE
```

Current disposition: `HOLD_ACTIVE_CAMPAIGN`.
