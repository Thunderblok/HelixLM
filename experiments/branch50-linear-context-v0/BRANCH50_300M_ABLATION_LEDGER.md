# Branch 50 300M Ablation Ledger

This ledger records the Branch-50 single-knob 300M-token ablation campaign. It
is the human-readable scientific log. The run directories, `resolved_config.json`,
`terminal.json`, checkpoints, MLflow runs, and spooled JSONL events remain the
primary evidence for each run.

## Scope

```text
MISSION=identify the best lawful Branch-50 512-token overnight configuration
CAMPAIGN=BRANCH50_300M_ABLATION_V0
MODEL_PUBLICATION=held
PRODUCTION_EFFECT=none
GPU=local RTX 5080
MLFLOW=https://mlflow.thunderline.net
```

Each ablation must change one declared knob only. Any run that changes multiple
model, optimizer, loader, or validation fields is a diagnostic run, not a clean
ablation.

## Fixed controls

These values stay fixed unless the row explicitly declares them as the ablated
knob.

```text
source_checkout=/home/mo/DEV/experiments/helix-branch50-linear-context-v0/source
current_source_head=e20e63b5d5d3a898166e25e565127ae0aef46c72
current_source_tree=cb75803b272f8c5b5084c24bbc34727c6d10c121
model_base_head=03d0698dd3365c81695d9ed8d4568d35d6044fbb
model_base_tree=745c042db9860bca4cdfa180543f8a60a769c936
model_source_diff_required=false

dataset=david-thrower/HelixLM-medium-1500.0Mt-2988750pt-20260528
train_manifest_sha256=83f56ff60e238e6483a5fe705070b20234df555253dfb77e1b309317e3b33b4c
validation_manifest_sha256=5a7bdaedb42c9f56cc6b666dd2bdd5751406ad24c53773eacea7747efb340406
ordering_algorithm=u16_shard_permutation_then_per_shard_window_permutation_v1
tokenizer=gpt2
pad_token_id=50256
eos_token_id=50256
bos_token_id=50256
vocab_size=50257

seq_len=512
batch_size=12
gradient_accumulation=7
effective_sequences=84
raw_tokens_per_optimizer_step=43008
causal_targets_per_sample=511
causal_targets_per_optimizer_step=42924
target_causal_targets=300000000
aligned_causal_targets=300038760
optimizer_steps=6990

d_model=512
n_heads=8
n_columns=3
nodes_per_column=[2,3,2]
attention_mode=multi_scale_windowed
local_window=64
coarse_window=128
compressed_windows=8
compressed_views=8
consensus_type=cosine
corrector_type=ffn
use_ssm=false
use_titans_memory=false
use_cca=false

master_dtype=float32
amp_dtype=bfloat16
strict_nan_check=true
dropout=0.05
attention_dropout=0.05
weight_decay=0.05
grad_clip=1.0
grad_buffer_ratio=0.0
warmup_microbatches=2000
warmup_optimizer_steps=285
scheduler_policy=linear_warmup_then_constant
scheduler_min_lr_ratio=1.0
validation_batches=16
eval_every=100
checkpoint_every=500
seed=42
```

Counting law:

```python
causal_targets = (labels[:, 1:] != -100).sum()
```

Do not count `labels != -100`; the first label in each sequence is not a
next-token prediction target.

## Active runner contract

Runner:

```text
experiments/branch50-linear-context-v0/run_branch50_300m_ablation.py
```

The runner currently enforces:

- clean source checkout
- no drift from `model_base_head` for `helix_lm/` and `requirements.txt`
- CUDA device capability `(12, 0)`
- BF16 support
- common U16 runner SHA-256 receipt match
- batch size `12` and accumulation `7`
- checkpoint readback at terminal
- immediate train loss/PPL logging and accumulated train loss/PPL logging
- token-weighted validation loss/PPL logging

Verification note: `python3 -m py_compile run_branch50_300m_ablation.py` passed
with Python 3.14.2. `--help` could not be executed in the generic shell Python
because that interpreter lacks `numpy`; run admission still depends on the
training environment.

## Planned ablation matrix

Run in this order unless an earlier court fails hard. `n_loops` variants are
last by instruction.

| Order | Ablation ID | Single changed knob | Value | Fixed controls | Status | Decision |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | `control` | none | baseline | all controls above | planned | pending |
| 2 | `lr1e4` | learning rate | `0.00010` | control geometry, scheduler semantics, dropout, FFN, loops | planned | pending |
| 3 | `warmup500` | warmup | `500` microbatches | baseline LR and all other controls | planned | pending |
| 4 | `regularization-low` | one regularization field | TBD | every other baseline control | placeholder | pending |
| 5 | `ffn2` | `ffn_expansion` | `2.0` | baseline optimizer and all other controls; rebind parameters | planned | pending |
| 6 | `ffn4` | `ffn_expansion` | `4.0` | baseline optimizer and all other controls; rebind parameters | placeholder | pending |
| 7 | `loops2` | `n_loops` | `2` | best non-loop controls; parameter/execution fingerprints must be rebound | last | pending |
| 8 | `loops4` | `n_loops` | `4` | best non-loop controls; memory gate required before full 300M | last | pending |

`ffn_expansion` is an active Branch-50 config field used by the graph-level
output FFN. It is distinct from the previously removed/ignored
`output_ffn_dim` control. Do not treat `output_ffn_dim` as an ablation knob.

The initial matrix compares every row directly with the unchanged control.
Winning knobs are not combined during this phase. A later promotion campaign
may combine independently supported settings under a new declared contract.

## Per-run evidence template

Copy this block for each run after launch.

```text
RUN_ID=
ABLATED_KNOB=
ABLATED_VALUE=
HYPOTHESIS=

LOCAL_RUN_ROOT=
MLFLOW_RUN_ID=
MLFLOW_URL=
RUNNER_SHA256=
COMMON_RUNNER_SHA256=
SOURCE_HEAD=
SOURCE_TREE=
MODEL_BASE_HEAD=
MODEL_BASE_TREE=
SOURCE_DIRTY=

PARAMETER_COUNT_TOTAL=
PARAMETER_COUNT_TRAINABLE=
INITIAL_MODEL_ROOT=
FINAL_MODEL_ROOT=
CHECKPOINT_SHA256=
CHECKPOINT_READBACK=

DATASET=
TRAIN_MANIFEST_SHA256=
VALIDATION_MANIFEST_SHA256=
ORDERING_ALGORITHM=
TOKENIZER=

SEQ_LEN=
BATCH_SIZE=
GRADIENT_ACCUMULATION=
CAUSAL_TARGETS_PER_OPTIMIZER_STEP=
TARGET_CAUSAL_TARGETS=
ALIGNED_CAUSAL_TARGETS=
OPTIMIZER_STEPS=

LEARNING_RATE=
WARMUP_MICROBATCHES=
WARMUP_OPTIMIZER_STEPS=
SCHEDULER_POLICY=
SCHEDULER_MIN_LR_RATIO=
WEIGHT_DECAY=
GRAD_CLIP=
DROPOUT=
ATTENTION_DROPOUT=
FFN_EXPANSION=
N_LOOPS=

TERMINAL_STATUS=
TRAIN_LOSS_FINAL=
TRAIN_PPL_FINAL=
TRAIN_ACCUM_LOSS_FINAL=
TRAIN_ACCUM_PPL_FINAL=
VAL_LOSS_FINAL=
VAL_PPL_FINAL=
BEST_VAL_LOSS=
BEST_VAL_PPL=
RAW_TOKENS_PER_SECOND=
CAUSAL_TARGETS_PER_SECOND=
PEAK_VRAM_BYTES=
SKIPPED_BATCHES=
NONFINITE_EVENTS=
MLFLOW_ERRORS=

OBSERVATIONS=
PROMOTE_HOLD_DECISION=
DECISION_REASON=
```

## Promotion courts

A run can be promoted only when all of these are true:

```text
terminal_status=PASS
checkpoint_readback=PASS
skipped_batches=0
nonfinite_events=0
mlflow_errors=[]
source_dirty=false
model_source_diff=false
validation_loss_finite=true
validation_ppl_finite=true
train_immediate_loss_logged=true
train_immediate_ppl_logged=true
train_accumulated_loss_logged=true
train_accumulated_ppl_logged=true
val_loss_logged=true
val_ppl_logged=true
```

Compare candidates against the control by:

```text
primary=best and final validation NLL
secondary=validation PPL, loss slope, throughput, peak VRAM, checkpoint health
tertiary=train immediate vs accumulated behavior
```

Do not promote a candidate on train loss alone. Do not interpret one high
immediate train PPL as quality failure without checking accumulated train loss
and fixed validation.

## Live observations

### 2026-08-28 ledger initialization

```text
EVENT=ledger_created
SOURCE_HEAD=e20e63b5d5d3a898166e25e565127ae0aef46c72
SOURCE_TREE=cb75803b272f8c5b5084c24bbc34727c6d10c121
RUNNER_STATUS=untracked_in_worktree
RUNNER_SYNTAX=PASS
RUNNER_HELP=not_run_generic_python_missing_numpy
ABLATION_ARTIFACTS_FOUND=none
```

### 2026-08-29 control run in progress

```text
EVENT=control_run_progress_observation
OBSERVATION_STATUS=in_progress
PROMOTION_STATUS=non_promotable_until_terminal_and_validation

RUN_ID=branch50-ablation-control-s512-b12-a7-t300000000-20260829T020213Z
LOCAL_RUN_ROOT=/home/mo/DEV/experiments/helix-branch50-linear-context-v0/artifacts/ablation-300m-v0/branch50-ablation-control-s512-b12-a7-t300000000-20260829T020213Z
WATCHDOG=installed_externally_every_30_min

LATEST_EVIDENCED_STEP=1598
TOTAL_OPTIMIZER_STEPS=6990
CAUSAL_TARGETS_SEEN=68592552
CAUSAL_TARGETS_PER_SECOND_APPROX=20843

TRAIN_ACCUM_LOSS=5.33244
TRAIN_IMMEDIATE_LOSS=5.48629
SKIPPED_BATCHES=0
NONFINITE_EVENTS=0
PEAK_ALLOCATED_VRAM_BYTES=12653001216

GPU_TOTAL_PROCESS_AND_SYSTEM_USE_MIB=15365
GPU_FREE_MIB=453
GPU_UTILIZATION_PERCENT=80
GPU_TEMPERATURE_C=61
GPU_POWER_W=204.64

DISK_FREE_KB=374079128

VALIDATION_STATUS=not_claimed_from_this_observation
TERMINAL_STATUS=not_claimed_from_this_observation
DECISION=pending_terminal_and_validation
```

### 2026-08-29 control run step-2000 milestone

```text
EVENT=control_run_step_2000_progress_observation
OBSERVATION_STATUS=in_progress
PROMOTION_STATUS=non_promotable_until_terminal

RUN_ID=branch50-ablation-control-s512-b12-a7-t300000000-20260829T020213Z
LATEST_EVIDENCED_STEP=2000
TOTAL_OPTIMIZER_STEPS=6990
CAUSAL_TARGETS_SEEN=85848000
RAW_TOKENS_SEEN=86016000
SAMPLES_SEEN=168000
CAUSAL_TARGETS_PER_SECOND=20786.190

TRAIN_ACCUM_LOSS=5.1489300
TRAIN_IMMEDIATE_LOSS=5.16936445
TRAIN_PPL=175.80307
GRADIENT_NORM=1.697095
SKIPPED_BATCHES=0
NONFINITE_EVENTS=0

LATEST_CHECKPOINT=latest.pt
LATEST_CHECKPOINT_SHA256=f431a13685403ed4e3e5c8029d15d81cea9a3e1b79d3e0b61a860ae562bd2562

VALIDATION_STEP=2000
VAL_LOSS=5.150536358356476
VAL_PPL=172.52400019461052
BEST_VAL_LOSS=5.150536358356476
BEST_VAL_PPL=172.52400019461052
VAL_CAUSAL_TARGETS=98112

PEAK_ALLOCATED_VRAM_BYTES=12655427072
GPU_TOTAL_PROCESS_AND_SYSTEM_USE_MIB=15365
GPU_FREE_MIB=453
GPU_UTILIZATION_PERCENT=93
GPU_TEMPERATURE_C=62
GPU_POWER_W=211.57

DISK_FREE_KB=374070824

TERMINAL_STATUS=not_claimed_from_this_observation
DECISION=pending_terminal
```

### 2026-08-29 control run step-2500 milestone

```text
EVENT=control_run_step_2500_progress_observation
OBSERVATION_STATUS=in_progress
PROMOTION_STATUS=non_promotable_until_terminal

RUN_ID=branch50-ablation-control-s512-b12-a7-t300000000-20260829T020213Z
LATEST_EVIDENCED_STEP=2500
TOTAL_OPTIMIZER_STEPS=6990
CAUSAL_TARGETS_SEEN=107310000
RAW_TOKENS_SEEN=107520000
SAMPLES_SEEN=210000
CAUSAL_TARGETS_PER_SECOND=20729.93184

TRAIN_ACCUM_LOSS=4.966458184
TRAIN_IMMEDIATE_LOSS=5.033377647
TRAIN_PPL=153.45044
GRADIENT_NORM=1.402777672
SKIPPED_BATCHES=0
NONFINITE_EVENTS=0

LATEST_CHECKPOINT=latest.pt
LATEST_CHECKPOINT_SHA256=eb6efcd394c93c10d79fa99df0b1e7da58ae5cce9a4c243bb13bd05f76277662

VALIDATION_STEP=2500
VAL_LOSS=4.984133929014206
VAL_PPL=146.0770071680938
BEST_VAL_LOSS=4.984133929014206
BEST_VAL_PPL=146.0770071680938
VAL_CAUSAL_TARGETS=98112

PEAK_ALLOCATED_VRAM_BYTES=12655427072
GPU_TOTAL_PROCESS_AND_SYSTEM_USE_MIB=15365
GPU_FREE_MIB=453
GPU_UTILIZATION_PERCENT=97
GPU_TEMPERATURE_C=62
GPU_POWER_W=202.19

DISK_FREE_KB=374686272
QUEUE_SUPERVISOR=waiting
LOOPS_ABLATION=held_until_after_non_loop_ablations

TERMINAL_STATUS=not_claimed_from_this_observation
DECISION=pending_terminal
```

### 2026-08-29 control run step-3000 durable checkpoint

```text
EVENT=control_run_step_3000_durable_checkpoint_observation
OBSERVATION_STATUS=in_progress
PROMOTION_STATUS=non_promotable_until_terminal
EVIDENCE_CLASS=durable_checkpoint_spool_validation_gpu_disk_queue

RUN_ID=branch50-ablation-control-s512-b12-a7-t300000000-20260829T020213Z
RUN_ROOT=/home/mo/DEV/experiments/helix-branch50-linear-context-v0/artifacts/ablation-300m-v0/branch50-ablation-control-s512-b12-a7-t300000000-20260829T020213Z
CHECKPOINT_STEP=3000
LATEST_EVIDENCED_STEP_AFTER_CHECKPOINT=3013
TOTAL_OPTIMIZER_STEPS=6990

CAUSAL_TARGETS_SEEN_AT_CHECKPOINT=128772000
RAW_TOKENS_SEEN_AT_CHECKPOINT=129024000
SAMPLES_SEEN_AT_CHECKPOINT=252000
SHARD_POSITION=16
WINDOW_POSITION=2000

CAUSAL_TARGETS_PER_SECOND_AT_CHECKPOINT=20691.179588395702
RAW_TOKENS_PER_SECOND_AT_CHECKPOINT=20731.67113357847

TRAIN_ACCUM_LOSS_AT_CHECKPOINT=4.928734711238316
TRAIN_ACCUM_PPL_AT_CHECKPOINT=138.20453302117684
TRAIN_IMMEDIATE_LOSS_AT_CHECKPOINT=5.095064163208008
TRAIN_IMMEDIATE_PPL_AT_CHECKPOINT=163.2143166425734
GRADIENT_NORM_PRE_CLIP_AT_CHECKPOINT=2.0770516395568848
LEARNING_RATE_AT_CHECKPOINT=0.00015
SKIPPED_BATCHES_AT_CHECKPOINT=0
NONFINITE_EVENTS_AT_CHECKPOINT=0

LATEST_CHECKPOINT=latest.pt
LATEST_CHECKPOINT_SHA256_SPOOL=08de7ae2e12d92ab12c55100bd8d07bd1f9e1fac179cdfe42a180de83f76b571
LATEST_CHECKPOINT_SHA256_READBACK=08de7ae2e12d92ab12c55100bd8d07bd1f9e1fac179cdfe42a180de83f76b571
LATEST_CHECKPOINT_READBACK=PASS
LATEST_CHECKPOINT_SIZE_BYTES=641365146
LATEST_CHECKPOINT_OPTIMIZER_STATE_ENTRIES=284
LATEST_CHECKPOINT_SCHEDULER=warmup_then_constant
LATEST_CHECKPOINT_BASE_LR=0.00015
LATEST_CHECKPOINT_WARMUP_MICROBATCHES=2000
LATEST_CHECKPOINT_WARMUP_OPTIMIZER_STEPS=285
LATEST_CHECKPOINT_MIN_RATIO=1.0
LATEST_CHECKPOINT_VAL_ORDERING=periodic_checkpoint_written_before_same_step_validation
LATEST_CHECKPOINT_LAST_VAL_LOSS=4.881312429904938
LATEST_CHECKPOINT_BEST_VAL_LOSS=4.881312429904938
LATEST_CHECKPOINT_LAST_VAL_STEP=2900
BEST_MODEL_SHA256_READBACK=017af26499ea313b64604764afd5a99b2544b639621e72da0aa6400c557876e3
BEST_MODEL_SIZE_BYTES=214482851
BEST_MODEL_VALIDATION_STEP=3000
BEST_MODEL_VAL_LOSS=4.858914524316788

VALIDATION_STEP=3000
VAL_LOSS=4.858914524316788
VAL_PPL=128.8842254578135
BEST_VAL_LOSS=4.858914524316788
BEST_VAL_PPL=128.8842254578135
VAL_CAUSAL_TARGETS=98112

PEAK_ALLOCATED_VRAM_BYTES=12655427072
GPU_TOTAL_PROCESS_AND_SYSTEM_USE_MIB=15365
GPU_FREE_MIB=453
GPU_UTILIZATION_PERCENT=84
GPU_TEMPERATURE_C=61
GPU_POWER_W=202.17

DISK_FREE_KB=374344264
DISK_FREE_BYTES=383328526336
PROCESS_STATUS=active_after_checkpoint
QUEUE_SUPERVISOR=waiting_for_terminal_or_midpoint_3500
TERMINAL_STATUS=not_present

EVIDENCE=mlflow_spool_step_3000_train_checkpoint_validation_events_sha256_readback_checkpoint_metadata_gpu_disk_queue
INFERENCE=control_run_remains_numerically_lawful_so_far_but_cannot_be_promoted_before_terminal
NOTE=latest_pt_step3000_does_not_include_same_step_validation_improvement_best_model_pt_does
NOTE_2=checkpoint_validation_ordering_is_being_repaired_in_isolated_full_campaign_prep_not_in_active_control
DECISION=pending_terminal
RECOMMENDATION=continue_control_to_terminal_unless_resource_pressure_or_numerical_failure_appears
```

### 2026-08-29 control run step-3500 durable checkpoint

```text
EVENT=control_run_step_3500_durable_checkpoint_observation
OBSERVATION_STATUS=in_progress
PROMOTION_STATUS=non_promotable_until_terminal
EVIDENCE_CLASS=durable_checkpoint_spool_validation_gpu_disk_queue_source_sync

RUN_ID=branch50-ablation-control-s512-b12-a7-t300000000-20260829T020213Z
RUN_ROOT=/home/mo/DEV/experiments/helix-branch50-linear-context-v0/artifacts/ablation-300m-v0/branch50-ablation-control-s512-b12-a7-t300000000-20260829T020213Z

SOURCE_WORKTREE=/home/mo/DEV/experiments/helix-branch50-linear-context-v0/source
SOURCE_BRANCH=50-from-49-linear-context-scaling
SOURCE_HEAD=2211e25414e1c98f0f6b0df8364efe6d324c372b
SOURCE_TREE=a4c80c85a073af6fd6210c1502a16634f2b564ee
SOURCE_REMOTE=oko/50-from-49-linear-context-scaling
SOURCE_REMOTE_HEAD=2211e25414e1c98f0f6b0df8364efe6d324c372b
SOURCE_AHEAD_BEHIND_VS_REMOTE=0 0
SOURCE_REMOTE_SYNC=PASS
SOURCE_WORKTREE_STATUS=dirty_doc_only_this_ledger

CHECKPOINT_STEP=3500
LATEST_EVIDENCED_STEP_AFTER_CHECKPOINT=3525
TOTAL_OPTIMIZER_STEPS=6990

CAUSAL_TARGETS_SEEN_AT_CHECKPOINT=150234000
RAW_TOKENS_SEEN_AT_CHECKPOINT=150528000
SAMPLES_SEEN_AT_CHECKPOINT=294000
SHARD_POSITION=18
WINDOW_POSITION=12750

CAUSAL_TARGETS_PER_SECOND_AT_CHECKPOINT=20668.304125780065
RAW_TOKENS_PER_SECOND_AT_CHECKPOINT=20708.750904891178

TRAIN_ACCUM_LOSS_AT_CHECKPOINT=4.817449365343366
TRAIN_ACCUM_PPL_AT_CHECKPOINT=123.6493040225938
TRAIN_IMMEDIATE_LOSS_AT_CHECKPOINT=5.020231246948242
TRAIN_IMMEDIATE_PPL_AT_CHECKPOINT=151.44632124466776
GRADIENT_NORM_PRE_CLIP_AT_CHECKPOINT=1.7393559217453003
LEARNING_RATE_AT_CHECKPOINT=0.00015
SKIPPED_BATCHES_AT_CHECKPOINT=0
NONFINITE_EVENTS_AT_CHECKPOINT=0

LATEST_CHECKPOINT=latest.pt
LATEST_CHECKPOINT_SIZE_BYTES=641365146
LATEST_CHECKPOINT_SHA256_SPOOL=dffd7bd4d978eddb1329df5deadaff98cc00ef1706febeadd1e21edca405ab8d
LATEST_CHECKPOINT_SHA256_READBACK=dffd7bd4d978eddb1329df5deadaff98cc00ef1706febeadd1e21edca405ab8d
LATEST_CHECKPOINT_READBACK=PASS
LATEST_CHECKPOINT_MODEL_ROOT=3866b76ae10cced852b79a0b06ba8560625a6409af83d7a1c454d481cb427470
LATEST_CHECKPOINT_ABLATION_CONTRACT_ROOT=e9afc1f4e4d897169f04502e26244d0a4647d26e2d67cbc1bbf9df52303067f4
LATEST_CHECKPOINT_TRAIN_MANIFEST_SHA256=83f56ff60e238e6483a5fe705070b20234df555253dfb77e1b309317e3b33b4c
LATEST_CHECKPOINT_VAL_MANIFEST_SHA256=5a7bdaedb42c9f56cc6b666dd2bdd5751406ad24c53773eacea7747efb340406
LATEST_CHECKPOINT_OPTIMIZER_STATE_ENTRIES=284
LATEST_CHECKPOINT_SCHEDULER=linear_warmup_then_constant
LATEST_CHECKPOINT_BASE_LR=0.00015
LATEST_CHECKPOINT_WARMUP_MICROBATCHES=2000
LATEST_CHECKPOINT_WARMUP_OPTIMIZER_STEPS=285
LATEST_CHECKPOINT_MIN_RATIO=1.0
LATEST_CHECKPOINT_VAL_ORDERING=periodic_checkpoint_written_before_same_step_validation
LATEST_CHECKPOINT_LAST_VAL_LOSS=4.782320976257324
LATEST_CHECKPOINT_BEST_VAL_LOSS=4.782320976257324
LATEST_CHECKPOINT_BEST_VAL_STEP=3400

BEST_MODEL=best-model.pt
BEST_MODEL_SIZE_BYTES=214482851
BEST_MODEL_SHA256_READBACK=893b0189bc5f1f57044818a0b0b950b039255c5d595afa520c84e14b68a3d744
BEST_MODEL_VALIDATION_STEP=3500
BEST_MODEL_VAL_LOSS=4.76538422703743

VALIDATION_STEP=3500
VAL_LOSS=4.76538422703743
VAL_PPL=117.3762077373327
BEST_VAL_LOSS=4.76538422703743
BEST_VAL_PPL=117.3762077373327
VAL_CAUSAL_TARGETS=98112

PEAK_ALLOCATED_VRAM_BYTES=12655427072
GPU_NAME=NVIDIA_GeForce_RTX_5080
GPU_TOTAL_PROCESS_AND_SYSTEM_USE_MIB=15367
GPU_FREE_MIB=451
GPU_UTILIZATION_PERCENT=80
GPU_TEMPERATURE_C=61
GPU_POWER_W=201.92

DISK_FREE_KB=374329656
PROCESS_STATUS=active_after_checkpoint
TERMINAL_STATUS=not_present

EVIDENCE=mlflow_spool_step_3500_train_checkpoint_validation_events_sha256_readback_checkpoint_metadata_gpu_disk_process_source_sync
INFERENCE=control_run_remains_numerically_lawful_so_far_but_cannot_be_promoted_before_terminal
NOTE=latest_pt_step3500_does_not_include_same_step_validation_improvement_best_model_pt_does
DECISION=pending_terminal
RECOMMENDATION=continue_control_to_terminal_unless_resource_pressure_or_numerical_failure_appears
```

The prior Branch-50 100M context-promotion evidence retained 512 over 1024:

```text
CONTROL_512_RUN=branch50-quality-s512-b12-a7-t100000000-20260828T215009Z
CONTROL_512_MLFLOW=3f960461105f479598b1643ee1d34b8c
CONTROL_512_VAL_LOSS=5.025004148483276
CONTROL_512_VAL_PPL=152.17088738974164

CANDIDATE_1024_RUN=branch50-quality-s1024-b6-a7-t100000000-20260828T231135Z
CANDIDATE_1024_MLFLOW=4561589acb4c44d39d67f9e9635e267a
CANDIDATE_1024_VAL_LOSS=5.03864461183548
CANDIDATE_1024_VAL_PPL=154.26079001833116

VERDICT=RETAIN_512_SEED42_QUALITY_GATE
```

### 2026-08-29 runner admission and exact-resume court

```text
RUNNER_SHA256=b81b3479d6a72df51b1b460b3a1e00aabec723a81687de0fae01ddb115103bd9
ADMITTED_SOURCE_HEAD=5d640890045efe9215346dec94f62215d4a509e4

STEP1_MLFLOW=ccbfc160b5814b1f936c3a71d5f5e5be
STEP1_MODEL_ROOT=3d26f9cd96f0f8fa9253fcfbf9a32d29fd89d0a1f1646501223dbe1e9c65b92d
STEP1_CURSOR_CAUSAL_TARGETS=42924
STEP1_CHECKPOINT_READBACK=PASS

RESUMED_STEP2_MLFLOW=25156a1a16574bfc8cfa249c58d1757f
RESUMED_STEP2_MODEL_ROOT=0368a2cb2716045c01704aca0682c2e07c74cfd29073e0d2c0a54ec1ef8af62c
RESUMED_STEP2_CURSOR_CAUSAL_TARGETS=85848
RESUMED_STEP2_CHECKPOINT_READBACK=PASS

UNINTERRUPTED_STEP2_MLFLOW=0055c448c6cf41a38d0d703d10c63f40
UNINTERRUPTED_STEP2_MODEL_ROOT=0368a2cb2716045c01704aca0682c2e07c74cfd29073e0d2c0a54ec1ef8af62c
UNINTERRUPTED_STEP2_CURSOR_CAUSAL_TARGETS=85848
UNINTERRUPTED_STEP2_CHECKPOINT_READBACK=PASS

RESUMED_EQUALS_UNINTERRUPTED_MODEL_ROOT=PASS
RESUMED_EQUALS_UNINTERRUPTED_CURSOR=PASS
MLFLOW_ERRORS=[]

HOSTILE_COURT=cross_ablation_resume
MUTATION=control_checkpoint_requested_under_lr1e4_contract
RESULT=REFUSED
REASON=resume_scheduler_does_not_match_ablation

ADMISSION_TERMINAL=PASS
```

These are context-promotion facts, not 300M ablation results.
