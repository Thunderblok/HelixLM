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
