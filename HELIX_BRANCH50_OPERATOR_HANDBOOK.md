# Helix Branch 50 Operator Handbook

This is the durable operator and evidence record for the Branch 50
linear-context session conducted on 2026-08-28. It records what ran, what was
held constant, what the results establish, and what must happen before a
checkpoint can be described or published as a model.

## Mission and authority

```text
MISSION=validate linear context scaling, then earn a 1024-token quality promotion
BRANCH=50-from-49-linear-context-scaling
PRODUCTION_EFFECT=none
CLOUD_SPEND=none
MODEL_PUBLICATION=held pending Lighteval and founder admission
MLFLOW=https://mlflow.thunderline.net (projection only)
```

The committed packet and exact Git identities are the custody record. MLflow
is a public projection, not the sole source of truth.

## Source identity

The model bytes exercised by the completed courts were clean at:

```text
SOURCE_HEAD=03d0698dd3365c81695d9ed8d4568d35d6044fbb
SOURCE_TREE=745c042db9860bca4cdfa180543f8a60a769c936
SOURCE_DIRTY=false
```

That commit is the merged Branch-49 tokenizer correction used as the Branch 50
starting point. This documentation commit changes the repository tree but does
not alter the admitted `helix_lm/` model bytes. Reproduce the historical court
from the exact source commit above. A successor run must bind its new head/tree
and record any model-source diff explicitly.

## Fixed model contract

The context court changed sequence length only:

```text
vocab_size=50_257
d_model=512
n_heads=8
n_loops=3
n_columns=3
nodes_per_column=[2,3,2]
parameter_count=53_592_340

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
learning_rate=0.00015
weight_decay=0.05
grad_clip=1.0
grad_buffer_ratio=0.0

tokenizer=gpt2
pad_token_id=50_256
eos_token_id=50_256
bos_token_id=50_256
architectures=[HelixForCausalLM]
```

The linear result does not cover full/hybrid attention, an attention
corrector, SSM, Titans memory, CCA, a growing compressed-window count, or
generation without a cache.

## Dataset and counting law

The real-corpus trial used the immutable GPT-2-tokenized U16 projection of:

```text
david-thrower/HelixLM-medium-1500.0Mt-2988750pt-20260528
```

The MLflow-bound roots were:

```text
TRAIN_MANIFEST_SHA256=83f56ff60e238e6483a5fe705070b20234df555253dfb77e1b309317e3b33b4c
VALIDATION_MANIFEST_SHA256=5a7bdaedb42c9f56cc6b666dd2bdd5751406ad24c53773eacea7747efb340406
ORDERING_ALGORITHM=u16_shard_permutation_then_per_shard_window_permutation_v1
```

Keep raw input positions separate from causal next-token targets:

```python
causal_targets = (labels[:, 1:] != -100).sum()
```

Do not count `labels != -100`; the first label in each sequence is not a
next-token target. No certified mid-epoch resume contract exists for the
historical full-epoch runner.

## Court 1: linear context scaling

One warmed-up real optimizer step ran at each context with batch size 1.
Measurements cover forward plus backward FLOPs and dynamic allocated VRAM
above resident model/optimizer state.

| Context | Forward/backward FLOPs | Dynamic peak allocated | Gradient values | Nonfinite gradients |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 311,178,559,488 | 1,037,631,488 B | 53,328,660 | 0 |
| 1024 | 621,436,993,536 | 1,956,857,344 B | 53,328,660 | 0 |
| 2048 | 1,241,953,861,632 | 3,793,543,680 B | 53,328,660 | 0 |

```text
FLOP_FIT_R2=1.0
MEMORY_FIT_R2=0.9999999434684236
COURT=PASS
```

Evidence:

```text
experiments/branch50-linear-context-v0/evidence/linear-context-court.json
SHA256=5674b8e6e78e42b25482a3e7aba33b6b489254f372b8c7a9014c41ffe8d5f0e5
```

This establishes an approximately linear context slope for this exact
training graph. It does not establish linear generation cost; the current
model has no certified KV-cache generation path.

## Court 2: real-corpus 1024-token trial

The 1024 trial matched the 512 baseline's raw tokens per optimizer step:

```text
512 control=12 * 7 * 512=43_008 raw tokens/update
1024 trial=6 * 7 * 1024=43_008 raw tokens/update
```

It completed 100 optimizer steps:

```text
MLFLOW_RUN_ID=8e1f6c8b33c048cba447e87ee0a1c505
MLFLOW_STATUS=FINISHED
RAW_TOKENS_SEEN=4_300_800
CAUSAL_TARGETS_SEEN=4_296_600
FINAL_TRAIN_LOSS=8.543785095214844
FINAL_ACCUMULATED_TRAIN_LOSS=8.539064407348633
FINAL_VALIDATION_LOSS=8.495547771453857
FINAL_VALIDATION_PPL=4892.935805067157
RAW_TOKENS_PER_SECOND=19660.80261428606
CAUSAL_TARGETS_PER_SECOND=19641.602611733048
PEAK_ALLOCATED_VRAM_BYTES=12_651_372_544
MLFLOW_ERRORS=[]
```

The large loss/PPL values are expected for a 100-step cold-start smoke and are
not quality evidence. The trial establishes execution, numerical integrity,
matched token geometry, logging, and memory feasibility at 1024 tokens.

Evidence:

```text
experiments/branch50-linear-context-v0/evidence/real-corpus-1024-terminal.json
experiments/branch50-linear-context-v0/evidence/real-corpus-1024-config.json
```

The terminal checkpoint was retained locally, not committed:

```text
CHECKPOINT_BYTES=641_375_356
CHECKPOINT_SHA256=11ca6b4d6b9b4c47ecb70ead762008f87c4875c39c1d5d8a193b2ae301d18e99
```

## Executed harness identities

The executed harnesses remained outside this repository during the court.
Their hashes prevent a later rewrite from masquerading as the executed bytes:

```text
profile_linear_context.py=e2103548d43bcfb51e290dc2d50f8295405abd52fb8cc4aacc33e4ab4de4ce6d
run_branch50_linear_context_trial.py=6af5ba0402dfb91cb845eb651f211726c277b07339f325c696384f16fd692b90
launch_linear_context_gate.sh=43cafb03856845f198ab33b5ba153a90832adbae036fce97f6498c752487e7ec
shared_branch49_u16_runner=ed42a69331ebee2ff6e5d65706f2a138f8af46bf725eccf461ce2e10015f4110
```

The compact Git packet contains court output and resolved config, not the
641 MB checkpoint, U16 corpus, Python caches, or mutable MLflow spool.

## Admission boundary

```text
LINEAR_FLOP_SLOPE_512_1024_2048=PASS
LINEAR_DYNAMIC_MEMORY_SLOPE_512_1024_2048=PASS
FINITE_GRADIENTS_ALL_CONTEXTS=PASS
REAL_CORPUS_1024_100_STEPS=PASS
REALTIME_MLFLOW=PASS

1024_QUALITY_PROMOTION=NOT_ESTABLISHED
2048_REAL_CORPUS_TRAINING=NOT_RUN
1024_CHECKPOINT_RESUME=NOT_RUN
THREE_SEED_COMPARISON=NOT_RUN
MODEL_PUBLICATION=HELD
LEGACY_DEPRECATION=HELD
```

## Next lawful milestone

Run `BRANCH50_1024_QUALITY_PROMOTION_V0`:

```text
SOURCE/INITIALIZATION/DATA/TOKENIZER/OPTIMIZER/SCHEDULE=identical
RUN_A=seq512, batch12, accum7
RUN_B=seq1024, batch6, accum7
FIRST_TRANCHE=100M causal targets, seed42
VALIDATION=same exact held-out token positions
COMPARE=same-token and same-step validation NLL, throughput, VRAM, gradients, resume
```

If lawful, repeat for seeds `42`, `8675309`, and `2026`. Promote 1024 only
when its three-seed median quality is not worse, throughput is at least 90% of
the 512 control, numerical integrity is perfect, and checkpoint resume passes.

## Lighteval publication gate

The project is **Lighteval** (not LiteEval). It must run before any checkpoint
is printed, pushed, announced, or treated as a model release. Keep Lighteval in
a separate evaluation environment; do not add it to the training dependency
set merely to satisfy this gate.

As of this session, the latest stable upstream release is `v0.13.0`, while
upstream `main` reports `0.13.1.dev0`. Freeze an exact package version or Git
commit in the evaluation receipt; never infer an evaluation version from
`main`.

The checkpoint first passes:

```text
save_pretrained=PASS
AutoModelForCausalLM.from_pretrained=PASS
AutoTokenizer.from_pretrained=PASS
architectures=[HelixForCausalLM]
model/config/tokenizer roots recorded
fixed-prompt generation round trip=PASS
fixed-continuation loglikelihood court=PASS
```

Because `HelixForCausalLM` is custom, use either the Transformers/Accelerate
backend after proving clean AutoModel registration/load, or a repository-owned
`LightevalModel` adapter implementing `greedy_until`, `loglikelihood`, and
`loglikelihood_rolling`. Any `trust_remote_code=true` path must bind and review
the exact remote-code revision.

Run `--max-samples` smoke first, then the frozen checked-in task manifest with
sample-level details saved locally. Bind:

```text
lighteval version and Git SHA
task manifest SHA256
task/dataset revisions
few-shot counts and seeds
model/tokenizer/source/environment roots
results JSON root
details Parquet roots
stdout/stderr root
device identity and wall time
```

No Hub push is part of evaluation. Changing tasks after seeing results creates
a new evaluation family and cannot overwrite the original gate.

Official references:

- https://github.com/huggingface/lighteval
- https://huggingface.co/docs/lighteval/main/en/quicktour
- https://huggingface.co/docs/lighteval/main/en/evaluating-a-custom-model
- https://huggingface.co/docs/lighteval/main/en/saving-and-reading-results

## Operator vocabulary

```text
completed=the process reached a terminal
executed=the requested computation ran
verified=an independent court accepted the evidence
promoted=a declared gate admitted the candidate
published=an external release action occurred
```

A completed smoke is not a promoted model. A FINISHED MLflow run is not a
publication. A checkpoint is an artifact, not a scientific conclusion.
