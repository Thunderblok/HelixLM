# Branch 62 pretraining handoff

## What the path does

`PretrainTrainer` is the continuous causal-pretraining path. It accepts either:

- a verified disk-backed sample store through `train_store_dir`; or
- text-like training input through `train_texts`, including a Hugging Face
  `IterableColumn`.

When an `IterableColumn` is supplied, `PretrainTrainer` compiles it into an
indexed store automatically. Supplying `pretrain_store_dir` gives that store a
durable location. Omitting it uses a private temporary store, which is useful
for bounded experiments but is not the recommended full-corpus posture.

`Trainer` remains the legacy document-aware SFT path. Branch 62 does not change
its chunking or batching behavior.

## How data becomes model input

```text
pinned ordered source rows
-> tokenize without tokenizer-added special tokens
-> append one EOS per nonempty document
-> concatenate the token stream
-> emit exact non-overlapping seq_len windows
-> discard and count the incomplete tail
-> assign stable sample IDs
-> persist one global permutation per epoch
-> exclude the fixed validation IDs
-> replay the remaining sample IDs from disk
```

The compiler publishes a new store atomically. An existing store is reused only
after its manifest, file hashes, sequence length, and declared source identity
verify. An existing invalid directory is refused rather than overwritten.

## Why the store exists

A seed alone does not identify the order presented to the model. The compiled
manifest, persisted permutation, and fixed validation-ID root make sample order
replayable and make optimizer-step checkpoints meaningful. They also allow the
storage path to be tested independently from GPU training.

There is no separate required preprocessing command. `PretrainTrainer` owns
compilation when passed an `IterableColumn`; the caller selects a durable store
path when the compiled corpus must survive beyond the process.

## Launcher roles and reviewed reference

There are two related Branch 62 launchers, and neither should be described as a
textual continuation of the other:

| Launcher | Authority | Training lifecycle | Evidence/publication posture |
|---|---|---|---|
| David's upstream `113M_param_train.py` at `1c140ce35cd53363833c301b7e556f83620643c3` | Production-style three-epoch launcher | Three one-epoch stages with learning rates `2e-4`, `6.67e-5`, and `2.22e-5` | Saves a local checkpoint per stage, then optionally publishes each stage to Hugging Face |
| Thunderblok's `113M_param_train.py` | RTX-5080 comparison and evidence launcher | One resolved profile, fixed learning rate, and a bounded optimizer-step contract | Requires MLflow by default, writes local JSONL evidence, and keeps rotating resumable local checkpoints before optional final publication |

They share the continuous pretraining data contract and much of the model
shape. Their control flow, checkpoint cadence, learning-rate lifecycle, and
operator purpose are intentionally different. A matching topology does not
make their results interchangeable; comparisons bind the exact source commit,
run contract, sample manifest, permutation, seed, optimizer, and evaluator.

In this repository, `113M_param_train.py` is the canonical executable only for
the Thunderblok Branch 62 comparison campaign. It is not a replacement for
David's upstream production launcher. The filename is historical; the emitted
run contract records the actual parameter count.

### Why the settings are resolved objects

`TrainingProfile` describes the hardware/comparison shape. `RunSettings` is the
immutable, fully resolved contract after environment overrides. Keeping those
roles separate prevents an RTX-5080 microbatch adjustment from being mistaken
for a change to the model or data law, and makes the exact executed values
available to tests and MLflow.

### RTX 5080 relative profile (default)

```text
d_model=768
n_heads=12
batch=3
gradient_accumulation=28
effective_batch=84
sequence_length=1024
learning_rate=2e-4
FFN expansion=3.0
```

This is the practical 16 GB shape. It references MLflow run
`0abd42410cce4fc880009320c4663287`, but it is not described as an exact replay
of that bounded run.

```bash
HELIX_PRETRAIN_STORE_DIR=/data/sutra-gpt2-t1024 \
HELIX_PROFILE=rtx5080-relative \
python 113M_param_train.py
```

### Branch 60 exact shape

```text
d_model=1024
n_heads=16
batch=2
gradient_accumulation=42
effective_batch=84
sequence_length=1024
learning_rate=2e-4
FFN expansion=3.0
```

This matches the model/data/optimizer shape of MLflow run
`6ad46206ff1d49a3a96d71fd7723f16b`. Branch 62 remains a new executable source,
so matching configuration does not relabel it as the earlier run.

```bash
HELIX_PRETRAIN_STORE_DIR=/data/sutra-gpt2-t1024 \
HELIX_PROFILE=branch60-exact-shape \
python 113M_param_train.py
```

The exact-shape profile retains gradient accumulation 42 only to reproduce the
earlier effective batch on the wider model. The default 5080 profile uses 28.

## Checkpoints and restart

The launcher writes rotating local checkpoints before any optional external
publication. An indexed training state binds model, optimizer, scheduler, AMP
scaler, Torch RNG, optimizer step, sample cursor, sample-manifest root,
permutation root, and validation-ID root.

Resume requires the original verified store:

```bash
HELIX_PRETRAIN_STORE_DIR=/data/sutra-gpt2-t1024 \
HELIX_RESUME_TRAINING_STATE=/runs/checkpoints/latest-0/pretrain_training_state.pt \
python 113M_param_train.py
```

A mismatch is a refusal. Checkpoints from the earlier external adapter are
evidence artifacts, not certified Branch 62 resume inputs.

## MLflow and local evidence

MLflow admission is required by default and begins before training. Every metric
event is also appended to `mlflow-events.jsonl`; local checkpoints and the local
terminal remain the custody authority. The launcher records source identity,
data roots, causal-target counts, throughput, loss, perplexity, learning rate,
gradient norm, VRAM, and the lateral/vertical graph parameters.

Hugging Face upload is opt-in through `HELIX_PUSH_TO_HUB=1` and occurs only
after the final model has been saved locally. The generated model name is at
most 96 characters and encodes width, columns, configured nodes, loops, FFN
expansion, sequence length, and epoch count.

## Known comparison ceiling

`nodes_per_column=(3, 3, 3)` is currently recorded as configured but not claimed
to alter graph construction on this branch. The run contract therefore records
`nodes_per_column_graph_effective=false`. Wiring node counts into the graph is a
separate model-topology change and needs a matched ablation.

## Review checklist for upstream integration

David's review is most useful on the boundary between the two launchers:

1. Confirm which production lifecycle remains upstream authority: staged
   epochs, learning-rate decay, and per-epoch Hugging Face checkpoints.
2. Confirm which evidence mechanics should eventually move upstream: exact
   source identity, sample/permutation roots, local resumable custody, JSONL
   metrics, and MLflow projection.
3. Keep hardware batch and accumulation choices profile-specific while holding
   effective batch, sequence length, tokenizer, corpus order, and evaluator
   fixed for matched comparisons.
4. Decide separately whether topology randomness should receive its own seed.
   The current `seed` influences more than data order, so changing it is not yet
   a pure multi-seed replication.

Dependency upgrades are a separate compatibility campaign. Updating PyTorch,
CUDA-facing wheels, Transformers, Datasets, or evaluation libraries must not be
smuggled into this launcher-documentation change or compared against an older
run as though only the launcher wording changed.
