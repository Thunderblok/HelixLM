# Continuous pretraining

## Choose the trainer that matches the data contract

Use `Trainer` for document-aware supervised fine-tuning. It preserves document
boundaries, pads variable-length examples, and supplies attention masks.

Use `PretrainTrainer` for continuous causal pretraining. It joins nonempty
documents with one EOS token, emits exact non-overlapping windows, and trains
without padding.

Neither path silently delegates to the other.

## Accepted pretraining inputs

`PretrainTrainer` accepts either:

- text-like input through `train_texts`, including a list of strings or a
  Hugging Face `IterableColumn`; or
- a verified disk-backed sample store through `train_store_dir`.

An `IterableColumn` is compiled automatically. Set `pretrain_store_dir` when
the resulting indexed store must survive beyond the current process. Without
that argument, the trainer uses a private temporary store suitable for bounded
tests and demonstrations.

```python
from datasets import load_dataset
from helix_lm import HelixConfig, HelixForCausalLM, HelixTokenizer, PretrainTrainer

tokenizer = HelixTokenizer("gpt2")
config = HelixConfig.small_v2(
    vocab_size=len(tokenizer),
    tokenizer_name="gpt2",
    seq_len=1024,
)
model = HelixForCausalLM(config)

rows = load_dataset("your/dataset", split="train", streaming=True)
trainer = PretrainTrainer(
    model=model,
    cfg=config,
    tokenizer=tokenizer,
    train_texts=rows["text"],
    pretrain_store_dir="./pretrain_store",
    output_dir="./checkpoints",
)
trainer.train(num_epochs=1)
```

To reuse an existing store, omit `train_texts` and pass its directory:

```python
trainer = PretrainTrainer(
    model=model,
    cfg=config,
    tokenizer=tokenizer,
    train_texts=None,
    train_store_dir="./pretrain_store",
    output_dir="./checkpoints",
)
```

An existing store is reused only after its manifest, file hashes, sequence
length, and declared source identity verify. An invalid store is refused rather
than overwritten.

## Data and sample-order contract

```text
pinned ordered source rows
-> tokenize without tokenizer-added special tokens
-> append one EOS to each nonempty document
-> concatenate the token stream
-> emit exact non-overlapping seq_len windows
-> discard and count the incomplete tail
-> assign stable sample IDs
-> persist one global epoch permutation
-> exclude declared validation IDs
-> replay the remaining sample IDs from disk
```

A seed alone does not identify the order presented to the model. Reproducible
training binds the compiled manifest, persisted permutation, validation-ID
root, tokenizer, source revision, and sequence length.

## Single-GPU launcher

Run `113M_param_train.py` for the supported step-bounded single-GPU path. The
filename is historical; the emitted run contract records the measured parameter
count and resolved settings.

The checked-in `single-gpu-16gb` defaults use:

```text
d_model=768
n_heads=12
batch=3
gradient_accumulation=28
effective_batch=84
sequence_length=1024
learning_rate=2e-4
FFN expansion=3.0
lateral probability=0.8
vertical probability=0.9
vertical depth=2
```

```bash
HELIX_PRETRAIN_STORE_DIR=/data/pretrain-gpt2-t1024 \
python 113M_param_train.py
```

The defaults are ordinary module-level constants near the top of the file.
Every setting that changes the executable subject has a matching environment
override. The most commonly changed groups are:

| Group | Environment variables |
|---|---|
| Data | `HELIX_DATASET`, `HELIX_DATASET_REVISION`, `HELIX_DATASET_SPLIT`, `HELIX_TEXT_COLUMN`, `HELIX_TOKENIZER` |
| Model | `HELIX_D_MODEL`, `HELIX_N_HEADS`, `HELIX_N_COLUMNS`, `HELIX_NODES_PER_COLUMN`, `HELIX_N_LOOPS`, `HELIX_SEQUENCE_LENGTH`, `HELIX_FFN_EXPANSION` |
| Topology | `HELIX_LATERAL_P`, `HELIX_VERTICAL_P`, `HELIX_VERTICAL_DEPTH` |
| Attention | `HELIX_ATTENTION_MODE`, `HELIX_LOCAL_WINDOW`, `HELIX_COARSE_WINDOW`, `HELIX_COMPRESSED_WINDOWS`, `HELIX_COMPRESSED_VIEWS` |
| Optimization | `HELIX_BATCH_SIZE`, `HELIX_GRAD_ACCUM`, `HELIX_EPOCHS`, `HELIX_LEARNING_RATE`, `HELIX_WARMUP_MICROBATCHES`, `HELIX_MAX_OPTIMIZER_STEPS` |
| Evidence | `HELIX_MLFLOW_URI`, `HELIX_MLFLOW_EXPERIMENT`, `HELIX_REQUIRE_MLFLOW`, `HELIX_CHECKPOINT_EVERY`, `HELIX_EVAL_EVERY` |

Use `HELIX_PRINT_CONTRACT=1` to inspect the fully resolved configuration without
loading data or allocating a model:

```bash
HELIX_D_MODEL=1024 \
HELIX_N_HEADS=16 \
HELIX_BATCH_SIZE=2 \
HELIX_GRAD_ACCUM=42 \
HELIX_PRINT_CONTRACT=1 \
python 113M_param_train.py
```

Treat any override set as a different executable subject even when effective
batch size is unchanged. Compare runs only when source, data order, tokenizer,
optimizer, seed, and evaluator are all bound.

## Checkpoints and resume

The launcher rotates local checkpoints before any optional publication. A
training-state checkpoint binds model, optimizer, scheduler, AMP scaler, Torch
RNG, optimizer step, sample cursor, sample-manifest root, permutation root, and
validation-ID root.

Resume requires the original verified store:

```bash
HELIX_PRETRAIN_STORE_DIR=/data/pretrain-gpt2-t1024 \
HELIX_RESUME_TRAINING_STATE=/runs/checkpoints/latest-0/pretrain_training_state.pt \
python 113M_param_train.py
```

A mismatched store, order, validation set, or training contract is refused.

## Metrics and publication

The launcher records source identity, resolved model and graph settings, sample
roots, causal-target counts, throughput, loss, perplexity, learning rate,
gradient norm, and GPU memory. MLflow is a projection of the append-only local
JSONL record, not the only copy of run evidence.

Hugging Face publication is disabled by default. When enabled, the final model
and tokenizer are saved locally first and then published under a bounded,
configuration-bearing name. Publication requires only `HF_USER` and `HF_TOKEN`:

```bash
HF_USER=your-account \
HF_TOKEN=... \
HELIX_PUSH_TO_HUB=1 \
python 113M_param_train.py
```

## Executable topology contract

`nodes_per_column` controls the number of compute nodes instantiated in each
column. Aggregation gate nodes are added separately and do not consume that
budget. The launcher reads the configured and observed counts back from the
constructed graph and refuses to train when either differs from the requested
tuple. Admitted runs therefore report both `nodes_per_column` and
`observed_nodes_per_column`, with `nodes_per_column_graph_effective=true` only
after that equality check passes.

Changing the tuple changes graph topology, parameter count, and checkpoint
compatibility. A node-count comparison must still freeze source, data order,
tokenizer, optimizer, seed, evaluator, and token or compute budget.
