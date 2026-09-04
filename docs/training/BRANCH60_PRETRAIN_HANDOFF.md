# Branch 60 pretraining handoff

## Why this change exists

Branch 60's `PretrainTrainer` originally wrapped a streaming text iterator in
`ContinuousWindowDataset`. That preserved the desired EOS-joined continuous
windows, but its 50,000-window reservoir was only an approximate shuffle. With
multiple DataLoader workers, each worker could also replay the same unsplit
iterable. The result was fast to prototype but could not establish a global
sample order or exact mid-epoch replay.

The admitted path compiles the continuous stream once into disk-backed uint16
windows. Every window receives a stable sample ID. Each epoch then receives one
persisted uint32 permutation, and DataLoader workers perform random access over
that shared order. The permutation file—not an independently repeated random
call—is the comparison authority.

## What did not change

- `Trainer` and its document-aware SFT chunking remain intact.
- David owns the future rename of `Trainer` to `SFTTrainer`.
- Tokenizer behavior remains GPT-2 unless the run contract explicitly says
  otherwise.
- Model topology remains three columns, `(3, 3, 3)` nodes, and vertical depth
  two for the current campaign.
- The fourth-column/depth-three experiment is deferred.

## Compile a sample store

```bash
python prepare_pretrain_dataset.py \
  --dataset codelion/sutra-10B \
  --revision 415549cff1a92b69df8b88c6108faa6097457068 \
  --split train \
  --tokenizer gpt2 \
  --seq-len 1024 \
  --output-dir /path/to/sutra-gpt2-t1024
```

The output manifest binds source identity, sequence length, EOS identity,
source rows observed, encoded tokens observed, sample count, causal target
count, dropped tail, shard byte sizes, and shard hashes.

## Use it from `PretrainTrainer`

```python
trainer = PretrainTrainer(
    model=model,
    cfg=cfg,
    train_store_dir="/path/to/sutra-gpt2-t1024",
    train_texts=None,
    val_texts=None,
    validation_sample_count=252,
    tokenizer=tokenizer,
    seed=42,
    train_permutation_epoch=0,
    train_cursor=0,
    num_workers=4,
    grad_accum_steps=42,
    use_amp=True,
    amp_dtype="bfloat16",
)
```

The trainer creates or reuses the epoch permutation under the sample store and
writes `pretrain_data_state.json` beside each local model checkpoint. Publish a
checkpoint to Hugging Face only after the local checkpoint and data-state file
exist and have been read back.

Indexed checkpoints also contain `pretrain_training_state.pt`, which binds and
restores model, optimizer, scheduler, AMP scaler when present, Torch CPU/CUDA
RNG, global step, sample cursor, dataset-manifest root, and permutation root.
Pass that file through `resume_training_state` with the same sample store. The
trainer refuses a manifest, permutation, or epoch-identity mismatch. Each later
epoch activates a separately persisted permutation derived from the same seed
plus its epoch number. Treat the object as an exact recovery point only when it
was written at an optimizer-step boundary.

The launcher also writes `ckpt_epochN` using `save_pretrained()`. That directory
contains Hugging Face model/tokenizer files; it is not an exact indexed recovery
object unless the two indexed state files are present and read back beside it.

## Run the checked-in launcher

```bash
HELIX_PRETRAIN_STORE_DIR=/path/to/sutra-gpt2-t1024 \
HELIX_DATASET=codelion/sutra-10B \
HELIX_DATASET_REVISION=415549cff1a92b69df8b88c6108faa6097457068 \
HELIX_VALIDATION_SAMPLES=252 \
HELIX_PUSH_TO_HUB=0 \
python 113M_param_train.py
```

Resume with the same sample store, dataset, and revision used for the original
run:

```bash
HELIX_PRETRAIN_STORE_DIR=/path/to/sutra-gpt2-t1024 \
HELIX_DATASET=codelion/sutra-10B \
HELIX_DATASET_REVISION=415549cff1a92b69df8b88c6108faa6097457068 \
HELIX_VALIDATION_SAMPLES=252 \
HELIX_RESUME_TRAINING_STATE=/path/to/pretrain_training_state.pt \
HELIX_PUSH_TO_HUB=0 \
python 113M_param_train.py
```

For the indexed path, validation is the fixed tail of the epoch-zero persisted
permutation. Those IDs are excluded from every training epoch even as later
epochs activate new persisted orders. The checkpoint binds the validation ID
root and count. The launcher resumes the checkpoint's LR stage, refuses
incompatible scheduler, runtime, validation, or data roots, and then continues
later stages. It does not create an MLflow run by itself. Bind MLflow externally,
or add explicit launcher support before calling the resulting run comparable.

## Overnight admission run

The first indexed-data comparison should be bounded explicitly rather than
described as a full Sutra epoch. At roughly 5,700 causal targets per second, a
250 million-target budget is approximately twelve hours. Record the exact
sample limit and permutation root in MLflow. Compare throughput and loss only
against a run with the same model, optimizer, corpus revision, tokenizer,
sequence length, sample order, and evaluator.

Required MLflow parameters include:

```text
source_head
source_tree
dataset
dataset_revision
sample_manifest_sha256
permutation_sha256
tokenizer
seq_len
d_model
n_columns
nodes_per_column
n_loops
ffn_expansion
lateral_p
vertical_p
vertical_depth
batch_size
grad_accum_steps
effective_batch_size
learning_rate
```

Required metrics include causal targets per second, raw input bytes per second,
loss, perplexity, learning rate, gradient norm, skipped batches, GPU memory,
GPU utilization, data wait time, step time, and checkpoint write time.

## Independent data-path court

`pretrain_data_court.py` keeps admission evidence outside the trainer. Its
fixture phase independently builds live continuous windows and compiled
windows, then proves exact token, label, mask, persisted-order, and causal
target equivalence. Its optional complete-store phase replays every persisted
sample ID, rejects duplicates and omissions, hashes the observed ordered IDs
and tokens, and measures storage-only throughput.

The default performance floor is 25 samples per second. That is deliberately a
data-supply threshold rather than a model-speed promise: the admitted Branch 60
configuration consumes approximately 6.1 samples per second at roughly 6,200
causal targets per second, so the storage path must retain at least fourfold
headroom. Record the measured terminal outside an active training window; a
concurrent replay would contaminate both measurements.
