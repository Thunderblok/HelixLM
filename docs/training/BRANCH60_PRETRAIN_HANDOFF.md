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
    val_texts=validation_texts,
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

