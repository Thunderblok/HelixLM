# Training documentation guide

## David's compatibility decisions

- `helix_lm.trainer.Trainer` remains the legacy document-aware SFT path. Do not
  change or rename it here; David owns the later `SFTTrainer` rename.
- Continuous, globally ordered causal pretraining belongs to
  `PretrainTrainer`, `pretrain_data.py`, and `prepare_pretrain_dataset.py`.
- Width is the current priority. The admitted Branch 60 run uses
  `d_model=1024`, 16 heads, three columns, `(3, 3, 3)` nodes, four loops, FFN
  expansion 3.0, and vertical depth two.
- Do not add a fourth column or raise vertical depth to three until the current
  data path and baseline are terminalized. That future topology change needs a
  matched source, data order, optimizer, tokenizer, and evaluator.
- Save checkpoints locally before any optional Hugging Face upload. Publication
  is never a prerequisite for local training or recovery.

## Pretraining data contract

The comparable disk path is:

```text
pinned ordered source rows
-> tokenize without tokenizer-added special tokens
-> append one EOS to each nonempty document
-> concatenate the token stream
-> emit exact non-overlapping seq_len windows
-> discard and count the incomplete tail
-> assign stable sample IDs
-> persist one global epoch permutation
-> replay that exact order from disk
```

Do not use a bounded reservoir shuffle as a synonym for a global shuffle. Do
not infer sample-order identity from a shared seed alone.

## Run and checkpoint contract

Record at minimum:

- source commit, source tree, and canonical launcher hash;
- dataset repository, revision, split, and text column;
- tokenizer and vocabulary identity;
- sample-manifest, permutation, and validation-ID roots;
- sequence length, width, columns, nodes, loops, FFN expansion, lateral and
  vertical probabilities, and vertical depth;
- optimizer, learning rate, effective batch, precision, and seed;
- exact causal-target counts, loss, perplexity, throughput, data wait, step
  time, checkpoint time, GPU utilization, and memory;
- checkpoint cursor plus RNG, optimizer, data-manifest, and permutation state.

Count causal next-token targets with `labels[:, 1:] != -100`. A checkpoint is
not an exact resume object unless its sample cursor agrees with the optimizer
step and it binds the same run contract.

## Validation and comparison law

- Validation samples must be disjoint from training samples and fixed by ID.
- Write a recoverable local checkpoint before entering a scheduled validation.
- Compare runs only through the same evaluator and the same validation IDs.
- A lower training loss is promising, not a downstream-quality verdict.
- Missing MLflow telemetry is a tracking defect; it does not erase local
  checkpoints, but a new long run should refuse to start if its MLflow run
  cannot be created.

## Active Branch 60 experiment

The exact current source and results are recorded in the run contract and
MLflow, not inferred from this guide. The current campaign uses Sutra-10B at a
pinned revision, GPT-2 tokenization, sequence length 1024, BF16, learning rate
`2e-4`, effective batch 84, and a fixed tail-of-permutation validation holdout.
See `BRANCH60_PRETRAIN_HANDOFF.md` for the implementation rationale and operator
entry points.
