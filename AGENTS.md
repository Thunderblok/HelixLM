# HelixLM working map

## Product boundaries

- `helix_lm/` contains the reusable model, dataset, and trainer implementation.
- `113M_param_train.py` is the Branch 60 production-pretraining entry point.
- `quick_demo_cpu_multiscale.py` is a bounded CPU demonstration, not the
  full-corpus data authority.
- Experiment outputs, checkpoints, downloaded corpora, and MLflow artifacts do
  not belong in this repository.

## David's current ownership and compatibility decisions

- Keep `helix_lm.trainer.Trainer` behavior unchanged. Its document-aware
  chunking is the supervised-fine-tuning path. David owns its eventual rename
  to `SFTTrainer`.
- Pretraining work belongs in `PretrainTrainer` and pretraining-specific data
  components. Do not make the legacy Trainer call the pretraining compiler.
- Keep the current experiment topology at three columns and vertical depth two.
  A fourth column and depth three are a later, separately admitted ablation.
- Width takes priority over adding topology in the current campaign. The active
  Branch 60 configuration uses `d_model=1024`.
- Compile data locally, save checkpoints locally first, and make Hugging Face
  publication an explicit optional action.

## Data truth

- Pretraining means an EOS-joined continuous token stream, exact non-overlapping
  windows, no padding, and causal labels equal to the input tokens.
- A shared seed is not proof of shared sample order. Persist the exact epoch
  permutation and bind its hash in run/checkpoint metadata.
- Disk and in-memory paths must agree on sample IDs, batch boundaries, token
  tensors, and causal-target counts.
- Dataset identity includes repository name, revision, split, text column,
  tokenizer identity, sequence length, sample manifest root, and permutation
  root.

## Validation before a long run

Run the pretraining data courts, compile a bounded corpus slice, replay it with
multiple DataLoader workers, and perform at least one real optimizer step. A
run is not comparable unless MLflow records source, model topology, data roots,
batching, learning rate, FFN expansion, lateral/vertical probabilities, and
vertical depth.

`pretrain_data_court.py` is the independent sample-equivalence and storage
throughput terminal. Run its fixture court during ordinary verification. Run
its complete-store replay only while training is idle; otherwise the benchmark
would perturb the GPU run whose input path it is qualifying.
