# Documentation map

## Purpose

Documentation in this tree explains reproducible Helix behavior. It must not
turn an experiment result, proposed topology, or optional publication step into
a production claim.

## Where training records belong

- Put operator-facing training contracts and handoffs in `training/`.
- Keep raw logs, checkpoints, datasets, MLflow spools, and generated manifests
  outside the repository. Link them by immutable identity when needed.
- Keep the canonical executable path in source. Documentation may explain it,
  but must not become a second implementation.

## Evidence language

- Bind comparisons to source commit and tree, corpus revision, tokenizer,
  sample-manifest root, permutation root, architecture, seed, optimizer, and
  evaluator.
- Call a run complete only after its terminal and artifacts are read back.
- Separate observed metrics from estimates. In particular, raw UTF-8 exposure
  derived from sample position is an estimate; causal-target counts are exact.
- A local checkpoint is the custody authority. Hugging Face and MLflow are
  projections and must never be the only copy of model or run state.

## Current topology ceiling

The Branch 60 indexed pretraining comparison freezes `d_model=1024`, three
columns, `(3, 3, 3)` nodes, and vertical depth two. A fourth column and vertical
depth three are deferred to a separately matched experiment. Do not describe
them as part of the current run.
