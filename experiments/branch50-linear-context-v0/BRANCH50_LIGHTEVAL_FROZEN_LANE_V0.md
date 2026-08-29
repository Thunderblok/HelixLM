# Branch 50 Frozen Lighteval Lane V0

Status: `PREPARED_BY_SOURCE_NOT_EXECUTED`

This lane prepares a reproducible Lighteval packet for the final selected
Branch-50 checkpoint. It does not run a GPU evaluation, does not push results,
and does not turn a smoke sample into a comparative benchmark.

## Required sequence

```text
1. Run lighteval_checkpoint_preflight.py on the selected checkpoint.
2. Feed the resulting save_pretrained export and PASS receipt to
   prepare_lighteval_frozen_lane.py.
3. Inspect the generated manifest, YAML, command JSON, and run script.
4. Run the generated smoke command only if a smoke/non-comparative check is
   desired.
5. For final evaluation, create a new manifest without --max-samples and bind
   the full results/details directories as artifacts.
```

## Frozen dependency contract

```text
lighteval=0.13.0
transformers=5.8.1
trust_remote_code=false
push_to_hub=false
push_to_tensorboard=false
public_run=false
wandb=false
```

The helper refuses to prepare the lane if the installed Lighteval or
Transformers versions do not match the pinned values.

## Model contract

The generated YAML uses the preflight export path for both model and tokenizer:

```text
model_name=<preflight export dir>
tokenizer=<preflight export dir>
revision=main
max_length=512
dtype=float32 by default
device=cuda by default
trust_remote_code=false
compile=false
add_special_tokens=false
pairwise_tokenization=false
```

The preflight receipt must already prove:

```text
checkpoint model root matches observed root
save_pretrained export reloads with same model root
fresh-process registered-source reload passes
parameter_count=53592340
seq_len=512
tokenizer=gpt2
pad/eos/bos token IDs=50256
```

## Dataset and tokenizer identity

```text
dataset_repo=david-thrower/HelixLM-medium-1500.0Mt-2988750pt-20260528
dataset_revision=bd85adc4fddfd33f5ccb8ce8e58cad2c0251185b
train_manifest_sha256=b67f33931c0e545c8701166dbf990a7af64cf1c3966c5500d20bd2381bc9b115
val_manifest_sha256=2c15971275e2834e378ea358fc2acf05f7251d2199ebfa8854e5974a29f7932b
tokenizer=gpt2
tokenizer_local_cache_ref=607a30d783dfa663caf39e06633721c8d4cfcd7e
```

## Task and sample contract

Default smoke task:

```text
tasks=arc:easy|0
max_samples=16
```

Any command using `--max-samples` is explicitly:

```text
smoke_only=true
comparative_benchmark=false
```

Final reporting must bind:

```text
source_head
source_tree
checkpoint_root
checkpoint_sha256
tokenizer identity
model YAML SHA-256
command JSON SHA-256
Lighteval version
Transformers version
task string
results directory root
details directory root
```

## Example dry-run preparation

```bash
/home/mo/DEV/experiments/helix-branch50-linear-context-v0/.venv-lighteval/bin/python \
  experiments/branch50-linear-context-v0/prepare_lighteval_frozen_lane.py \
  --export-dir /path/to/preflight-export \
  --preflight-receipt /path/to/preflight-receipt.json \
  --output-dir /path/to/lighteval-packet \
  --lighteval-bin /home/mo/DEV/experiments/helix-branch50-linear-context-v0/.venv-lighteval/bin/lighteval
```

The generated `run_branch50_lighteval_smoke.sh` is executable but is not run by
the prep step unless `--execute` is supplied.
