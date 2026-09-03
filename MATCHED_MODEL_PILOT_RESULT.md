# Matched GPT-2 versus LengthMAX model pilot

## Terminal

```text
status=PASS
pilot_pair_id=matched-tokenizer-b49-s110-20260903T160530Z
source_head=fb39c9e8238c5915056bd0eca5a5d5067a9885b5
source_tree=23222652d14a397475445e9eccf496cd3cae329c
pair_terminal_root=9833201d2ff14dae162f597dcc769e4fa89239977ab4d7d260acf1392593c317
pair_terminal_sha256=c90ffb7e67eb30462c7992a1e48af45f8b0a8b77401d8abde3e63f9e41b19130
production_effect=none
```

Both arms used the same Branch-49 model configuration, parameter count, seed,
optimizer, learning-rate schedule, causal-target budget, raw corpus row order,
and validation routine. The independently created models had the same initial
state root:

```text
initial_model_state_root=9daf304721cea1064ee8e726fb646cddd5974de02d0e916781a3cd47d8851b5f
resolved_config_root=449a8ba0d3826c5c52c40f487e6d786b7541ba0fbdcb90c51f5a00f3c243ac1a
parameters=53,592,340
optimizer_steps_per_arm=110
causal_targets_per_arm=4,721,640
```

## Result

| Arm | MLflow run | Estimated raw bytes seen | Validation loss | Validation PPL | Estimated validation bits/byte |
| --- | --- | ---: | ---: | ---: | ---: |
| GPT-2 | `136aa2e8c34343debe53dcc10ee2f991` | 21,068,484 | 8.369762 | 4,314.61 | 2.680731 |
| LengthMAX | `e3f49fed9a8f45fa8263b5fe9c7b982b` | 26,804,725 | 9.888589 | 19,704.23 | 2.487990 |

At equal causal-target compute, LengthMAX exposed approximately 27.23% more
raw UTF-8 bytes and reduced estimated validation bits/byte by approximately
7.19%.

The PPL values are not cross-tokenizer scores. Each loss is measured per token,
and the token units differ. PPL remains useful only for comparing checkpoints
within the same tokenizer arm.

## Evidence bounds

- Raw train and validation row counts, raw UTF-8 byte counts, and
  length-prefixed source-record stream hashes matched across arms.
- Raw-byte exposure and bits/byte are estimates derived from each complete
  materialized split's tokens-per-raw-byte ratio. They are not exact byte
  attribution for each selected token window.
- Both arms were still in linear warmup at step 110; warmup completes at
  optimizer step 285.
- GPT-2 ran first. A longer admission experiment should reverse or alternate
  arm order to control for thermal and temporal host effects.
- No downstream Lighteval task result is claimed by this pilot.

## Disposition

This pilot supports continuing the LengthMAX model-training investigation. It
does not support replacing GPT-2 in production. The next experiment should use
the same pair contract, train beyond warmup, alternate arm order, and add exact
byte-accounted evaluation before any tokenizer promotion decision.

## Durable artifacts

```text
pair_terminal=/home/mo/DEV/experiments/helix-lengthmax-matched-pilot-artifacts-v0/matched-tokenizer-b49-s110-20260903T160530Z/pair-terminal.json
gpt2_terminal_sha256=47d526d9addc47d77864de879823f8397ef2ce857faa6949a4661df401baaba5
lengthmax_terminal_sha256=5bc135c7c5c5cded85a8efd547ed7ad3db15b199f6dd56b46e0b95c3b58b8a3c
gpt2_log_sha256=b1dcb84b57c08d94333a199bf657cf6e6af6ed5c934285ac0a2ebfc961fc95ac
lengthmax_log_sha256=b1e7173519cc19db99394676f9e6f01141c31c19cfaef6bd8f045494ce328966
```
