# Helix Automata State Capacity Proof V0

## Claim under test

A fixed Helix checkpoint may retain useful information from longer histories
when supplied with bounded deterministic state and retrieval, without treating
external state storage or additional computation as free.

This experiment does **not** claim that automata place more tokens inside model
parameters. It measures whether the same learned parameters can service more
history under explicit state, storage, retrieval, latency, and FLOP accounting.

## Frozen baseline

```text
dataset=codelion/sutra-10B
dataset_revision=415549cff1a92b69df8b88c6108faa6097457068
tokenizer=gpt2
sequence_length=1024
d_model=768
heads=12
loops=3
topology=(2,3,2)
ffn_expansion=2.5
parameters=101,228,948
```

The Sutra run trains the base checkpoint. It is not the long-memory evaluation
corpus and cannot establish historical recall by itself.

The admitted baseline budget is 1,500,000,000 requested causal targets, aligned
to 1,500,028,992 targets across 45,822 complete optimizer steps at batch 4 and
gradient accumulation 8. The learning rate warms linearly over 2,000
microbatches (250 optimizer steps) to `1.5e-4`, then remains constant. A
different budget or schedule produces a different checkpoint identity.

The streaming packer binds an exact dataset revision and ordered row stream.
It checkpoints only at full 1,024-token boundaries and records the next unread
row/token offset, raw UTF-8 bytes read, emitted tokens, causal targets, and
sequence count. Resume must recreate the uninterrupted token sequence exactly.

## Observer boundary

`StateProbeV0` observes detached hidden states in 64-token segments. It writes
32 bounded registers, exact source-token roots, state size, and transition
roots. It has no model-facing return path and therefore cannot affect baseline
training or inference.

## Controlled comparison

One frozen checkpoint and evaluator are compared:

```text
A = local-context transformer
B = A + retrieval
C = A + deterministic automata state + DAG retrieval
```

All arms use the same checkpoint, tokenizer, case order, model context bound,
answer scorer, and repetition count. Arm B retrieves at most two raw observed
event documents using deterministic lexical overlap. It cannot retrieve a
derived transition that was never present in the raw history. Arm C replays
typed benchmark events into a bounded current-state map, validates parent and
supersession references, and retrieves that state plus the exact causal path.
This is an oracle typed-state experiment, not a claim that V0 extracts those
events from arbitrary prose.

The long-history corpus places benchmark-owned facts, rules, goals,
contradictions, and causal edges at controlled distances of 10K, 50K, 250K,
and 750K tokens. The automaton begins with typed benchmark-owned registers;
free-form neural semantic extraction is outside V0.

## Required accounting

```text
accuracy_by_history_distance
raw_history_bytes
live_context_bytes
kv_cache_bytes
automata_state_bytes
transition_log_bytes
retrieved_bytes
state_transition_latency
retrieval_latency
inference_flops
active_parameters
unavailable_or_invalid_transitions
```

`kv_cache_bytes` is zero only if the frozen Helix model returns no
`past_key_values`; the evaluator refuses a surprise cache. FLOPs are explicitly
reported as an estimate using
`2 × active_parameters × input_tokens × recurrent_loops`, not as a profiler
measurement. Model quality is scored by teacher-forced exact answer-token
accuracy and mean answer-token negative log likelihood.

```text
state_compression_ratio =
  raw_history_bytes /
  (live_context_bytes + candidate_live_state_bytes + retrieved_bytes)
```

The append-only transition log is reported separately and is never treated as
free. Durable lineage storage and retrieval cost remain part of the terminal.

Promotion requires C to improve accuracy at long distance after all external
state, retrieval, and computation costs are reported. A passing state-contract
court alone does not establish model benefit.
