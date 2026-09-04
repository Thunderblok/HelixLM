# `helix_lm` module guide

## Where changes belong

| Concern | Module |
| --- | --- |
| Model-wide configuration | `config.py` |
| Graph construction and topology | `graph.py` |
| Node implementations | `nodes.py` |
| Hugging Face model adapter | `hf_model.py` |
| Tokenizer adapter | `tokenizer.py` |
| Supervised/document-aware data | `dataset.py` |
| Pretraining sample store and order | `pretrain_data.py` |
| Training loops | `trainer.py` |

## Trainer boundary

`Trainer` is the legacy document-aware SFT implementation. Preserve its
document boundaries, padding masks, overlap semantics, and public call shape.
Do not rename it here; David owns the later `Trainer` to `SFTTrainer` migration.

`PretrainTrainer` is the only trainer that may consume the continuous
pretraining sample store. Its canonical data contract is:

```text
ordered source rows
-> tokenize without tokenizer-added specials
-> append one EOS per nonempty document
-> concatenate
-> exact seq_len windows
-> discard the incomplete tail
-> persist stable sample IDs
-> replay a persisted global permutation
```

Checkpoint data state must identify both the compiled sample manifest and the
permutation. A model checkpoint without those roots is not an exact mid-epoch
resume receipt.

For indexed training, `pretrain_training_state.pt` is the exact local recovery
object at an optimizer-step boundary: model, optimizer, scheduler, AMP scaler
when present, Torch RNG, global step, sample cursor, manifest root, and
permutation root. A new epoch must activate a new persisted epoch permutation;
reusing the prior epoch's DataLoader is not a global-shuffle implementation.

## Deferred topology work

Do not add a fourth column or raise vertical depth to three in this lane. The
current comparison freezes three columns, `(3, 3, 3)` nodes, and vertical depth
two. Future topology work gets its own matched ablation after the pretraining
pipeline is proven.
