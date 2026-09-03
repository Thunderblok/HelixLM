# Experimental LengthMAX tokenizer adapter

This isolated Branch-49 source copy adds an explicit tokenizer backend for the
matched-vocabulary pilot:

```python
tokenizer = HelixTokenizer(
    "lengthmax:/absolute/path/to/iterative-hybrid-tokenizer.json"
)
```

The adapter is experimental. It does not replace the default GPT-2 tokenizer
and it never falls back to GPT-2 when its artifact is absent or malformed.

## Contract

- Algorithm: `iterative-byte-bpe-vocab-leftmost-longest-v0`
- Vocabulary size: `50_257`, exactly matching GPT-2
- Byte tokens: ids `0..255`
- Learned multi-byte tokens: ids `256..50_255`
- `<|endoftext|>`: id `50_256`
- `pad`, `eos`, `bos`, and `unk`: id `50_256`, matching the existing GPT-2
  training posture
- `encode(add_special_tokens=True)`: does not insert tokens, matching GPT-2
- Text encoding: UTF-8 bytes with deterministic leftmost-longest matching
- Checkpoints: copy the tokenizer artifact and bind its SHA-256 in
  `helix_tokenizer_config.json`
- Reload: `HelixTokenizer.from_pretrained(checkpoint_dir)` verifies the hash and
  the complete tokenizer identity before use

## Exact experimental artifact

```text
path=/home/mo/DEV/experiments/helix-lengthmax-david-v1/iterative-hybrid-dev/iterative-hybrid-tokenizer.json
sha256=5b2aca3a56fd9588eac88eb832ffd3491c14bc55cac06967cee05910537e52d1
```

The artifact was selected from training data only. Its earlier held-out
token-count result is evidence for tokenizer efficiency, not model-quality
promotion.

## Verification

```bash
/home/mo/DEV/experiments/helix-branch49-5080-scaling-v0/.venv/bin/python \
  -m unittest -v test_lengthmax_tokenizer.py
```

The real-artifact smoke decoded 1,000 source rows exactly, covering 5,082,141
UTF-8 bytes and 875,868 tokenizer tokens. Checkpoint save/reload also passed.

## Pilot boundary

The matched-model pilot must train both arms from scratch with identical model
shape, corpus order, optimizer, seed, and evaluation. It must record both raw
bytes observed and causal target tokens observed. A tokenizer-efficiency result
alone is not a model-quality result.

