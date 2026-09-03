# Hugging Face checkpoint publication V0

Training and publication are separate operations. The training runner first
writes its resumable local checkpoint. `lighteval_checkpoint_preflight.py`
then converts that checkpoint to a Transformers directory and proves an exact
save/reload model-state match. Only that admitted export may be passed to
`publish_hf_checkpoint.py`.

The publisher stages locally by default. It performs a network write only when
`--upload` is supplied, reads the credential only from `HF_TOKEN` (or the
explicit `--token-env` name), creates a private model repository unless
`--public` is supplied, uploads the complete staged folder, and reads back the
remote file list before returning `UPLOADED`.

By default the Hub packet contains the loadable Transformers model and
tokenizer export. Supplying `--trainer-checkpoint` additionally includes the
preflight-bound optimizer/RNG checkpoint as `trainer-state.pt`, permitting an
exact training resume at the cost of a substantially larger upload. The
publisher refuses a trainer checkpoint whose SHA-256 is not the checkpoint
examined by the preflight receipt.

## Model name

Generated model names are at most 96 characters:

```text
helix-55m-260902-2125-s512-l3-f3-r0p0002-e02-gadfb95e3
```

Legend:

- `55m`: rounded parameter count in millions
- `260902-2125`: UTC run timestamp (`YYMMDD-HHMM`)
- `s512`: sequence length
- `l3`: Helix loops
- `f3`: FFN expansion
- `r0p0002`: learning rate (`p` is the decimal point)
- `e02`: completed epoch represented by this checkpoint
- `gadfb95e3`: source commit prefix

## Example

First materialize and court the local model export:

```bash
python experiments/branch50-linear-context-v0/lighteval_checkpoint_preflight.py \
  --checkpoint /path/to/checkpoints/best-model.pt \
  --resolved-config /path/to/resolved_config.json \
  --output-dir /path/to/hf-export-epoch-01 \
  --receipt /path/to/hf-export-epoch-01.preflight.json
```

Then stage without network effects:

```bash
python experiments/branch50-linear-context-v0/publish_hf_checkpoint.py \
  --export-dir /path/to/hf-export-epoch-01 \
  --preflight-receipt /path/to/hf-export-epoch-01.preflight.json \
  --resolved-config /path/to/resolved_config.json \
  --stage-dir /path/to/hf-stage-epoch-01 \
  --publication-receipt /path/to/hf-stage-epoch-01.publication.json \
  --hf-namespace Thunderblok \
  --epoch 1 \
  --run-timestamp 260902-2125 \
  --source-head adfb95e372d3893e8bba254413699e208e972c60
```

Add `--upload` to perform the private Hub publication after `HF_TOKEN` is
present in the environment. Add `--public` only for an explicitly authorized
public release. Add `--trainer-checkpoint /path/to/checkpoint.pt` when the Hub
artifact must include exact-resume optimizer and RNG state rather than only the
loadable model export.

References:

- <https://huggingface.co/docs/huggingface_hub/en/guides/upload>
- <https://huggingface.co/docs/transformers/main/en/model_sharing>
