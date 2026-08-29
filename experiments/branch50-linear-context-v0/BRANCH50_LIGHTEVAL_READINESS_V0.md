# Branch 50 Lighteval Readiness V0

Status: `PREFLIGHT_PASS_LIGHEVAL_NOT_EXECUTED`

This record binds the evaluation-consumer boundary before a selected Branch 50
checkpoint exists. It is a proxy preflight over the completed 300M control
checkpoint, not an evaluation result and not a model-promotion decision.

## Bound checkpoint

```text
run=
branch50-ablation-control-s512-b12-a7-t300000000-20260829T020213Z

checkpoint=
checkpoints/best-model.pt

checkpoint_sha256=
5f8fdf5e69f672b7f349dba576270f06bc164b2fd75e3748567c0cb235013e3b

step=
6990

validation_loss=
4.360077083110809

declared_model_root=
004a6c0c763214b5c6d0821d9154ee1966b1398aab696829a08d004bf2bcb08c
```

## Preflight receipt

```text
receipt_path=
/home/mo/DEV/experiments/helix-branch50-linear-context-v0/artifacts/lighteval-preflight-v0/control300m-receipt.json

receipt_file_sha256=
215a1a52280dd6d77e4d613963aa8ab2aeebd86bbc3ec6ac1c398b423ea90f5a

receipt_root=
6854a0cae035e4eb353cfbf41169ed35a10650a6527948978ecf2f5161ade691

export_manifest_root=
06cecf8662852965d015244c54763a262aed607169884e6b53dae8a50d5cbad2
```

## Courts

```text
checkpoint_declared_root_matches_observed=PASS
export_reload_root_matches_checkpoint=PASS
fresh_process_reload=PASS
registered_source_required=true
trust_remote_code=false
parameter_count=53592340
tie_word_embeddings=true
tied_weight_alias_observed=true
tokenizer=gpt2
vocab_size=50257
bos_token_id=50256
eos_token_id=50256
pad_token_id=50256
loglikelihood_smoke=PASS
generation_smoke=PASS
publication_effect=none
```

The Transformers reload report lists `lm_head.weight` as missing because the
serialized safetensors file stores the shared embedding tensor once. This is
not accepted on that message alone. The preflight independently proves that
the reloaded input embedding and language-model head are aliased and that the
complete reloaded model root exactly matches the checkpoint root.

## Runtime

```text
device=cpu
python=3.14.2
torch=2.12.0.dev20260408+cu128
transformers=5.8.1
```

## Held claims

```text
lighteval_executed=false
benchmark_score=not_established
selected_checkpoint=not_established
publication_ready=false
```

The final selected checkpoint must repeat this preflight and then run the
frozen Lighteval task/config/CLI contract. Partial `max_samples` smoke results
must not be used for candidate comparison.
